"""Backflip task configuration."""

import math
from copy import deepcopy

from mjlab.entity.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    ActionTermCfg,
    CommandTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
# Change import to your backflip specific mdp
from mjlab.tasks.backflip import mdp 
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# Backflip is best learned on flat ground initially
SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(
        terrain_type="plane", # Changed from generator/rough to plane
        terrain_generator=None,
        max_init_terrain_level=0,
    ),
    num_envs=1,
    extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="",  # Override in robot cfg.
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
)

SIM_CFG = SimulationCfg(
    nconmax=35,
    njmax=300,
    mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
    ),
)


def create_backflip_env_cfg(
    robot_cfg: EntityCfg,
    action_scale: float | dict[str, float],
    viewer_body_name: str,
    site_names: tuple[str, ...],
    feet_sensor_cfg: ContactSensorCfg,
    self_collision_sensor_cfg: ContactSensorCfg,
    foot_friction_geom_names: tuple[str, ...] | str,
    posture_std_initial: dict[str, float], # Renamed for clarity (only one initial pose)
) -> ManagerBasedRlEnvCfg:
    """Create a backflip task configuration."""
    
    scene = deepcopy(SCENE_CFG)
    scene.entities = {"robot": robot_cfg}
    scene.sensors = (feet_sensor_cfg, self_collision_sensor_cfg)

    viewer = deepcopy(VIEWER_CONFIG)
    viewer.body_name = viewer_body_name

    # ---------------------------------------------------------------------------
    # Actions
    # ---------------------------------------------------------------------------
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=(".*",),
            scale=action_scale,
            use_default_offset=True,
        )
    }

    # ---------------------------------------------------------------------------
    # Commands
    # ---------------------------------------------------------------------------
    # Backflip is not a command-driven task in the traditional sense (no joystick).
    # We leave this empty or minimal.
    commands: dict[str, CommandTermCfg] = {}

    # ---------------------------------------------------------------------------
    # Observations
    # ---------------------------------------------------------------------------
    policy_terms: dict[str, ObservationTermCfg] = {
        # [Crucial] The network MUST know the current phase (0.0 to 1.0)
        "phase": ObservationTermCfg(
            func=mdp.phase_observation, # You must define this in your mdp.py
            noise=None, # Phase is an internal clock, usually noiseless
        ),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        # Removed "command" observation
    }

    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        # Critic usually sees the same as policy in simple tasks, 
        # or priviledged info like exact contact forces if needed.
        "critic": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # ---------------------------------------------------------------------------
    # Events (Reset & Randomization)
    # ---------------------------------------------------------------------------
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                # Tighter reset range for backflip. 
                # If it spawns tilted, it might fall before jumping.
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.01, 0.01),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),
                "operation": "abs",
                "field": "geom_friction",
                "ranges": (0.6, 1.2), # High friction helps with jumping
            },
        ),
        # Removed "push_robot" - pushing a robot mid-air during a flip 
        # usually makes it unlearnable in early stages.
    }

    # ---------------------------------------------------------------------------
    # Rewards
    # ---------------------------------------------------------------------------
    rewards = {
        # --- Task Objective ---
        "track_trajectory": RewardTermCfg(
            func=mdp.track_backflip_trajectory, # Defined in your mdp.py
            weight=10.0,
            params={
                "sigma_z": 0.05,     # Sharp penalty for height error
                "sigma_pitch": 0.2,  # Penalty for rotation error
            },
        ),

        # --- Regularization ---
        "action_rate": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.01, # Reduced weight, explosive motion requires fast actions
        ),
        "dof_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-1.0,
        ),
        "torque_penalty": RewardTermCfg( # Optional: minimize energy
            func=mdp.action_l2, # Using action L2 as proxy for torque/effort
            weight=-0.005,
        ),
        
        # Removed upright, flat_orientation, and gait rewards.
        # We WANT to rotate, so flat_orientation would penalize the flip.
    }

    # ---------------------------------------------------------------------------
    # Terminations
    # ---------------------------------------------------------------------------
    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        # Removed "fell_over" / "bad_orientation".
        # During a backflip, the robot goes upside down. 
        # Standard checks would terminate the episode immediately.
        
        # Optional: Add a check for "early failure"
        # e.g., if height < 0.15m during the middle of the flip (Phase 0.2-0.8)
        "early_fall": TerminationTermCfg(
             func=mdp.backflip_early_fall, # You would need to define this
             time_out=False,
             params={"height_threshold": 0.25}
        )
    }

    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        events=events,
        sim=SIM_CFG,
        viewer=viewer,
        decimation=4,
        # IMPORTANT: Backflip is a short, ballistic motion.
        # 20s is way too long. 1.0s to 1.5s is standard.
        episode_length_s=1.2, 
    )