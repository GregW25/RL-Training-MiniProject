"""Unitree Go2 backflip task configurations."""

from copy import deepcopy

from mjlab.asset_zoo.robots import (
    GO2_ACTION_SCALE,   # Ensure this exists in your asset zoo
    get_go2_robot_cfg,  # Ensure this exists in your asset zoo
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
# Import your backflip task logic instead of velocity
from mjlab.tasks.backflip import mdp
from mjlab.tasks.backflip.backflip_env_cfg import VIEWER_CONFIG, create_backflip_env_cfg
from mjlab.utils.retval import retval


@retval
def UNITREE_GO2_FLAT_ENV_CFG() -> ManagerBasedRlEnvCfg:
    """Create Unitree Go2 backflip task configuration."""
    foot_names = ("FR", "FL", "RR", "RL")
    site_names = ("FR", "FL", "RR", "RL")
    # Go2 usually follows similar naming conventions, but verify geoms in XML
    geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

    # 1. Define Sensors
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        # Pattern matching for foot collision geoms
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            # Grab all collision geoms...
            pattern=r".*_collision\d*$",
            # Except for the foot geoms.
            exclude=tuple(geom_names),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # 2. Create the base configuration using your backflip-specific factory
    cfg = create_backflip_env_cfg(
        robot_cfg=get_go2_robot_cfg(),
        action_scale=GO2_ACTION_SCALE,
        viewer_body_name="base",  # Go2 root link is often named 'base' or 'trunk'
        site_names=site_names,
        feet_sensor_cfg=feet_ground_cfg,
        self_collision_sensor_cfg=nonfoot_ground_cfg,
        foot_friction_geom_names=geom_names,
        # Posture randomization might be tighter for backflip initialization
        posture_std_initial={
            r".*_hip_joint.*": 0.05,
            r".*_thigh_joint.*": 0.05,
            r".*_calf_joint.*": 0.05,
        },
    )

    # 3. Viewer Setup
    cfg.viewer = deepcopy(VIEWER_CONFIG)
    cfg.viewer.body_name = "base"
    cfg.viewer.distance = 2.0  # Increased distance to see the full flip
    cfg.viewer.elevation = -15.0

    # 4. Terminations
    assert cfg.terminations is not None
    
    # Terminate if non-foot parts touch ground (failed flip)
    # Note: You might want to disable this during early training phases
    # or rely on the Phase variable to determine if this is allowed (e.g. landing on knees)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "nonfoot_ground_touch"},
    )

    return cfg