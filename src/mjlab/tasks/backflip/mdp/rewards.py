"""Reward functions for the Backflip task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
# Import quat_apply_inverse to project world gravity into body frame
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ==============================================================================
# JIT Trajectory Generator (Internal Helper)
# ==============================================================================

@torch.jit.script
def _compute_backflip_ref(
    current_time: torch.Tensor,
    max_time: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes target height (z) and target PITCH angle based on phase.
    
    Returns:
        target_z (torch.Tensor): Target CoM height [N].
        target_pitch (torch.Tensor): Target pitch angle in radians [N].
    """
    # Calculate phase [0, 1]
    phase = torch.clamp(current_time / max_time, 0.0, 1.0)
    
    # === Keyframes Configuration ===
    # Phase points: Start, Squat, Takeoff, Apex, Pre-land, Land
    kp_phase = torch.tensor([0.0, 0.2, 0.3, 0.5, 0.8, 1.0], device=phase.device)
    
    # Target Height Z (meters)
    kp_z = torch.tensor([0.30, 0.20, 0.35, 0.90, 0.35, 0.28], device=phase.device)
    
    # Target Pitch (Radians): Rotating backwards from 0 to -2pi (-360 deg)
    kp_pitch = torch.tensor([0.0, 0.0, -0.2, -3.14, -5.8, -6.28], device=phase.device)
    
    # === Linear Interpolation ===
    target_z = torch.zeros_like(phase)
    target_pitch = torch.zeros_like(phase)
    
    # Vectorized piecewise linear interpolation
    for i in range(len(kp_phase) - 1):
        mask = (phase >= kp_phase[i]) & (phase < kp_phase[i+1])
        # Only compute if any envs are in this phase segment
        if mask.any():
            # alpha = (phi - p_start) / (p_end - p_start)
            alpha = (phase[mask] - kp_phase[i]) / (kp_phase[i+1] - kp_phase[i])
            target_z[mask] = kp_z[i] + alpha * (kp_z[i+1] - kp_z[i])
            target_pitch[mask] = kp_pitch[i] + alpha * (kp_pitch[i+1] - kp_pitch[i])
            
    # Handle end of episode (phase >= 1.0)
    mask_end = phase >= 1.0
    target_z[mask_end] = kp_z[-1]
    target_pitch[mask_end] = kp_pitch[-1]

    return target_z, target_pitch


# ==============================================================================
# Reward Functions
# ==============================================================================

def track_backflip_trajectory(
    env: ManagerBasedRlEnv,
    sigma_z: float,
    sigma_pitch: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for tracking the reference backflip trajectory (Height & Orientation).
    
    This function implements a Yaw-Invariant orientation reward. Instead of matching 
    quaternions directly (which would force the robot to face a specific Yaw), 
    we match the 'Projected Gravity Vector'.
    """
    asset: Entity = env.scene[asset_cfg.name]
    
    # 1. Get current time and compute reference trajectory
    current_time = env.episode_length_buf * env.step_dt
    max_time = env.max_episode_length * env.step_dt
    
    target_z, target_pitch = _compute_backflip_ref(current_time, max_time)

    # 2. Height Reward
    # Using 'root_link_pos_w' from EntityData class
    actual_z = asset.data.root_link_pos_w[:, 2]
    z_error = torch.square(actual_z - target_z)
    r_z = torch.exp(-z_error / (sigma_z**2))

    # 3. Orientation Reward (Projected Gravity Matching)
    # -----------------------------------------------------------------------
    # Goal: Construct the "ideal body-frame gravity vector".
    # Since gravity is Z-axis symmetric, we don't care about Yaw.
    # -----------------------------------------------------------------------
    
    # A. Construct target Quaternion for pure Pitch rotation.
    # Euler to Quat (Y-axis rotation): w = cos(theta/2), y = sin(theta/2)
    half_theta = target_pitch * 0.5
    target_pitch_quat = torch.zeros((len(target_pitch), 4), device=env.device)
    target_pitch_quat[:, 0] = torch.cos(half_theta) # w
    target_pitch_quat[:, 2] = torch.sin(half_theta) # y (rotation around Body Y)
    
    # B. Compute "Target Projected Gravity"
    # What should the gravity vector look like in the body frame if the pitch is correct?
    # World gravity is [0, 0, -1].
    gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    target_proj_gravity = quat_apply_inverse(target_pitch_quat, gravity_vec_w)

    # C. Compute "Actual Projected Gravity"
    # Use the robot's actual quaternion (which includes arbitrary Yaw).
    actual_quat = asset.data.root_link_quat_w
    actual_proj_gravity = quat_apply_inverse(actual_quat, gravity_vec_w)

    # D. Compute Error (Cosine Similarity)
    # If the robot has the correct Pitch and Roll, these two vectors will align,
    # regardless of the robot's Yaw.
    # Dot product: 1.0 means perfect match.
    dot_prod = torch.sum(target_proj_gravity * actual_proj_gravity, dim=-1)
    
    # Error is 0 when dot product is 1. Formula: (dot - 1.0)^2
    orientation_error = torch.square(dot_prod - 1.0)
    
    r_pitch = torch.exp(-orientation_error / (sigma_pitch**2))

    # Combine rewards
    return r_z * r_pitch


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize large changes in actions (action rate)."""
    curr_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    return torch.sum(torch.square(curr_action - prev_action), dim=1)


def joint_pos_limits(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits."""
    asset: Entity = env.scene[asset_cfg.name]
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    assert soft_joint_pos_limits is not None
    
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize large action magnitudes (proxy for energy/torque)."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)