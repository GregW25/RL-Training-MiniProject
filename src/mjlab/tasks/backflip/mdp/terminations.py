"""Termination functions for the Backflip task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def backflip_early_fall(
    env: ManagerBasedRlEnv,
    height_threshold: float = 0.25,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """
    Terminate if the robot falls (height < threshold) during the mid-air phase.

    This prevents the robot from simply lying on the ground to collect 'survival rewards'
    if it has already failed the flip. It accelerates training by resetting failed attempts immediately.
    """
    # 1. Retrieve the robot entity from the scene
    asset: Entity = env.scene[asset_cfg.name]

    # 2. Calculate the current phase normalized to [0, 1]
    # 'episode_length_buf' stores the current step count
    current_time = env.episode_length_buf * env.step_dt
    # 'max_episode_length' stores the max steps allowed per episode
    max_time = env.max_episode_length * env.step_dt
    
    # Phase 0.0 = Start, Phase 1.0 = End
    phase = current_time / max_time

    # 3. Define the "Flight Window"
    # We expect the robot to be in the air roughly between 30% and 80% of the motion.
    # - Before 0.3: The robot is squatting to jump (valid low height).
    # - After 0.8: The robot is landing and absorbing impact (valid low height).
    # We only penalize low height strictly during the flight phase.
    in_flight_window = (phase > 0.3) & (phase < 0.8)

    # 4. Check if the CoM height is below the threshold
    # root_link_pos_w shape is [Num_Envs, 3], where index 2 is the Z-axis (height).
    current_height = asset.data.root_link_pos_w[:, 2]
    is_too_low = current_height < height_threshold

    # 5. Determine termination
    # Terminate ONLY if the robot is too low AND it is currently in the flight window.
    should_terminate = in_flight_window & is_too_low

    return should_terminate

def illegal_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return torch.any(sensor.data.found, dim=-1)
