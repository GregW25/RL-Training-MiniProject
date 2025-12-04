from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  UNITREE_GO2_FLAT_ENV_CFG,
)
from .rl_cfg import UNITREE_GO1_PPO_RUNNER_CFG

register_mjlab_task(
  task_id="Mjlab-Backflip-Flat-Unitree-Go2",
  env_cfg=UNITREE_GO2_FLAT_ENV_CFG,
  rl_cfg=UNITREE_GO1_PPO_RUNNER_CFG,
  runner_cls=VelocityOnPolicyRunner,
)
