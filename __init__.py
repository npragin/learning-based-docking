# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .bluerov2_env import BlueROV2Env, BlueROV2EnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-BlueROV2-Direct-v1",
    entry_point="omni.isaac.lab_tasks.direct.learning-based-control:BlueROV2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BlueROV2EnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.DefaultPPORunnerCfg
    },
)