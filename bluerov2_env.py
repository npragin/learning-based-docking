from __future__ import annotations

import os
import random
import math
import torch
from collections.abc import Sequence

from .assets.bluerov2_heavy import BLUEROV2_HEAVY_CFG, BLUEROV2_HEAVY_THRUSTER_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg, AssetBaseCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform, normalize
from omni.isaac.lab.markers import CUBOID_MARKER_CFG, VisualizationMarkers, RED_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils.math import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul, quat_error_magnitude
import omni.isaac.lab.utils.math as math_utils

##
# Hydrodynamic model
##
from omni.isaac.lab.utils.math import quat_apply, quat_conjugate
from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .thruster_dynamics import DynamicsFirstOrder, ConversionFunctionBasic, get_thruster_com_and_orientations
from .base_auv_env import BaseAUVSceneCfg, AUVEnvWindow, BaseAUVEnvCfg, BaseAUVEnv

DS_POS = (-3.0, 0.0, 5.0)
DS_ROT = (0.707, 0, 0, 0.707)
DS_SCALE = (1.4, 1.4, 1.4)
DS_GOAL_OFFSET = (0.3, 0.0, 0.0)

@configclass
class DockingSceneCfg(BaseAUVSceneCfg):
    robot: RigidObjectCfg = BLUEROV2_HEAVY_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    docking_station_bot: RigidObjectCfg = RigidObjectCfg(
        prim_path = "/World/envs/env_.*/DockingStationBottom",
        spawn = sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "data/docking/dsbot/dsbot.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, 
                solver_position_iteration_count=4, 
                solver_velocity_iteration_count=0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            scale=DS_SCALE
        ),
        init_state = RigidObjectCfg.InitialStateCfg(pos=DS_POS, rot=DS_ROT)
    )

    docking_station_top: RigidObjectCfg = RigidObjectCfg(
        prim_path = "/World/envs/env_.*/DockingStationTop",
        spawn = sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "data/docking/dstop/dstop.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, 
                solver_position_iteration_count=4, 
                solver_velocity_iteration_count=0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            scale=DS_SCALE
        ),
        init_state = RigidObjectCfg.InitialStateCfg(pos=DS_POS, rot=DS_ROT)
    )

    docking_station_left: RigidObjectCfg = RigidObjectCfg(
        prim_path = "/World/envs/env_.*/DockingStationLeft",
        spawn = sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "data/docking/dsleft/dsleft.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, 
                solver_position_iteration_count=4, 
                solver_velocity_iteration_count=0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            scale=DS_SCALE
        ),
        init_state = RigidObjectCfg.InitialStateCfg(pos=DS_POS, rot=DS_ROT)
    )

    docking_station_right: RigidObjectCfg = RigidObjectCfg(
        prim_path = "/World/envs/env_.*/DockingStationRight",
        spawn = sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "data/docking/dsright/dsright.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, 
                solver_position_iteration_count=4, 
                solver_velocity_iteration_count=0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            scale=DS_SCALE
        ),
        init_state = RigidObjectCfg.InitialStateCfg(pos=DS_POS, rot=DS_ROT)
    )

    docking_station_back: RigidObjectCfg = RigidObjectCfg(
        prim_path = "/World/envs/env_.*/DockingStationBack",
        spawn = sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "data/docking/dsback/dsback.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, 
                solver_position_iteration_count=4, 
                solver_velocity_iteration_count=0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            scale=DS_SCALE
        ),
        init_state = RigidObjectCfg.InitialStateCfg(pos=DS_POS, rot=DS_ROT)
    )

@configclass
class BlueROV2EnvCfg(BaseAUVEnvCfg):
    # mdp
    num_actions = 8

    # spawn
    spawn_x_range = 1
    spawn_y_range = 1
    spawn_z_range = 1

    # scene
    scene = DockingSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)

    # rewards
    rew_scale_terminated = 0.0
    rew_scale_alive = 0.0
    rew_scale_completion = 0.0
    rew_scale_pos = 0.2
    rew_scale_energy = 0.0
    rew_scale_jerk = 0.0
    rew_scale_successful_dock = 0.0
    rew_scale_orientation = 0.03

    # observation history
    history_length = 1

    # successful dock definition
    successful_dock_params = {
        "max_docking_velocity": 0.5,             # meters per second
        "docking_orientation_threshold": 0.52,  # radians - 30 degrees
        "docking_distance_threshold": 0.2       # meters
    }

    # dynamics
    rotor_constant = 0.1 / 200.0
    thruster_axis = 2
    volume = 0.039

    max_wave_amplitude = 0
    max_current_force = 0

    class domain_randomization:
        use_custom_randomization = True

        mass_range = [11.5, 11.5]
        com_to_cob_offset_radius = 0
        volume_range = [0.008, 0.016]

class BlueROV2Env(BaseAUVEnv):
    cfg: BlueROV2EnvCfg

    def __init__(self, cfg: BlueROV2EnvCfg, render_mode: str | None = None, **kwargs):
        self._prev_lin_vel = None
        super().__init__(cfg, render_mode, **kwargs)

    def _get_thruster_cfg(self):
        return BLUEROV2_HEAVY_THRUSTER_CFG

    def _get_reward_func(self):
        return _compute_rewards
    
    def _collect_reward_info(self):
        offset_from_goal_b = quat_apply(quat_conjugate(self._robot.data.root_quat_w), (self._goal + self.scene.env_origins) - self._robot.data.root_pos_w)

        dt = self.sim.cfg.dt * self.cfg.decimation

        auv_target_quat = torch.tensor([1, 0, 0, 0], device=self.device).repeat(self.num_envs, 1)

        quat_diff = quat_mul(quat_conjugate(self._robot.data.root_quat_w), auv_target_quat)
        cos_half_angle = torch.abs(quat_diff[:, 0])
        orientation_diff = 2.0 * torch.acos(torch.clamp(cos_half_angle, 0.0, 1.0))

        auv_orientation_error = quat_error_magnitude(auv_target_quat, self._robot.data.root_quat_w)

        # Calculate velocity
        curr_lin_vel = self._robot.data.root_lin_vel_w
        curr_lin_vel_magnitude = torch.norm(curr_lin_vel, dim=1)

        # Calculate velocity
        curr_lin_acc = (curr_lin_vel - self._prev_lin_vel) / dt
        curr_lin_acc_magnitude = torch.norm(curr_lin_acc, dim=1)

        # Calculate jerk, 0 when first step
        jerk = (curr_lin_acc - self._prev_lin_acc) / dt
        jerk_magnitude = torch.norm(jerk, dim=1)
        jerk_magnitude = torch.where(self._first_step, torch.zeros_like(jerk_magnitude), jerk_magnitude)

        # Calculate action cost
        action_cost = torch.norm(self._actions, dim=1)

        ### Update evaluation metrics
        curr_position = self._robot.data.root_pos_w
        distance_traveled = torch.norm(curr_position - self._prev_position, dim=1)

        close_to_dock = torch.norm(offset_from_goal_b, dim=1, p=2) < self.cfg.successful_dock_params["docking_distance_threshold"]
        slow_to_dock = curr_lin_vel_magnitude < self.cfg.successful_dock_params["max_docking_velocity"]
        oriented_to_dock = orientation_diff < self.cfg.successful_dock_params["docking_orientation_threshold"]
        successful_dock = torch.logical_and(close_to_dock, slow_to_dock).float()
        # successful_dock = close_to_dock.float()

        self._evaluation_metrics[:, 0] = torch.logical_or(self._evaluation_metrics[:, 0], successful_dock)
        self._evaluation_metrics[:, 1] += curr_lin_acc_magnitude
        self._evaluation_metrics[:, 2] += distance_traveled
        self._evaluation_metrics[:, 3] += dt
        self._evaluation_metrics[:, 4] += jerk_magnitude
        self._evaluation_metrics[:, 5] += action_cost

        # Update members
        self._prev_lin_vel.copy_(curr_lin_vel)
        self._prev_lin_acc.copy_(curr_lin_acc)
        self._prev_position.copy_(curr_position)
        self._first_step.fill_(False)

        return {
            "rew_scale_pos": self.cfg.rew_scale_pos,
            "rew_scale_orientation": self.cfg.rew_scale_orientation,
            "rew_scale_jerk": self.cfg.rew_scale_jerk,
            "rew_scale_energy": self.cfg.rew_scale_energy,
            "rew_scale_successful_dock": self.cfg.rew_scale_successful_dock,
            "offset_from_goal": offset_from_goal_b,
            "acc_magnitude": curr_lin_acc_magnitude,
            "jerk_magnitude": jerk_magnitude,
            "successful_dock": successful_dock,
            "orientation_error": auv_orientation_error
        }

    def _reset_agent(self, env_ids: Sequence[int]):
        if self._prev_lin_vel == None:
            # For jerk calculation and other stuff
            self._prev_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
            self._prev_lin_acc = torch.zeros(self.num_envs, 3, device=self.device)
            self._first_step = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

            self._prev_position = torch.zeros(self.num_envs, 3, device=self.device)

            # [successful_dock: bool, energy_consumed, path_length, travel_time, jerk, action_cost]
            self._evaluation_metrics = torch.zeros(self.num_envs, 6, device=self.device)

        self._default_root_state[env_ids, :] = self._robot.data.default_root_state[env_ids]
        self._default_root_state[env_ids, :3] += self.scene.env_origins[env_ids]

        self._default_root_state[env_ids, 0] += torch.empty(len(env_ids), device=self.device).uniform_(-1 * self.cfg.spawn_x_range, self.cfg.spawn_x_range)
        self._default_root_state[env_ids, 1] += torch.empty(len(env_ids), device=self.device).uniform_(-1 * self.cfg.spawn_y_range, self.cfg.spawn_y_range)
        self._default_root_state[env_ids, 2] += torch.empty(len(env_ids), device=self.device).uniform_(-1 * self.cfg.spawn_z_range, self.cfg.spawn_z_range)

        self._prev_lin_vel[env_ids] = 0
        self._prev_lin_acc[env_ids] = 0
        self._first_step[env_ids] = True

        # Reset evaluation metrics
        self._evaluation_metrics[env_ids] = 0

        self._prev_position[env_ids] = self._robot.data.root_pos_w[env_ids]

    def _reset_goal(self, env_ids: Sequence[int]):
        self._goal[env_ids, :] = torch.tensor(list((x + y for (x, y) in zip(DS_POS, DS_GOAL_OFFSET))), device=self.device)

@torch.jit.script
def _compute_rewards(
    rew_scale_pos: float,
    rew_scale_orientation: float,
    rew_scale_energy: float,
    rew_scale_jerk: float,
    rew_scale_successful_dock: float,
    offset_from_goal: torch.Tensor,
    acc_magnitude: torch.Tensor,
    jerk_magnitude: torch.Tensor,
    successful_dock: torch.Tensor,
    orientation_error: torch.Tensor
):
    rew_pos = rew_scale_pos * torch.exp(-1 * (torch.norm(offset_from_goal, dim=1) ** 2))

    rew_energy = rew_scale_energy * torch.exp(-1 * acc_magnitude)

    rew_jerk = -rew_scale_jerk * jerk_magnitude

    rew_successful_dock = rew_scale_successful_dock * successful_dock

    rew_orientation = rew_scale_orientation * torch.exp(-1 * orientation_error)

    total_rew = rew_pos + rew_energy + rew_jerk + rew_successful_dock + rew_orientation
    return total_rew
