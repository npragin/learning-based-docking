from __future__ import annotations

from abc import ABC, abstractmethod

import os
import random
import math
import torch
from collections.abc import Sequence
from dataclasses import field

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg, AssetBaseCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.sim.schemas import define_mass_properties, MassPropertiesCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform, normalize
from omni.isaac.lab.markers import CUBOID_MARKER_CFG, VisualizationMarkers, RED_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils.math import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul
import omni.isaac.lab.utils.math as math_utils

##
# Hydrodynamic model
##
from omni.isaac.lab.utils.math import quat_apply, quat_conjugate
from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .thruster_dynamics import DynamicsFirstOrder, ConversionFunctionBasic, get_thruster_com_and_orientations

class AUVEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class BaseAUVSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path = "/World/groundPlane", spawn = sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path = "/World/Light", spawn = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    @property
    def robot(self):
        pass

@configclass
class BaseAUVEnvCfg(DirectRLEnvCfg):
    ui_window_class_type = AUVEnvWindow

    sim: SimulationCfg = SimulationCfg(dt=1 / 120)

    debug_vis = True

    # mdp
    num_actions = 6
    num_observations = 7
    goal_dims = 3
    history_length = 1

    # training and eval
    cap_episode_length = True
    episode_length_s = 3.0
    episode_length_before_reset = None
    eval_mode = False

    # initial conditions
    starting_depth = 5

    # boundaries
    use_boundaries = True
    max_auv_x = 5
    max_auv_y = 5
    max_auv_z = 5

    # simulation
    decimation = 2

    # dynamics
    thruster_axis = 0
    min_wave_frequency = 50
    max_wave_frequency = 100
    max_wave_amplitude = 100
    max_current_force = 50
    water_rho = 997.0 # kg/m^3
    water_beta = 0.001306 # Pa s, dynamic viscosity of water @ 50 deg F
    rotor_constant = 0.1 / 100.0 # rotor constant used in Gazebo, note /10 because 0.04 is "10x bigger than it should be"
    dyn_time_constant = 0.05 # time constant for linear dynamics for each rotor 
    com_to_cob_offset = [0.0, 0.0, 0.01] # in meters, add this (xyz) to COM to get COB location
    volume = 0.022747843530591776 # assuming cubic meters - NEUTRALLY BOUYANT. In orignal sim file volume = 0.0223

    # domain randomization
    # todo: isaaclabs has a built-in method somehow
    class domain_randomization:
        use_custom_randomization = False
        # com_to_cob_offset_radius = 0 # uniform from sphere around predicted com_to_cob_offset
        # volume_range = [0.022747843530591776, 0.022747843530591776] # uniform [lowerbound, upperbound]
        mass_range = [13, 13.5] # uniform [lowerbound, upperbound]
        com_to_cob_offset_radius = 0.5 # uniform from sphere around predicted com_to_cob_offset
        volume_range = [0.019747843530591773, 0.02574784353059178] # uniform [loierbound, upperbound]

class BaseAUVEnv(DirectRLEnv):
    cfg: BaseAUVEnvCfg

    def __init__(self, cfg: BaseAUVEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Debug mode?
        self._debug = False

        # Initialize buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._goal = torch.zeros(self.num_envs, self.cfg.goal_dims, device=self.device)
        self._default_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._completion_buffer = torch.zeros(self.num_envs, device=self.device)
        self._completed_envs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._default_env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_pos_w = self._default_env_origins # just for visualizations at the moment
        self._env_step_count = torch.zeros(self.num_envs, device=self.device)
        self._step_count = 0
        self._current_directions = torch.zeros(self.num_envs, 3, device=self.device)
        self._current_forces = torch.zeros(self.num_envs, device=self.device)
        self._wave_frequencies = torch.zeros(self.num_envs, device=self.device)
        self._wave_amplitudes = torch.zeros(self.num_envs, device=self.device)
        self._obs_history = torch.zeros(
            (self.num_envs, self.cfg.history_length, self.cfg.num_observations),
            device=self.device
        )
        self._action_history = torch.zeros(
            (self.num_envs, self.cfg.history_length, self.cfg.num_actions),
            device=self.device
        )
        
        # Get thruster configurations
        self.thruster_com_offsets, self.thruster_quats = get_thruster_com_and_orientations(self.device, self._get_thruster_cfg())
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.thruster_quats = self.thruster_quats.repeat(self.num_envs, 1)

        torch.manual_seed(0)

        if self.cfg.eval_mode:
            print("Setting manual seed")
            torch.manual_seed(0)

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        if self._debug: print("mass: ", list(self._robot.root_physx_view._masses))

        # Get specific information about the AUV
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        # todo: get inertias from the model or physx view
        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float, requires_grad=False)
        self.default_inertia = self._robot.root_physx_view.get_inertias()

        self.inertia_tensors[:, 0] = self.default_inertia[0][0]
        self.inertia_tensors[:, 1] = self.default_inertia[0][4]
        self.inertia_tensors[:, 2] = self.default_inertia[0][8]

        # todo: cleaner way to handle this
        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
            self.com_to_cob_offsets = torch.tensor(self.cfg.com_to_cob_offset).repeat(self.num_envs, 1).to(self.device)
        else:
            self.com_to_cob_offsets = self.cfg.com_to_cob_offset.copy()

        if type(self.cfg.volume) != torch.Tensor:
            self.volumes = torch.full((self.num_envs, 1), self.cfg.volume, device=self.device)
        else:
            self.volumes = self.cfg.volume.copy()

        self.inertia_tensors_mean = self.inertia_tensors.mean(dim=1, keepdim=True) 

        # Initialize dynamics calculators
        self._init_thruster_dynamics()
        
        # Set initial goals
        self._reset_idx(self._robot._ALL_INDICES)

    def _get_thruster_cfg(self):
        raise NotImplementedError

    def _init_thruster_dynamics(self):
        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
          self.cfg.com_to_cob_offset = torch.tensor(self.cfg.com_to_cob_offset, device=self.device, dtype=torch.float32, requires_grad=False).reshape(1,3).repeat(self.num_envs, 1)

        # get force calculation functions and rotor dynamics models
        self.force_calculation_functions = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.thruster_dynamics = DynamicsFirstOrder(self.num_envs, self.cfg.num_actions, self.cfg.dyn_time_constant, self.device)
        self.thruster_conversion = ConversionFunctionBasic(self.cfg.rotor_constant)

    def _setup_scene(self):
        self.cfg.scene.robot.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.starting_depth))
        self._robot = RigidObject(self.cfg.scene.robot)

        self.scene.articulations["robot"] = self._robot

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._debug: print("original actions vec: ", actions)
        if self._debug: print("concatenated actions shape: ", self._actions)

        self._actions[:] = actions
        self._actions[:] = torch.clip(self._actions, -1, 1).to(self.device)

        self._action_history = torch.roll(self._action_history, shifts=-1, dims=1)
        self._action_history[:, -1, :] = self._actions

    def _apply_action(self) -> None:
        temp = self._compute_dynamics(self._actions)
        self._thrust[:,0,:], self._moment[:,0,:] = temp
        self._robot.set_external_force_and_torque(self._thrust, self._moment)

    def _get_observations(self) -> dict:
        #desired_pos_b = quat_apply(quat_conjugate(self._robot.data.root_quat_w), self._goal - self._robot.data.root_pos_w)
        offset_from_goal_b = quat_apply(quat_conjugate(self._robot.data.root_quat_w), (self._goal + self.scene.env_origins) - self._robot.data.root_pos_w)

        # Uniquefy and normalize all quaternions
        # goal = self._goal
        # root_quat_w = self._robot.data.root_quat_w
        # goal = math_utils.normalize(math_utils.quat_unique(self._goal))
        # root_quat_w = math_utils.normalize(math_utils.quat_unique(self._robot.data.root_quat_w))

        curr_obs = torch.cat(
            [
                offset_from_goal_b,
                self._robot.data.root_quat_w
            ],
            dim=-1
        )

        assert curr_obs.size() == torch.Size([self.num_envs, self.cfg.num_observations])

        self._obs_history = torch.roll(self._obs_history, shifts=-1, dims=1)
        self._obs_history[:, -1, :] = curr_obs

        obs_vector_length = self.cfg.history_length * (self.cfg.num_observations + self.cfg.num_actions)
        obs_action_vector = torch.zeros(
            (self.num_envs, obs_vector_length),
            device=self.device
        )
        
        for i in range(self.cfg.history_length):
            obs_start = i * (self.cfg.num_observations + self.cfg.num_actions)
            obs_action_vector[:, obs_start:obs_start + self.cfg.num_observations] = self._obs_history[:, i, :]
        
            action_start = obs_start + self.cfg.num_observations
            obs_action_vector[:, action_start:action_start + self.cfg.num_actions] = self._action_history[:, i, :]

        flattened_history = obs_action_vector.reshape(self.num_envs, -1)

        observations = {"policy": flattened_history}
        return observations

    def _get_reward_func(self):
        raise NotImplementedError

    def _collect_reward_info(self):
        raise NotImplementedError

    def _get_rewards(self) -> torch.Tensor:
        total_reward = (self._get_reward_func())(**(self._collect_reward_info()))

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.cap_episode_length:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs)

        self._step_count = self._step_count + 1
        self._env_step_count = self._env_step_count + 1

        if self.cfg.episode_length_before_reset:
            if self._step_count == self.cfg.episode_length_before_reset:
                time_out = torch.ones(self.num_envs)

        if self.cfg.use_boundaries:
            out_of_bounds = (
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > self.cfg.max_auv_x) | 
                (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > self.cfg.max_auv_y) | 
                (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._reset_agent(env_ids)

        self._env_step_count[env_ids] = 0
        self._step_count = 0
        
        # Apply domain randomization
        self._reset_domain(env_ids)

        # Reset goals
        self._reset_goal(env_ids)

        self._robot.write_root_pose_to_sim(self._default_root_state[env_ids, :7], env_ids)
        self._robot.write_root_velocity_to_sim(self._default_root_state[env_ids, 7:], env_ids)
        self._obs_history[env_ids] = 0.0
        self._action_history[env_ids] = 0.0


    def _reset_agent(self, env_ids: Sequence[int]):
        raise NotImplementedError

    # OVERRIDE THIS FUNC TO CHANGE GOAL
    def _reset_goal(self, env_ids: Sequence[int]):
        # Get random orientation
        # self._goal[env_ids, 0:4] = math_utils.random_orientation(len(env_ids), device=self.device)

        self._goal[env_ids, :] = torch.tensor(list((x + y for (x, y) in zip(DS_POS, DS_GOAL_OFFSET))), device=self.device)

        # Get random yaw orientation with 0 pitch and roll
        # self._goal[env_ids,0:4] = math_utils.random_yaw_orientation(len(env_ids), device=self.device)

        # Get fix RPY
        # rs = torch.zeros(len(env_ids), device=self.device) + 0.0
        # ps = torch.zeros(len(env_ids), device=self.device) + 0.0
        # ys = torch.zeros(len(env_ids), device=self.device) + 0.0
        # self._goal[env_ids,0:4] = math_utils.quat_from_euler_xyz(rs, ps, ys)

    def _reset_domain(self, env_ids: Sequence[int]):
        # Set wave forces
        self._wave_frequencies[env_ids] = self.cfg.min_wave_frequency + ((self.cfg.max_wave_frequency - self.cfg.min_wave_frequency) * torch.rand(len(env_ids), device=self.device))
        self._wave_amplitudes[env_ids] = self.cfg.max_wave_amplitude * torch.rand(len(env_ids), device=self.device)

        # Set current forces
        self._current_directions[env_ids] = self._sample_from_sphere(len(env_ids), 1)
        self._current_forces[env_ids] = self.cfg.max_current_force * torch.rand(len(env_ids), device=self.device)

        # Randomize COM to COB offset
        if self.cfg.domain_randomization.use_custom_randomization:
            self.com_to_cob_offsets[env_ids] = self.cfg.com_to_cob_offset[env_ids] + self._sample_from_sphere(len(env_ids), self.cfg.domain_randomization.com_to_cob_offset_radius)

        # Randomize volume
        if self.cfg.domain_randomization.use_custom_randomization:
            vol_lower, vol_upper = self.cfg.domain_randomization.volume_range
            self.volumes[env_ids] = math_utils.sample_uniform(vol_lower, vol_upper, self.volumes[env_ids].shape, self.device)

        if self.cfg.domain_randomization.use_custom_randomization:
            # Randomize mass
            env_ids = env_ids.clone().cpu()
            body_ids = torch.tensor([0], dtype=torch.int, device="cpu")
            masses = self._robot.root_physx_view.get_masses()
            # masses[env_ids[:, None], body_ids] = self._robot.data.default_mass[env_ids[:, None], body_ids].clone()
            masses = torch.empty(masses.shape[0], 1).uniform_(self.cfg.domain_randomization.mass_range[0], self.cfg.domain_randomization.mass_range[1])
            self._robot.root_physx_view.set_masses(masses, env_ids)

            # update inertia
            ratios = masses[env_ids[:, None], body_ids] / self._robot.data.default_mass[env_ids[:, None], body_ids]
            inertias = self._robot.root_physx_view.get_inertias()
            inertias[env_ids] = self.default_inertia[env_ids] * ratios

            self.inertia_tensors[:, 0] = inertias[0][0]
            self.inertia_tensors[:, 1] = inertias[0][4]
            self.inertia_tensors[:, 2] = inertias[0][8]

    def _sample_from_circle(self, num_env_ids, r):
        sampled_radius = r * torch.sqrt(torch.rand((num_env_ids), device=self.device))
        sampled_theta = torch.rand((num_env_ids), device=self.device) * 2 * 3.14159
        sampled_x = sampled_radius * torch.cos(sampled_theta)
        sampled_y = sampled_radius * torch.sin(sampled_theta)
        return (sampled_x, sampled_y)

    def _sample_from_sphere(self, num_env_ids, r):
        coords = torch.randn((num_env_ids, 3), device=self.device)
        norms = torch.norm(coords, dim=1).unsqueeze(1)
        coords /= norms

        radii = r * torch.pow(torch.rand((num_env_ids, 1), device=self.device), 1/3)

        return radii * coords

    def _compute_dynamics(self, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute dynamics from actions.
        actions are -1 for full reverse thrust, 1 for full forward thrust - THESE REPRESENT PWM VALUES
        BASED ON LINE 91 of https://gitlab.com/warplab/ros/warpauv/warpauv_simulation/-/blob/master/src/robot_sim_interface.py
        Args:
            actions (torch.Tensor): Actions shape (num_envs, num_actions)

        Returns:
            [torch.Tensor]: Forces sent to the simulation
            [torch.Tensor]: Torques sent to the simulation
        """

        if self._debug: print("actions: ", actions)

        thruster_forces = torch.zeros((self.num_envs, self.cfg.num_actions, 3), device=self.device, dtype=torch.float)
        thruster_torques = torch.zeros((self.num_envs, self.cfg.num_actions, 3), device=self.device, dtype=torch.float)
        motorValues = torch.clone(actions) # at this point these are PWM commands between -1 and 1

        if self._debug: print("motorValues: ", motorValues)

        # convert the PWM commands to rad/s using method in https://gitlab.com/warplab/ros/warpauv/warpauv_simulation/-/blob/master/src/robot_sim_interface.py
        motorValues[torch.abs(motorValues) < 0.08] = 0 
        motorValues[motorValues >= 0.08] = -139.0 * (torch.pow(motorValues[motorValues >= 0.08], 2.0)) + 500 * motorValues[motorValues >= 0.08] + 8.28
        motorValues[motorValues <= -0.08] = 161.0 * (torch.pow(motorValues[motorValues <= -0.08], 2.0)) + 517.86 * motorValues[motorValues <= -0.08] - 5.72

        # get the current motor velocities using thruster dynamics
        # TODO: CHECK THAT SIM DT IS CORRECT HERE
        motorValues = self.thruster_dynamics.update(motorValues, self.episode_length_buf * self.sim.cfg.dt)

        # get thruster forces from their speeds using the thruster conversion function 
        motorValues = self.thruster_conversion.convert(motorValues)

        # TODO: this could be taken out of the physics step
        thruster_forces[..., self.cfg.thruster_axis] = 1.0 # start with forces in the z direction (TODO: CUREE uses x-direction)

        thruster_forces = quat_apply(self.thruster_quats, thruster_forces) # rotate the forces into the thruster's frame

        # apply the force magnitudes to the thruster forces
        thruster_forces = thruster_forces * motorValues.unsqueeze(-1) # make motorValues shape (num_envs, 6, 1))

        # calculate the thruster torques 
        # T = r x F
        # T (num_envs, num_thrusters_per_env, 3)
        # r (num_thrusters_per_env, 3)
        # F (num_envs, num_thrusters_per_env, 3)
        # it should broadcast r to be (num_envs, num_thrusters_per_env, 3)
        thruster_torques = torch.cross(self.thruster_com_offsets, thruster_forces, dim=-1)

        # now sum together all the forces/torques on each robot
        thruster_forces = torch.sum(thruster_forces, dim=-2) # sum over the thruster indices
        thruster_torques = torch.sum(thruster_torques, dim=-2) # sum over the thruster indices

        ## Calculate hydrodynamics
        if self._debug: print("gravity magnitude: ", self._gravity_magnitude) 
        buoyancy_forces, buoyancy_torques = self.force_calculation_functions.calculate_buoyancy_forces(self._robot.data.root_quat_w, self.cfg.water_rho, self.volumes, abs(self._gravity_magnitude), self.cfg.com_to_cob_offset)

        density_forces, density_torques, viscosity_forces, viscosity_torques = self.force_calculation_functions.calculate_density_and_viscosity_forces(
          self._robot.data.root_quat_w, self._robot.data.root_lin_vel_w, self._robot.data.root_ang_vel_w, self.inertia_tensors, self.inertia_tensors_mean, self.cfg.water_beta, self.cfg.water_rho, self._robot.root_physx_view._masses.to(self.device)
        )

        wave_forces = self.force_calculation_functions.calculate_wave_forces(self._robot.data.root_quat_w, self._wave_frequencies, self._wave_amplitudes, self._env_step_count)

        current_forces = self.force_calculation_functions.calculate_current_forces(self._robot.data.root_quat_w, self._current_directions, self._current_forces)

        if self._debug: print("density forces: ", density_forces)
        if self._debug: print("density torques: ", density_torques)

        if self._debug: print("viscosity forces: ", viscosity_forces)
        if self._debug: print("viscosity torques: ", viscosity_torques)

        if self._debug: print("buoyancy forces: ", buoyancy_forces)
        if self._debug: print("buoyancy torques: ", buoyancy_torques)

        if self._debug: print("thruster forces: ", thruster_forces)
        if self._debug: print("thruster torques: ", thruster_torques)

        forces = density_forces + buoyancy_forces + viscosity_forces + wave_forces + current_forces + thruster_forces
        torques = density_torques + buoyancy_torques + viscosity_torques + thruster_torques

        if self._debug: print("final forces", forces)
        if self._debug: print("final torques", torques)

        return forces, torques

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "goal_ang_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_ang"
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                self.goal_ang_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "goal_z_ang_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_z_ang"
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                self.goal_z_ang_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "x_b_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Command/x_b"
                self.x_b_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "z_b_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Command/z_b"
                self.z_b_visualizer = VisualizationMarkers(marker_cfg)
            
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            self.goal_ang_visualizer.set_visibility(True)
            self.goal_z_ang_visualizer.set_visibility(True)
            self.x_b_visualizer.set_visibility(True)
            self.z_b_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

            if hasattr(self, "goal_ang_visualizer"):
                self.goal_ang_visualizer.set_visibility(False)

            if hasattr(self, "goal_z_ang_visualizer"):
                self.goal_z_ang_visualizer.set_visibility(False)

            if hasattr(self, "x_b_visualizer"):
                self.x_b_visualizer.set_visibility(False)
            
            if hasattr(self, "z_b_visualizer"):
                self.z_b_visualizer.set_visibility(False)

    def _rotate_quat_by_euler_xyz(self, q: torch.tensor, x: float|torch.tensor, y: float|torch.tensor, z: float|torch.tensor, device=None):
        # Assumes q has shape [num_envs, 4]
        num_envs = q.shape[0]
        if device == None:
            device = self.device

        if type(x) == float:
            x = torch.zeros(num_envs, device=device) + x

        if type(y) == float:
            y = torch.zeros(num_envs, device=device) + y
        
        if type(z) == float:
            z = torch.zeros(num_envs, device=device) + z

        iq = math_utils.quat_from_euler_xyz(x, y, z)
        return math_utils.quat_mul(q, iq)


    def _debug_vis_callback(self, event):
        # Visualize the goal positions
        self.goal_pos_visualizer.visualize(translations = self._goal + self.scene.env_origins)
        # self.goal_pos_visualizer.visualize(translations = self._goal_pos_w)

        # Visualize goal orientations
        # goal_quats_w = self._goal
        # ang_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        # ang_marker_scales[:, 0] = 1
        # self.goal_ang_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=goal_quats_w, scales=ang_marker_scales)

        # Visualize goal orientations via another axis
        # goal_z_quat = self._rotate_quat_by_euler_xyz(goal_quats_w, 0.0, -torch.pi/2, 0.0)
        # ang_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        # ang_marker_scales[:, 0] = 1
        # self.goal_z_ang_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=goal_z_quat, scales=ang_marker_scales)

        # Visualize current X-direction
        # x_w = self._robot.data.root_quat_w
        # x_w_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        # x_w_marker_scales[:, 0] = 1
        # self.x_b_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=x_w, scales=x_w_marker_scales)
        pass
        # Visualize current Z-direction
        # z_w_quat = self._rotate_quat_by_euler_xyz(self._robot.data.root_quat_w, 0.0, -torch.pi/2, 0.0)
        # z_w_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        # z_w_marker_scales[:, 0] = 1
        # self.z_b_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=z_w_quat, scales=z_w_marker_scales)