from dataclasses import dataclass
from typing import Tuple
from omni.isaac.lab.utils.math import quat_conjugate, quat_inv, quat_apply, convert_quat
import numpy as np 
import torch 
import math

@dataclass
class HydrodynamicForceModels:
  num_envs: int 
  device: torch.device
  debug: bool = False

  def calculate_current_forces(self,
                            root_quats_w: torch.tensor,
                            current_directions_w: torch.tensor,
                            forces: torch.tensor):
    current_forces_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
    current_directions_b = quat_apply(quat_conjugate(root_quats_w), current_directions_w)
    current_forces_b = current_directions_b * forces.unsqueeze(1)

    return current_forces_b

  def calculate_wave_forces(self,
                            root_quats_w: torch.tensor,
                            frequency: torch.tensor,
                            amplitude: torch.tensor,
                            t: torch.tensor):
    wave_forces_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
    
    wave_directions_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
    wave_directions_w[..., 2] = 1.0

    wave_directions_b = quat_apply(quat_conjugate(root_quats_w), wave_directions_w)
    wave_forces_b = wave_directions_b * (amplitude * torch.sin(((2 * math.pi) / frequency) * t)).unsqueeze(1)

    return wave_forces_b

  def calculate_buoyancy_forces(self,
                                root_quats_w: torch.tensor, # robot orientations in world frame
                                fluid_density: float, # fluid density
                                volumes: torch.tensor, # rigid body volume 
                                g_mag: float, # magnitude of gravity
                                com_to_cob_offsets:torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """
    Compute wrenches (forces and torques) due to buoyancy on fully-submerged rigid body in fluid.
    Returned forces and torques are in the body root frame.
    Note that gravity is applied by Isaac Sim by default.
    """

    buoyancy_forces_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
    buoyancy_torques_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

    buoyancy_directions_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
    buoyancy_directions_w[..., 2] = 1.0 # opposing gravity vector in the world frame
    
    if self.debug: print(f"shape of root_quats: {root_quats.shape}, shape of buoyancy_vectors: {buoyancy_vectors_w.shape}")

    buoyancy_directions_b = quat_apply(quat_conjugate(root_quats_w), buoyancy_directions_w)

    # todo: we should actually be computing buoyancy forces at the root of the vehicle, not the COB, is this the same though?
    buoyancy_forces_at_cob_b = buoyancy_directions_b * fluid_density * volumes.repeat(1,3) * g_mag
    buoyancy_forces_b = buoyancy_forces_at_cob_b

    buoyancy_torques_b = torch.cross(com_to_cob_offsets, buoyancy_forces_at_cob_b, dim=-1) # torque = r x F

    if self.debug: print(f"Calculated buoyancy values: forces are {buoyancy_forces_b} and torques are {buoyancy_torques_b}")

    return (buoyancy_forces_b, buoyancy_torques_b)
  
  def _calculate_inferred_half_dimensions(self, inertias, masses):
    """
    Computes inferred half dimensions for an "equivalent inertia box" of the vehicle
    """
    r = torch.sqrt( (3/(2 * masses.repeat(1,3))) * (torch.roll(inertias, 1, 1) + torch.roll(inertias, -1, 1) - inertias))
    return r

  def calculate_quadratic_drag_forces(self,
                                  root_linvels_b: torch.tensor,
                                  root_angvels_b: torch.tensor,
                                  inertias: torch.tensor,
                                  masses: torch.tensor,
                                  fluid_density_rho
                                  ):

    ri = self._calculate_inferred_half_dimensions(inertias, masses)
    rj = torch.roll(ri, 1, 1)
    rk = torch.roll(ri, -1, 1)

    forces = -2. * fluid_density_rho * rj * rk * torch.abs(root_linvels_b) * root_linvels_b
    torques = -0.5 * fluid_density_rho * ri * (torch.pow(rj,4) + torch.pow(rk,4)) * torch.abs(root_angvels_b) * root_angvels_b

    return (forces, torques)

  def calculate_linear_viscous_forces(self, 
                                      root_linvels_b: torch.tensor,
                                      root_angvels_b: torch.tensor,
                                      inertias: torch.tensor,
                                      masses,
                                      fluid_viscosity_beta
                                      ):
    ri = self._calculate_inferred_half_dimensions(inertias, masses)
    r_eq = torch.mean(ri, 1, keepdim=True)

    r_eq = r_eq.repeat(1,3)
    forces = -6. * fluid_viscosity_beta * torch.pi * r_eq * root_linvels_b
    torques = -8. * fluid_viscosity_beta * torch.pi * torch.pow(r_eq, 3) * root_angvels_b
    return (forces, torques)

  def calculate_density_and_viscosity_forces(self, 
                                             root_quats_w: torch.tensor,
                                             root_linvels_w:torch.tensor, #[num_envs, 3]
                                             root_angvels_w:torch.tensor, #[num_envs, 3]
                                             inertias: torch.Tensor, #[num_envs, 3]
                                             inertias_mean: torch.Tensor, #[num_envs, 1]
                                             water_beta: float, 
                                             water_rho: float,
                                             masses: torch.tensor
                                             ):

    root_quats_b = quat_conjugate(root_quats_w)
    root_linvels_b = quat_apply(root_quats_b, root_linvels_w)
    root_angvels_b = quat_apply(root_quats_b, root_angvels_w)
  
    f_d, g_d = self.calculate_quadratic_drag_forces(root_linvels_b, root_angvels_b, inertias, masses, water_rho)
    f_v, g_v = self.calculate_linear_viscous_forces(root_linvels_b, root_angvels_b, inertias, masses, water_beta)
    return (f_d, g_d, f_v, g_v)

  def calculate_density_and_viscosity_forces_deprecated(self, 
                                             root_quats: torch.tensor,
                                             root_linvels:torch.tensor, #[num_envs, 3]
                                             root_angvels:torch.tensor, #[num_envs, 3]
                                             env_inertia_tensors: torch.Tensor, #[num_envs, 3]
                                             inertia_tensors_mean: torch.Tensor, #[num_envs, 1]
                                             water_beta: float, 
                                             water_rho: float,
                                             ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """
    Implementation based on the Mujoco Inertia model for Fluid forces, which infers geometry from body-equivalent-inertia boxes
    https://mujoco.readthedocs.io/en/latest/computation/fluid.html

    NOTE: this is deprecated, incorrectly uses inertia rather than half-body lengths and is based on an older mujoco model
    """

    # invert quat from wxyz to xyzw
    # root_quats = convert_quat(root_quats, to="xyzw")

    #first, transform the linear and angular velocities to the robot frame 
    inv_quats = quat_conjugate(root_quats)
    local_linvels = quat_apply(inv_quats, root_linvels)
    local_angvels = quat_apply(inv_quats, root_angvels)

    if self.debug: print(f"root_linvels: {root_linvels}\nlocal_linvels: {local_linvels}")

    density_forces = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
    density_torques = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

    viscosity_forces = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
    viscosity_torques = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

    if self.debug: print(f"shape of density forces: {density_forces.shape} | shape of local_linvels: {local_linvels.shape}")
    # apply density and viscosity forces!


    density_forces = -0.5 * water_rho * torch.roll(env_inertia_tensors, 1, 1) * torch.roll(env_inertia_tensors, -1, 1) * torch.abs(local_linvels) * local_linvels

    density_torques = -1 / 64.0 * water_rho * (
      torch.pow(torch.roll(env_inertia_tensors, 1, 1), 4) + torch.pow(torch.roll(env_inertia_tensors, -1, 1), 4)
    ) * torch.abs(local_angvels) * local_angvels

    
    viscosity_forces = - 3.0 * water_beta * np.pi * inertia_tensors_mean * local_linvels
    viscosity_torques = -1.0 * water_beta * np.pi * torch.pow(inertia_tensors_mean, 3) * local_angvels


    if self.debug: print(f"Calculated viscosity values: forces are {viscosity_forces} and torques are {viscosity_torques}")
    if self.debug: print(f"Calculated density values: forces are {density_forces} and torques are {density_torques}")

    return (density_forces, density_torques, viscosity_forces, viscosity_torques)

if __name__ == "__main__":
  # do some unit tests! 
  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  water_rho = 997.0 # kg/m^3
  water_beta = 0.001306 # Pa s, dynamic viscosity of water @ 50 deg F
  g_mag = 9.81
  num_envs = 4
  com_to_cob_offset = torch.tensor([0.0, 0.0, 0.3], dtype=torch.float, device=device, requires_grad=False).reshape(1,3).repeat(num_envs, 1) 
  volume = 0.022747843530591776 # assuming cubic meters - NEUTRALLY BOUYANT

  forceModel = HydrodynamicForceModels(num_envs, device, True)

  root_quats = torch.tensor([[0.0, 0.0, 0.0, 1.0], # no rotation
                             [-0.7071068, 0, 0, 0.7071068], # 90 deg rotation about x
                             [ 0, -0.7071068, 0, 0.7071068 ], 
                             [ 0.3535534, 0.3535534, 0.1464466, 0.8535534 ], 
  
  ]).to(device)

  true_b_forces = torch.tensor([[0.0, 0.0, volume * water_rho * g_mag],
                                [0.0, -1 * volume * water_rho * g_mag, 0.0], 
                                [ volume * water_rho * g_mag, 0.0, 0.0], 
                                [ -0.5 * volume * water_rho * g_mag, 0.7071 * volume * water_rho * g_mag, 0.5 * volume * water_rho * g_mag], 
  
  ]).to(device)

  true_b_torques = torch.tensor([[0.0, 0.0, 0.0],
                                  [0.3 * water_rho * g_mag * volume, 0.0, 0.0],
                                  [0.0, 0.3 * water_rho * g_mag * volume, 0.0],
                                  [-0.3 * 0.7071 * water_rho * g_mag * volume, -0.15 * water_rho * g_mag * volume, 0.0],
  ]).to(device)

  b_force, b_torque = forceModel.calculate_buoyancy_forces( root_quats, water_rho, volume, g_mag, com_to_cob_offset)

  if(np.abs(b_force.cpu().numpy() - true_b_forces.cpu().numpy()).max() > 1e-9):
    print(f"ERROR: b_force is\n {b_force} \nand true_b_forces is \n {true_b_forces}\n with max value {np.abs(b_force.cpu().numpy() - true_b_forces.cpu().numpy()).max()}")
  if (np.abs(b_torque.cpu().numpy() - true_b_torques.cpu().numpy()).max() > 1e-9):
    print(f"ERROR: b_torque is\n {b_torque} and true_b_torques is\n {true_b_torques}")


  root_linvels = torch.tensor([[[0.0, 0.0, 0.0]],
                               [[0.0, 0.0, 0.0]],
  ])
  root_angvels = torch.tensor([[[0.0, 0.0, 0.0]],
                               [[0.0, 0.0, 0.0]],
  ])

  #env_inertia_tensors = 
  #dens_force, dense_torqe, visc_force, visc_torque = forceModel.calculate_density_and_viscosity_forces(root_linvels, root_angvels, env_inertia_tensors, water_beta, water_rho)