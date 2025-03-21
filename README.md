# Learning-Based AUV Docking

This repository contains the code for our project on developing robust autonomous underwater vehicle (AUV) docking policies using deep reinforcement learning in IsaacLab. Find the full report [here](readme_assets/report.pdf).

## Overview

Autonomous docking is critical for long-term AUV missions, enabling data transmission, battery charging, and recovery. However, autonomous docking for underwater vehicles presents unique challenges due to:
- Complex hydrodynamics and environmental disturbances
- Limited visibility and sensing capabilities underwater
- The high cost and risk of real-world testing

Our approach leverages simulation-based reinforcement learning to develop policies that can generalize to out-of-distribution (OOD) vehicle dynamics - a critical capability for real-world deployment.

## Docking Environment

Our simulation environment extends the AUV simulator built on NVIDIA's Isaac Labs framework. The physics model includes:

- PhysX for non-hydrodynamic physics and collisions
- MuJoCo-based hydrodynamic model for drag and buoyant forces
- Zero-order thruster dynamics based on Yoerger et al.

## Policy Architectures

We implemented and evaluated four distinct policy architectures:

1. **Naive Position-Orientation-Reward Policy**: Baseline trained on fixed dynamics
2. **Small Domain-Randomized Policy**: Trained with smaller randomization range
3. **Large Domain-Randomized Policy**: Trained with larger randomization range
4. **Domain-Randomized History-Based Policy**: Incorporates both domain randomization and a history-based observation space

## Methodology

### Reward Function

Our reward function encourages efficient docking behavior:

$$R(s_t, a_t) = λ_1 * R_{dist} + λ_2 * R_{orient}$$

Where:
$$R_{dist} = exp(-||p_t - p_dock||)$$
  - Rewards proximity to docking station
$$R_{orient} = exp(-||θ_t - θ_dock||)$$
  - Rewards correct orientation

### Evaluation Protocol

We evaluated all policies under four conditions:
1. **In-Distribution**: Mass = 11.5 kg
2. **Edge-of-Distribution**: Mass = 13 kg
3. **Out-of-Distribution**: Mass = 16.5 kg
4. **Extreme Out-of-Distribution**: Mass = 21 kg

### Performance Metrics
- **Success Rate**: Percentage of successful docking attempts
- **Time to Dock**: Average time to complete successful docking
- **Distance Traveled**: Total path length during docking
- **Energy Efficiency**: Cumulative thruster usage
- **Motion Smoothness**: Measured by cumulative jerk

## Key Results

- **Robustness**: The Large DR with History policy showed the strongest resilience at higher mass adjustments (OOD conditions)
- **Performance Trade-offs**: Domain-randomized policies generally underperformed compared to the Naive Policy in in-distribution conditions but showed better generalization to OOD scenarios
- **Control Efficiency**: Robustness to OOD conditions came at the cost of higher jerk and action costs

## Future Directions

- Optimizing the domain randomization approach to balance performance in both ID and OOD conditions
- Extending to real-world testing and validation
- Exploring additional parameters beyond mass for domain randomization

## Installation

- Install IsaacSim (https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html)
- Install IsaacLab (https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html)

- Clone this repository:

  - If using docker container:
  ```
  cd <IsaacLab_Path>/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/learning-based-docking
  git clone git@gitlab.com:npragin/learning-based-docking.git
  ```

  - If using workstation install:
  ```
  git clone git@gitlab.com:npragin/learning-based-docking.git
  ln -s learning-based-control <IsaacLab_Path>/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/learning-based-docking
  ```
  (Note: if using a workstation install, you can follow the docker instructions as well, but the soft link seems cleaner for local development. Docker is painful when working with links)

## Usage

```
./isaaclab.sh -p -m wandb login <wandb_api_key>
WANDB_USERNAME=<wandb_username> ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-BlueROV2-Direct-v1 --num_envs 1
```

### Additional Notes:

 - To import a URDF file into USD format for IsaacLab, you can first export a ROS xacro file into URDF, and then import that URDF file into the IsaacSim URDFImporter Workflow.

 ```
 rosrun xacro xacro --inorder -o <output.urdf> <input.xacro>
 ./isaaclab.sh -p source/standalone/tools/convert_urdf.py  <input_urdf> <output_usd> --merge-joints --make-instance
 ```