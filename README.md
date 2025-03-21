# learning-based-control

This is an IsaacLab-based project for Reinforcement Learning and Control of AUVs.

IMPORTANT NOTICE:
- Isaac-WarpAUV-Direct-v1 and warpauv_env.py were used to produce results for the ICRA/arxiv paper

To install:
- Install IsaacSim (https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html)
- Install IsaacLab (https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html)

- Clone this repository:

  - If using docker container:
  ```
  cd <IsaacLab_Path>/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/learning-based-control
  git clone git@gitlab.com:warplab/learning-based-control.git
  ```

  - If using workstation install:
  ```
  git clone git@gitlab.com:warplab/learning-based-control.git
  ln -s learning-based-control <IsaacLab_Path>/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/learning-based-control
  ```
  (Note: if using a workstation install, you can follow the docker instructions as well, but the soft link seems cleaner for local development. Docker is painful when working with links)

To run:
```
./isaaclab.sh -p -m wandb login 436f9588898e2f19db70b166dc5f7bac8d7e97f8
WANDB_USERNAME=whoi-warplab ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-WarpAUV-Direct-v1 --num_envs 1
```

To additional notes:

 - To import a URDF file into USD format for IsaacLab, you can first export a ROS xacro file into URDF, and then import that URDF file into the IsaacSim URDFImporter Workflow.

 ```
 rosrun xacro xacro --inorder -o <output.urdf> <input.xacro>
 ./isaaclab.sh -p source/standalone/tools/convert_urdf.py  <input_urdf> <output_usd> --merge-joints --make-instance
 ```