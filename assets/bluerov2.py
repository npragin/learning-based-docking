import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg

import os
USD_PATH = os.path.join(os.path.dirname(__file__), "../data/bluerov2/BlueRov2.usd")

BLUEROV2_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        copy_from_source=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5),
    )
)
"""Configuration for the BlueROV2."""

BLUEROV2_THRUSTER_CFG = {
    "thruster1": {
        "id": 0,
        "position": (-0.4127, .1506, -0.0889),
        "rot": (1, 0, 0, 0)
    },
    "thruster2": {
        "id": 1,
        "position": (-0.4127,-.1506,-0.0889),
        "rot": (1, 0, 0, 0)
    },
    "thruster3": {
        "id": 2,
        "position": (-0.303, 0.1461, -0.1587),
        "rot": (-0.303, 0.1461, -0.1587)
    },
    "thruster4": {
        "id": 3,
        "position": (-0.303, -0.1461, -0.1587),
        "rot": (0, -0.785398, -1.5708)
    },
    "thruster5": {
        "id": 4,
        "position": (0.0585, -0.1461, -0.0540),
        "rot": (0, 0.785398,-1.5708)
    },
    "thruster6": {
        "id": 5,
        "position": (0.0585, 0.1461, -0.0540),
        "rot": (0, 0.785398, 1.5708)
    }
}
"""Thruster Configuration"""