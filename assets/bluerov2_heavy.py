import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg

import os
USD_PATH = os.path.join(os.path.dirname(__file__), "../data/bluerov2_heavy/BlueRov2_Heavy.usd")

BLUEROV2_HEAVY_CFG = RigidObjectCfg(
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

BLUEROV2_HEAVY_THRUSTER_CFG = {
    "thruster1": {
        "id": 0,
        "position": (0.14, -0.092, 0.0),
        "rot": (-1.571, 1.571, -0.785)
    },
    "thruster2": {
        "id": 1,
        "position": (0.14, 0.092, 0.0),
        "rot": (-1.571, 1.571, -2.356)
    },
    "thruster3": {
        "id": 2,
        "position": (-0.15, -0.092, 0.0),
        "rot": (-1.571, 1.571, 0.785)
    },
    "thruster4": {
        "id": 3,
        "position": (-0.15, 0.092, 0.0),
        "rot": (-1.571, 1.571, 2.356)
    },
    "thruster5": {
        "id": 4,
        "position": (0.118, -0.215, 0.064),
        "rot": (0, 0, 0)
    },
    "thruster6": {
        "id": 5,
        "position": (0.118, -0.215, 0.064),
        "rot": (0, 0, 0)
    },
    "thruster7": {
        "id": 6,
        "position": (-0.118, -0.215, 0.064),
        "rot": (0, 0, 0)
    },
    "thruster8": {
        "id": 7,
        "position": (-0.118, 0.215, 0.064),
        "rot": (0, 0, 0)
    }
}
"""Thruster Configuration"""