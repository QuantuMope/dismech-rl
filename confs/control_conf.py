import alf
import numpy as np
from typing import Literal

from utils.action_converter import ActionConverter


def define_control_type(control_type: Literal["bend", "bend_2d", "bend_twist",
                                              "contact_bend",
                                              "contact_bend_2d"]):
    alf.config("ActionConverter",
               kappa_bar_range=(-2.0, 2.0),
               smooth_action=True)

    if "2d" in control_type:
        alf.config("ActionConverter", ws_dim=2)

    if control_type in ["bend", "bend_2d"]:
        alf.config("ActionConverter", delta_kappa_scale=0.05)

    elif control_type == "bend_twist":
        alf.config(
            "ActionConverter",
            delta_kappa_scale=0.05,
            include_twist=True,
            delta_theta_scale=0.05,
            twist_bar_range=(-np.pi / 2, np.pi / 2),
        )

    elif control_type in ["contact_bend", "contact_bend_2d"]:
        alf.config("ActionConverter", delta_kappa_scale=0.25)
