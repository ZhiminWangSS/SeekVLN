"""Compatibility for Habitat 0.1.7 with the validated NumPy/Gym runtime."""

import os
import importlib.machinery
import sys

import numpy as np

if os.environ.get("SEEKVLN_TENSORBOARD_NO_TF", "0") == "1":
    notf = type(sys)("tensorboard.compat.notf")
    notf.__spec__ = importlib.machinery.ModuleSpec("tensorboard.compat.notf", loader=None)
    sys.modules.setdefault("tensorboard.compat.notf", notf)

for alias, scalar_type in {
    "bool": bool, "float": float, "int": int, "object": object, "str": str,
}.items():
    if alias not in np.__dict__:
        setattr(np, alias, scalar_type)

if os.environ.get("SEEKVLN_HABITAT_DISCRETE_ZERO_COMPAT", "0") == "1":
    from gym.spaces import Discrete

    if not getattr(Discrete, "_seekvln_zero_placeholder_patch", False):
        original_init = Discrete.__init__

        def _init(self, n, *args, **kwargs):
            return original_init(self, 1 if int(n) == 0 else n, *args, **kwargs)

        Discrete.__init__ = _init
        Discrete._seekvln_zero_placeholder_patch = True
