#!

# --------------------------------------------------------------------------------
# FUNCTIONAL FORMS (USED IN THE BENCHMARKS STUDY)

import numpy as np
from functools import partial as _partial
from botorch.test_functions import Hartmann, Ackley, StyblinskiTang
from botorch.test_functions.multi_objective import (
    BraninCurrin as _BraninCurrin,
    DTLZ2 as _DTLZ2,
    ZDT3 as _ZDT3,
    DTLZ7 as _DTLZ7,
)


def _branin_currin_factory(dim=2, noise_std=0.0, negate=False, **_):
    # BraninCurrin.dim == 2 always; dim kwarg is accepted to match DatasetGenerator interface
    return _BraninCurrin(noise_std=noise_std if noise_std else None, negate=negate)


FUNCTION_CLASSES = {
    "Hartmann": Hartmann,
    "Ackley": Ackley,
    "StyblinskiTang": StyblinskiTang,
    "BraninCurrin": _branin_currin_factory,
    "DTLZ2": _partial(_DTLZ2, num_objectives=2),
    "ZDT3": _partial(_ZDT3, num_objectives=2),
    "DTLZ7": _partial(_DTLZ7, num_objectives=2),
}

# --- Hartmann

hartmann_dict = {

    "Hartmann3D": {
        "function_class": Hartmann,
        "function_params": {
            "n_dimensions": 3,
            "bounds": np.array([[0.0, 1.0]] * 3)
        },
    },

    "Hartmann6D": {
        "function_class": Hartmann,
        "function_params": {
            "n_dimensions": 6,
            "bounds": np.array([[0.0, 1.0]] * 6)
        },
    },
}

# --- Ackley

ackley_dict = {

    "Ackley3D": {
        "function_class": Ackley,
        "function_params": {
            "n_dimensions": 3,
            "bounds": np.array([[-32.768, 32.768]] * 3)
        },
    },

    "Ackley6D": {
        "function_class": Ackley,
        "function_params": {
            "n_dimensions": 6,
            "bounds": np.array([[-32.768, 32.768]] * 6)
        },
    },
}

# --- StyblinskiTang

styblinski_tang_dict = {

    "StyblinskiTang3D": {
        "function_class": StyblinskiTang,
        "function_params": {
            "n_dimensions": 3,
            "bounds": np.array([[-4.00, 4.00]] * 3)
        },
    },

    "StyblinskiTang6D": {
        "function_class": StyblinskiTang,
        "function_params": {
            "n_dimensions": 6,
            "bounds": np.array([[-4.00, 4.00]] * 6)
        },
    },
}

# --- BraninCurrin (2D input, 2 objectives — baseline multi-property benchmark)

branin_currin_dict = {

    "BraninCurrin": {
        "function_class": _branin_currin_factory,
        "function_params": {
            "n_dimensions": 2,
            "bounds": np.array([[0.0, 1.0]] * 2)
        },
    },
}

# --- DTLZ2 (spherical/convex Pareto front, scalable)

dtlz2_dict = {

    "DTLZ2_2obj_4D": {
        "function_class": _partial(_DTLZ2, num_objectives=2),
        "function_params": {
            "n_dimensions": 4,
            "bounds": np.array([[0.0, 1.0]] * 4)
        },
    },

    "DTLZ2_2obj_6D": {
        "function_class": _partial(_DTLZ2, num_objectives=2),
        "function_params": {
            "n_dimensions": 6,
            "bounds": np.array([[0.0, 1.0]] * 6)
        },
    },
}

# --- ZDT3 (2 objectives, discontinuous Pareto front — 5 segments)

zdt3_dict = {

    "ZDT3_2obj_4D": {
        "function_class": _partial(_ZDT3, num_objectives=2),
        "function_params": {
            "n_dimensions": 4,
            "bounds": np.array([[0.0, 1.0]] * 4)
        },
    },

    "ZDT3_2obj_6D": {
        "function_class": _partial(_ZDT3, num_objectives=2),
        "function_params": {
            "n_dimensions": 6,
            "bounds": np.array([[0.0, 1.0]] * 6)
        },
    },
}

# --- DTLZ7 (disconnected Pareto front, scalable)

dtlz7_dict = {

    "DTLZ7_2obj_4D": {
        "function_class": _partial(_DTLZ7, num_objectives=2),
        "function_params": {
            "n_dimensions": 4,
            "bounds": np.array([[0.0, 1.0]] * 4)
        },
    },
}

# --- ALL FUNCTIONS

FUNCTIONS_DICT = {
    "Hartmann3D": hartmann_dict["Hartmann3D"],
    "Hartmann6D": hartmann_dict["Hartmann6D"],
    "Ackley3D": ackley_dict["Ackley3D"],
    "Ackley6D": ackley_dict["Ackley6D"],
    "StyblinskiTang3D": styblinski_tang_dict["StyblinskiTang3D"],
    "StyblinskiTang6D": styblinski_tang_dict["StyblinskiTang6D"],
    # Multi-objective
    "BraninCurrin": branin_currin_dict["BraninCurrin"],
    "DTLZ2_2obj_4D": dtlz2_dict["DTLZ2_2obj_4D"],
    "DTLZ2_2obj_6D": dtlz2_dict["DTLZ2_2obj_6D"],
    "ZDT3_2obj_4D": zdt3_dict["ZDT3_2obj_4D"],
    "ZDT3_2obj_6D": zdt3_dict["ZDT3_2obj_6D"],
    "DTLZ7_2obj_4D": dtlz7_dict["DTLZ7_2obj_4D"],
}