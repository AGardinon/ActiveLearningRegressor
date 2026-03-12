#!

# --------------------------------------------------------------------------------
# FUNCTIONAL FORMS (USED IN THE BENCHMARKS STUDY)

import numpy as np
from botorch.test_functions import Hartmann, Ackley, StyblinskiTang

FUNCTION_CLASSES = {
    "Hartmann": Hartmann,
    "Ackley": Ackley,
    "StyblinskiTang": StyblinskiTang
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

# --- ALL FUNCTIONS

FUNCTIONS_DICT = {
    "Hartmann3D": hartmann_dict["Hartmann3D"],
    "Hartmann6D": hartmann_dict["Hartmann6D"],
    "Ackley3D": ackley_dict["Ackley3D"],
    "Ackley6D": ackley_dict["Ackley6D"],
    "StyblinskiTang3D": styblinski_tang_dict["StyblinskiTang3D"],
    "StyblinskiTang6D": styblinski_tang_dict["StyblinskiTang6D"]
}