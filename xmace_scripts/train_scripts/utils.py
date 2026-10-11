"""Shared helpers for training scripts."""

import math
from copy import deepcopy

import numpy as np


def merge_config(defaults, overrides):
    """
    In general, each training run will contain information from defaults.json and its individual
    run's input.json. This function merges the inputs from both jsons into one dictionary,
    with the input.json values overriding the default ones. 
    """
    config = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = merge_config(config[key], value)
        else:
            config[key] = deepcopy(value)
    return config


def split_atoms(atoms, valid_fraction, seed):
    """
    Split xyz files into training and validation sets based on a specified validation fraction
    First light validations on the input
    Then uses rng to randomly split the indices of the xyz file
    Return the indices also so they can be recorded and saved
    """
    if isinstance(valid_fraction, bool) or not isinstance(valid_fraction, (int, float)):
        raise TypeError("valid_fraction must be a number between 0 and 1.")
    if not math.isfinite(valid_fraction) or not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1.")

    num_frames = len(atoms)
    if num_frames < 2:
        raise ValueError("Automatic splitting requires at least two geometries.")

    num_valid = math.ceil(num_frames * valid_fraction)
    if num_valid >= num_frames:
        raise ValueError("valid_fraction leaves no geometries for training.")

    indices = np.random.default_rng(seed).permutation(num_frames)
    valid_indices = indices[:num_valid].tolist()
    train_indices = indices[num_valid:].tolist()

    train_atoms = [atoms[i] for i in train_indices]
    valid_atoms = [atoms[i] for i in valid_indices]

    return train_atoms, valid_atoms, train_indices, valid_indices

