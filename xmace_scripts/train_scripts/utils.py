"""Shared helpers for training scripts."""

from copy import deepcopy


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
