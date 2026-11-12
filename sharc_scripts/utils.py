"""Shared configuration merging and validation for SHARC scripts."""

from copy import deepcopy
from difflib import SequenceMatcher


def merge_config(defaults, overrides, run_only_keys=()):
    """
    In general, each training run will contain information from defaults.json and its individual
    run's input.json. This function merges the inputs from both jsons into one dictionary,
    with the input.json values overriding the default ones.
    Every key in the run JSON must exist in defaults.json, except the top-level run_only_keys.
    """
    unknown = _find_unknown_keys(defaults, overrides, set(run_only_keys))
    if unknown:
        valid_paths = _key_paths(defaults) + sorted(run_only_keys)
        lines = [f"  '{path}'{_suggest_key(path, valid_paths)}" for path in unknown]
        raise ValueError(
            "Unknown key(s) in run config (not present in defaults.json):\n" + "\n".join(lines)
        )
    return _merge_dicts(defaults, overrides)


def _merge_dicts(defaults, overrides):
    config = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = _merge_dicts(config[key], value)
        else:
            config[key] = deepcopy(value)
    return config


def _find_unknown_keys(defaults, overrides, allowed=frozenset(), prefix=""):
    # Only descend where the default is a dict; null/list defaults (e.g. E0s) take free-form values.
    unknown = []
    for key, value in overrides.items():
        path = prefix + key
        if key in allowed:
            continue
        if key not in defaults:
            unknown.append(path)
        elif isinstance(value, dict) and isinstance(defaults[key], dict):
            unknown += _find_unknown_keys(defaults[key], value, prefix=path + ".")
    return unknown


def _key_paths(defaults, prefix=""):
    paths = []
    for key, value in defaults.items():
        paths.append(prefix + key)
        if isinstance(value, dict):
            paths += _key_paths(value, prefix + key + ".")
    return paths


def _suggest_key(path, valid_paths, cutoff=0.6):
    name = path.rsplit(".", 1)[-1]
    score, best = max(
        (SequenceMatcher(None, name, valid.rsplit(".", 1)[-1]).ratio(), valid)
        for valid in valid_paths
    )
    return f" -> did you mean '{best}'?" if score >= cutoff else ""
