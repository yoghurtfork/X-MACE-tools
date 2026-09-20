"""
Match duration is defined by how long a trajectory's cumulative errors remain within the max thresholds
If any of energy, force, or NACs cumulative errors exceed the thresholds, it fails
"""

from __future__ import annotations

import numpy as np


def calculate_match_duration(
    cumulative_error: dict[str, np.ndarray | None],
    max_energy_error: float,
    max_force_error: float,
    max_nac_error: float | None = None,
) -> dict[str, float | str | None]:
    """Return the duration and first metric whose cumulative error exceeds its limit."""
    time = np.asarray(cumulative_error["time"])
    metrics = [("energy", "energy_cumulative_error", max_energy_error), ("forces", "forces_cumulative_error", max_force_error)]
    if max_nac_error is not None: # NACs are optional
        if cumulative_error["nacs_cumulative_error"] is None:
            raise ValueError("NAC matching requested but NAC cumulative error is unavailable")
        metrics.append(("nacs", "nacs_cumulative_error", max_nac_error))
    curves = []
    for name, key, maximum in metrics:
        curve = np.asarray(cumulative_error[key])
        curves.append((name, curve, maximum))
    for index in range(len(time)):
        for name, curve, maximum in curves:
            if curve[index] > maximum:
                return {"duration": float(time[index - 1] - time[0]) if index else 0.0, "failed_metric": name}
    return {"duration": float(time[-1] - time[0]), "failed_metric": None}


def calculate_ensemble_match_duration(cumulative_errors: dict[str, dict[str, np.ndarray | None]], *args, **kwargs) -> dict[str, dict[str, float | str | None]]:
    """Calculate match durations for all trajectories in an ensemble result."""
    return {identifier: calculate_match_duration(errors, *args, **kwargs) for identifier, errors in cumulative_errors.items()}
