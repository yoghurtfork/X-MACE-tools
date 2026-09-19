"""Filter for a SharcEnsemble"""

from pathlib import Path

from data_helpers.filtering.filter_sharc_trajectory import filter_sharc_trajectory
from sharc_scripts.tools.parse_sharc_output import SharcEnsemble


def filter_sharc_ensemble(
    ensemble: SharcEnsemble,
    filter_settings_path: str | Path,
    bonds: list[tuple[int, int]],
) -> dict[str, object]:
    """Return all filter results and an ensemble with trajectories truncated right before the first failed geometry"""
    trajectory_results = [
        filter_sharc_trajectory(trajectory, filter_settings_path, bonds)
        for trajectory in ensemble.trajectories
    ]
    return {
        "ensemble": SharcEnsemble(
            path=ensemble.path,
            trajectories=[
                result["filtered_trajectory"] for result in trajectory_results
            ],
        ),
        "trajectory_results": trajectory_results,
    }
