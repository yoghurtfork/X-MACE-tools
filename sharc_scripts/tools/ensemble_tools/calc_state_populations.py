"""Get state (S0, S1, S2...) population for a SHARC ensemble"""

from sharc_scripts.tools.parse_sharc_output import SharcEnsemble


def calculate_state_populations(
    ensemble: SharcEnsemble,
    time_fs: float,
) -> dict[str, float]:
    """
    Return populations among the trajectories available at time_fs
    If a trajectory is truncated before time_fs, it is not counted in n_available
    """
    counts: dict[str, int] = {}
    n_available = 0
    for trajectory in ensemble.trajectories:
        # find frames that match the time_fs requested
        matching_frame = next(
            (
                frame
                for frame in trajectory.frames
                if abs(frame.time_fs - time_fs) <= 1.0e-6
            ),
            None,
        )
        if matching_frame is not None:
            state_label = f"S{matching_frame.state - 1}"
            counts[state_label] = counts.get(state_label, 0) + 1
            n_available += 1

    return {
        state_label: count / n_available
        for state_label, count in counts.items()
    }
