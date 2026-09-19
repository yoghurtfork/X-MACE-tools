"""Get quantum yield for a SHARC ensemble"""

from sharc_scripts.tools.parse_sharc_output import SharcEnsemble


def calculate_quantum_yield(
    ensemble: SharcEnsemble,
    dihedral_indices: tuple[int, int, int, int],
    time_fs: float,
) -> dict[str, float | int]:
    """
    At time_fs, look at all the trajectories in ground state (exclude all excited states)
    Quantum yield is n_flipped / (n_flipped + n_not_flipped)
    'Flipped' means its cis/trans status changed between initial and current
    """
    zero_based_indices = tuple(index - 1 for index in dihedral_indices)
    n_excited_state = n_flipped = n_not_flipped = n_unavailable = 0

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
        if matching_frame is None:
            n_unavailable += 1
        elif matching_frame.state != 1:
            n_excited_state += 1
        else:
            initial_angle = trajectory.frames[0].geometry.get_dihedral(
                *zero_based_indices
            )
            final_angle = matching_frame.geometry.get_dihedral(*zero_based_indices)
            initial_is_cis = initial_angle <= 90.0 or initial_angle >= 270.0
            final_is_cis = final_angle <= 90.0 or final_angle >= 270.0
            if initial_is_cis != final_is_cis:
                n_flipped += 1
            else:
                n_not_flipped += 1

    denominator = n_flipped + n_not_flipped
    return {
        "quantum_yield": n_flipped / denominator if denominator else float("nan"),
        "n_excited_state": n_excited_state,
        "n_flipped": n_flipped,
        "n_not_flipped": n_not_flipped,
        "n_unavailable": n_unavailable,
    }
