"""Cumulative MAEs for SHARC MD match"""

from __future__ import annotations

import numpy as np
import torch

from mace.modules.nac_utils import align_geometry_nacs, enumerate_nac_phase_signs

from sharc_scripts.tools.md_match.parse_comparison import ComparisonEnsemble, ComparisonTrajectory
from sharc_scripts.tools.parse_sharc_output import SharcEnsemble, SharcTrajectory


def calculate_cumulative_error(
    sharc_trajectory: SharcTrajectory,
    comparison_trajectory: ComparisonTrajectory,
    min_energy_error: float,
    min_force_error: float,
    min_nac_error: float | None = None,
) -> dict[str, np.ndarray | None]:
    """Return cumulative energy, force, and raw NAC MAEs
    MAEs below the min threshold are not counted
    """
    common_length = min(len(sharc_trajectory.frames), len(comparison_trajectory.frames))

    energy_errors, force_errors, nac_errors = [], [], []
    for frame_index, (sharc, comparison) in enumerate(
        zip(sharc_trajectory.frames[:common_length], comparison_trajectory.frames[:common_length])
    ):
        # energy MAEs
        sharc_states = list(sharc.energies_ev)
        states = sorted(sharc_states)
        sharc_energies = np.asarray([sharc.energies_ev[state] for state in states])
        comparison_energies = np.asarray(
            [comparison.energies_ev[state] - sharc.ezero_ev for state in states] # account for SHARC ezero
        )
        energy_errors.append(float(np.mean(np.abs(sharc_energies - comparison_energies))))

        # force MAEs
        sharc_forces = np.asarray([sharc.forces_ev_per_angstrom[state] for state in states])
        comparison_forces = np.asarray(
            [comparison.forces_ev_per_angstrom[state] for state in states]
        )
        force_errors.append(float(np.mean(np.abs(sharc_forces - comparison_forces))))

        # raw NACs MAEs
        if min_nac_error is not None: # NACs are optional
            canonical_pairs = [
                (states[first], states[second])
                for first, second in zip(*np.triu_indices(len(states), k=1))
            ]
            sharc_nacs, comparison_nacs = [], []
            for first, second in canonical_pairs:
                # by searching through state pairs, find the NACs to compare
                if (first, second) in sharc.nacs_per_angstrom:
                    sharc_nac = sharc.nacs_per_angstrom[(first, second)]
                elif (second, first) in sharc.nacs_per_angstrom:
                    sharc_nac = -sharc.nacs_per_angstrom[(second, first)]
                else:
                    raise ValueError(f"frame {frame_index}: SHARC NAC pair {(first, second)} is missing")
                if (first, second) in comparison.nacs_per_angstrom:
                    comparison_nac = comparison.nacs_per_angstrom[(first, second)]
                elif (second, first) in comparison.nacs_per_angstrom:
                    comparison_nac = -comparison.nacs_per_angstrom[(second, first)]
                else:
                    raise ValueError(f"frame {frame_index}: comparison NAC pair {(first, second)} is missing")

                # SHARC writes a -123 placeholder if NACs are unavailable
                if np.all(np.isclose(np.abs(sharc_nac), 123.0 / 0.5291772105638411)):
                    raise ValueError(f"frame {frame_index}: SHARC NAC values are placeholders")

                sharc_nacs.append(sharc_nac)
                comparison_nacs.append(comparison_nac)
            sharc_nac_tensor = torch.as_tensor(np.stack(sharc_nacs, axis=1))
            comparison_nac_tensor = torch.as_tensor(np.stack(comparison_nacs, axis=1))

            # using X-MACE-TL's nac utils, find the consistent phase assignment that gives the smallest error
            pair_signs = sharc_nac_tensor.new_tensor(enumerate_nac_phase_signs(len(states)))
            residual = align_geometry_nacs(sharc_nac_tensor, comparison_nac_tensor, pair_signs)
            nac_errors.append(torch.mean(torch.abs(residual)).item())

    energy_errors = np.asarray(energy_errors)
    force_errors = np.asarray(force_errors)
    return {
        "time": np.asarray([frame.time_fs for frame in sharc_trajectory.frames[:common_length]]),
        "energy_cumulative_error": np.cumsum(np.where(energy_errors >= min_energy_error, energy_errors, 0.0)),
        "forces_cumulative_error": np.cumsum(np.where(force_errors >= min_force_error, force_errors, 0.0)),
        "nacs_cumulative_error": None if min_nac_error is None else np.cumsum(np.where(np.asarray(nac_errors) >= min_nac_error, nac_errors, 0.0)),
    }


def calculate_ensemble_cumulative_error(
    sharc_ensemble: SharcEnsemble, comparison_ensemble: ComparisonEnsemble, *args, **kwargs
) -> dict[str, dict[str, np.ndarray | None]]:
    """Calculate cumulative all-state MAEs for trajectories matched by identifier."""
    sharc = {trajectory.path.name: trajectory for trajectory in sharc_ensemble.trajectories}
    comparison = {trajectory.trajectory_id: trajectory for trajectory in comparison_ensemble.trajectories}
    return {
        identifier: calculate_cumulative_error(sharc[identifier], comparison[identifier], *args, **kwargs)
        for identifier in sorted(sharc)
    }
