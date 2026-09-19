"""
Filter for a single SharcTrajectory
Input the SharcTrajectory, filter settings JSON, and list of bond indices
Returns the failed criteria, filtered SharcTrajectory, step number, and time
"""

import json
from pathlib import Path

from sharc_scripts.tools.parse_sharc_output import SharcTrajectory


def filter_sharc_trajectory(
    trajectory: SharcTrajectory,
    filter_settings_path: str | Path,
    bonds: list[tuple[int, int]],
) -> dict[str, object]:
    """Return filter criteria and the trajectory before the first failed frame"""
    with Path(filter_settings_path).open(encoding="utf-8") as handle:
        settings = json.load(handle)

    bond_limits = settings["bond_length_limits_angstrom"]
    overrides = bond_limits["overrides"]
    energy_limits = settings["energy_limits_ev"]
    zero_based_bonds = [(first - 1, second - 1) for first, second in bonds] # convert to zero-based atomic indices

    for index, frame in enumerate(trajectory.frames):
        failed_criteria: list[dict[str, str]] = []

        # check geometry criteria
        for (first, second), (first_zero_based, second_zero_based) in zip(
            bonds, zero_based_bonds
        ):
            first_symbol = frame.geometry[first_zero_based].symbol
            second_symbol = frame.geometry[second_zero_based].symbol
            forward_key = f"{first_symbol}-{second_symbol}"
            reverse_key = f"{second_symbol}-{first_symbol}" # check the JSON for both C-H and H-C
            if forward_key in overrides:
                limit = overrides[forward_key]
            elif reverse_key in overrides:
                limit = overrides[reverse_key]
            else:
                limit = bond_limits["all_bonds"]
            distance = frame.geometry.get_distance(first_zero_based, second_zero_based)
            if distance > limit:
                failed_criteria.append(
                    {
                        "criterion": "bond_length",
                        "message": (
                            f"{first_symbol}{first}-{second_symbol}{second} bond length "
                            f"{distance:.3f} Å exceeds {limit:.3f} Å"
                        ),
                    }
                )

        # check energy criteria
        if index > 0:
            previous = trajectory.frames[index - 1]
            potential_step = abs(
                frame.potential_energy_ev - previous.potential_energy_ev
            )
            is_hop = frame.state != previous.state
            if is_hop and potential_step > energy_limits["hop_potential_step"]:
                failed_criteria.append(
                    {
                        "criterion": "hop_potential_step",
                        "message": (
                            f"Hop (S{previous.state - 1} -> S{frame.state - 1}) potential "
                            f"energy step {potential_step:.3f} eV exceeds "
                            f"{energy_limits['hop_potential_step']:.3f} eV"
                        ),
                    }
                )
            elif not is_hop and potential_step > energy_limits["active_potential_step"]:
                failed_criteria.append(
                    {
                        "criterion": "active_potential_step",
                        "message": (
                            f"Active potential energy step {potential_step:.3f} eV "
                            f"exceeds {energy_limits['active_potential_step']:.3f} eV"
                        ),
                    }
                )

            total_energy_step = abs(frame.total_energy_ev - previous.total_energy_ev)
            if total_energy_step > energy_limits["total_energy_step"]:
                failed_criteria.append(
                    {
                        "criterion": "total_energy_step",
                        "message": (
                            f"Total energy step {total_energy_step:.3f} eV exceeds "
                            f"{energy_limits['total_energy_step']:.3f} eV"
                        ),
                    }
                )

        total_energy_drift = abs(
            frame.total_energy_ev - trajectory.frames[0].total_energy_ev
        )
        if total_energy_drift > energy_limits["total_energy_drift"]:
            failed_criteria.append(
                {
                    "criterion": "total_energy_drift",
                    "message": (
                        f"Total energy drift {total_energy_drift:.3f} eV exceeds "
                        f"{energy_limits['total_energy_drift']:.3f} eV relative to step "
                        f"{trajectory.frames[0].step_number}"
                    ),
                }
            )

        if index > 0:
            kinetic_energy_step = abs(
                frame.kinetic_energy_ev - previous.kinetic_energy_ev
            )
            if not is_hop and kinetic_energy_step > energy_limits["kinetic_energy_step"]:
                failed_criteria.append(
                    {
                        "criterion": "kinetic_energy_step",
                        "message": (
                            f"Kinetic energy step {kinetic_energy_step:.3f} eV exceeds "
                            f"{energy_limits['kinetic_energy_step']:.3f} eV"
                        ),
                    }
                )

        if failed_criteria:
            return {
                "filtered": True,
                "failed_criteria": failed_criteria,
                "filtered_trajectory": SharcTrajectory(
                    path=trajectory.path,
                    frames=trajectory.frames[:index], # cuts before the first failed frame
                ),
                "step_number": frame.step_number,
                "time_fs": frame.time_fs,
            }

    return {
        "filtered": False,
        "failed_criteria": [],
        "filtered_trajectory": SharcTrajectory(
            path=trajectory.path,
            frames=trajectory.frames[:],
        ),
        "step_number": None,
        "time_fs": None,
    }
