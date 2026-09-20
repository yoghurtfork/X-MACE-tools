"""
Read extended XYZ files containing the energies, forces, and NACs 
for comparison against SHARC trajectories
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np


@dataclass
class ComparisonFrame:
    energies_ev: dict[int, float]
    forces_ev_per_angstrom: dict[int, np.ndarray] | None
    nacs_per_angstrom: dict[tuple[int, int], np.ndarray] | None
    time_fs: float | None = None


@dataclass
class ComparisonTrajectory:
    path: Path
    trajectory_id: str
    frames: list[ComparisonFrame]


@dataclass
class ComparisonEnsemble:
    path: Path
    trajectories: list[ComparisonTrajectory]


def read_comparison_trajectory(path: str | Path) -> ComparisonTrajectory:
    """Read an extended XYZ file containing a trajectory"""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    frames: list[ComparisonFrame] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        natoms = int(lines[cursor])
        comment = lines[cursor + 1]
        energies = np.asarray(json.loads(re.search(r'REF_energy="_JSON (.*?)"', comment).group(1)), dtype=float)[0]
        force_tensor = np.swapaxes(
            np.asarray(json.loads(re.search(r'REF_forces="_JSON (.*?)"', comment).group(1)), dtype=float),
            0,
            1,
        )
        state_nacs = None
        nac_match = re.search(r'REF_nacs="_JSON (.*?)"', comment)
        if nac_match is not None:
            nac_tensor = np.swapaxes(np.asarray(json.loads(nac_match.group(1)), dtype=float), 0, 1)
            pairs = [
                (first, second)
                for first in range(1, len(energies) + 1)
                for second in range(first + 1, len(energies) + 1)
            ]
            state_nacs = dict(zip(pairs, nac_tensor))
        time_match = re.search(r"\bt=\s*([-+0-9.eE]+)", comment)
        frames.append(
            ComparisonFrame(
                energies_ev={state: value for state, value in enumerate(energies, start=1)},
                forces_ev_per_angstrom={
                    state: force_tensor[state - 1]
                    for state in range(1, force_tensor.shape[0] + 1)
                },
                nacs_per_angstrom=state_nacs,
                time_fs=None if time_match is None else float(time_match.group(1)),
            )
        )
        cursor += natoms + 2
    if not frames:
        raise ValueError(f"{path}: no XYZ frames found")
    trajectory_path = path.parent
    return ComparisonTrajectory(trajectory_path, trajectory_path.name, frames)


def read_comparison_ensemble(
    root: str | Path,
    comparison_glob: str = "*_CASSCF.xyz",
) -> ComparisonEnsemble:
    """Discover and read one comparison XYZ file from every ``TRAJ_*`` directory."""
    root = Path(root)
    trajectories = sorted(
        (candidate for candidate in root.rglob("TRAJ_*") if candidate.is_dir()),
        key=lambda candidate: int(re.search(r"\d+$", candidate.name).group()),
    )
    if not trajectories:
        raise ValueError(f"{root}: no TRAJ_* directories found")
    result: list[ComparisonTrajectory] = []
    for trajectory in trajectories:
        candidates = sorted(trajectory.glob(comparison_glob))
        if len(candidates) != 1:
            raise ValueError(
                f"{trajectory}: expected exactly one comparison file matching {comparison_glob!r}, "
                f"found {len(candidates)}"
            )
        result.append(read_comparison_trajectory(candidates[0]))
    return ComparisonEnsemble(root, result)
