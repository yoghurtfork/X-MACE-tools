"""
Read geometries and energies from SHARC output.dat
at trajectory-level (from TRAJ folders) and ensemble-level (from ensemble root folders)
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys

from ase import Atoms
from ase.units import Bohr, Hartree


# One atomic unit of time in femtoseconds. SHARC writes ``dtstep`` in a.u.
_ATOMIC_TIME_FS = 0.024188843265857
_CUTOFF_TOLERANCE_FS = 1.0e-9


@dataclass
class SharcFrame:
    geometry: Atoms
    step_number: int
    time_fs: float
    state: int
    kinetic_energy_ev: float
    potential_energy_ev: float
    total_energy_ev: float


@dataclass
class SharcTrajectory:
    path: Path
    frames: list[SharcFrame]


@dataclass
class SharcEnsemble:
    path: Path
    trajectories: list[SharcTrajectory]


def read_sharc_trajectory(
    trajectory_path: str | Path,
    maximum_time_fs: float | None = None,
) -> SharcTrajectory | None:
    """Read a SHARC ``TRAJ_*`` directory, or warn and return ``None`` on failure."""
    path = Path(trajectory_path)
    if not path.is_dir() or not path.name.startswith("TRAJ_"):
        print(f"WARNING: Failed to read {path}", file=sys.stderr)
        return None

    current_step: int | None = None
    try:
        with (path / "output.dat").open(encoding="utf-8") as handle:
            # Read the settings and element list preceding the first timestep.
            natom: int | None = None
            dtstep: float | None = None
            nstates: int | None = None
            symbols: list[str] | None = None
            for line in handle:
                fields = line.split()
                label = line.strip()
                if label.startswith("natom"):
                    natom = int(fields[1])
                elif label.startswith("dtstep"):
                    dtstep = float(fields[1].replace("D", "E"))
                elif label.startswith("nstates_m"):
                    states_per_multiplicity = [int(value) for value in fields[1:]]
                    nstates = sum(
                        multiplicity * count
                        for multiplicity, count in enumerate(
                            states_per_multiplicity,
                            start=1,
                        )
                    )
                elif label == "! Elements":
                    if natom is None:
                        raise ValueError("elements precede natom")
                    symbols = []
                    for _ in range(natom):
                        element_line = next((item for item in handle if item.strip()), None)
                        if element_line is None:
                            raise ValueError("incomplete elements block")
                        symbols.append(element_line.split()[0])
                elif label.startswith("! 0 Step"):
                    break
            else:
                raise ValueError("no timestep in output")
            if natom is None or dtstep is None or nstates is None or symbols is None:
                raise ValueError("incomplete SHARC header")

            # Read each labelled timestep, beginning with its SHARC step number.
            frames: list[SharcFrame] = []
            step_line = next((item for item in handle if item.strip()), None)
            if step_line is None:
                raise ValueError("missing first step number")
            step_number = int(step_line.split()[0])
            while True:
                current_step = step_number
                time_fs = step_number * dtstep * _ATOMIC_TIME_FS
                # A later timestep need not be inspected once it exceeds the cutoff.
                if maximum_time_fs is not None and time_fs > maximum_time_fs + _CUTOFF_TOLERANCE_FS:
                    break

                hamiltonian: list[float] | None = None
                kinetic_energy: float | None = None
                state: int | None = None
                coordinates: list[list[float]] | None = None
                next_step: int | None = None

                for line in handle:
                    label = line.strip()
                    if label.startswith("! 0 Step"):
                        # Read the following timestep number for the next loop.
                        next_step_line = next((item for item in handle if item.strip()), None)
                        if next_step_line is None:
                            raise ValueError("missing step number")
                        next_step = int(next_step_line.split()[0])
                        break
                    if label.startswith("! 1 Hamiltonian (MCH)"):
                        # Read the MCH Hamiltonian diagonal for active-state energy.
                        hamiltonian = []
                        for row in range(nstates):
                            matrix_line = next((item for item in handle if item.strip()), None)
                            if matrix_line is None:
                                raise ValueError("incomplete MCH Hamiltonian")
                            hamiltonian.append(float(matrix_line.replace("D", "E").split()[2 * row]))
                    elif label.startswith("! 7 Ekin"):
                        # Read kinetic energy in atomic units.
                        energy_line = next((item for item in handle if item.strip()), None)
                        if energy_line is None:
                            raise ValueError("missing kinetic energy")
                        kinetic_energy = float(energy_line.split()[0].replace("D", "E"))
                    elif label.startswith("! 8 states (diag, MCH)"):
                        # Read the active MCH state (the second state index).
                        state_line = next((item for item in handle if item.strip()), None)
                        if state_line is None:
                            raise ValueError("missing active state")
                        state = int(state_line.split()[1])
                    elif label.startswith("! 11 Geometry"):
                        # Read Cartesian coordinates in Bohr.
                        coordinates = []
                        for _ in range(natom):
                            coordinate_line = next((item for item in handle if item.strip()), None)
                            if coordinate_line is None:
                                raise ValueError("incomplete geometry")
                            fields = coordinate_line.replace("D", "E").split()
                            coordinates.append([float(fields[0]), float(fields[1]), float(fields[2])])

                if hamiltonian is None or kinetic_energy is None or state is None or coordinates is None:
                    raise ValueError("incomplete timestep")
                if not 1 <= state <= len(hamiltonian):
                    raise ValueError("active MCH state outside Hamiltonian")

                potential_energy = hamiltonian[state - 1] * Hartree
                kinetic_energy_ev = kinetic_energy * Hartree
                # Convert to ASE/eV units and retain only the required frame data.
                frames.append(
                    SharcFrame(
                        geometry=Atoms(symbols=symbols, positions=[[value * Bohr for value in row] for row in coordinates]),
                        step_number=step_number,
                        time_fs=time_fs,
                        state=state,
                        kinetic_energy_ev=kinetic_energy_ev,
                        potential_energy_ev=potential_energy,
                        total_energy_ev=kinetic_energy_ev + potential_energy,
                    )
                )
                if maximum_time_fs is not None and abs(time_fs - maximum_time_fs) <= _CUTOFF_TOLERANCE_FS:
                    break
                if next_step is None:
                    break
                step_number = next_step

    except (OSError, UnicodeError, ValueError):
        suffix = "" if current_step is None else f": error at step number {current_step}"
        print(f"WARNING: Failed to read {path}{suffix}", file=sys.stderr)
        return None

    return SharcTrajectory(path=path, frames=frames)


def read_sharc_ensemble(
    ensemble_path: str | Path,
    maximum_time_fs: float | None = None,
) -> SharcEnsemble:
    """Recursively read all successful SHARC ``TRAJ_*`` directories in an ensemble."""
    path = Path(ensemble_path)
    trajectories: list[Path] = []
    # Discover trajectory directories recursively
    for root, directories, _ in os.walk(path):
        found = [name for name in directories if name.startswith("TRAJ_")]
        trajectories.extend(Path(root) / name for name in found)
        directories[:] = [name for name in directories if name not in found]

    # Sort trajectories (eg Singlet_1 comes before Singlet_2, TRAJ_00002 comes before TRAJ_00010)
    trajectories.sort(
        key=lambda candidate: _sort_key(candidate.relative_to(path))
    )

    successful: list[SharcTrajectory] = []
    # Read each discovered trajectory, omitting only those that fail individually.
    for candidate in trajectories:
        trajectory = read_sharc_trajectory(candidate, maximum_time_fs)
        if trajectory is not None:
            successful.append(trajectory)
    return SharcEnsemble(path=path, trajectories=successful)


def _sort_key(path: Path) -> tuple[tuple[tuple[int, object], ...], ...]:
    """Helps in sorting trajectories in an ensemble"""
    return tuple(
        tuple(
            (1, int(part)) if part.isdigit() else (0, part.casefold())
            for part in re.split(r"(\d+)", component)
        )
        for component in path.parts
    )
