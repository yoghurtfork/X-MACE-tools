"""Plot a trajectory using bond length and dihedral"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parse_sharc_output import SharcTrajectory


def plot_trajectory_2d(
    trajectory: SharcTrajectory,
    bond_length_indices: tuple[int, int],
    dihedral_indices: tuple[int, int, int, int],
) -> tuple[Figure, Axes]:
    bond_indices_zero_based = tuple(index - 1 for index in bond_length_indices)
    dihedral_indices_zero_based = tuple(index - 1 for index in dihedral_indices)
    bond_lengths = [
        frame.geometry.get_distance(*bond_indices_zero_based)
        for frame in trajectory.frames
    ]
    dihedrals = [
        frame.geometry.get_dihedral(*dihedral_indices_zero_based)
        for frame in trajectory.frames
    ]
    times = [frame.time_fs for frame in trajectory.frames]

    fig, ax = plt.subplots()
    points = ax.scatter(bond_lengths, dihedrals, c=times, cmap="viridis")
    fig.colorbar(points, ax=ax, label="Time (fs)")
    if trajectory.frames:
        ax.scatter(
            bond_lengths[0], dihedrals[0], color="red", edgecolor="white",
            label="Initial geometry",
        )
    hop_indices = [
        index
        for index, (previous, current) in enumerate(
            zip(trajectory.frames, trajectory.frames[1:]), start=1
        )
        if previous.state != current.state
    ]
    if hop_indices:
        ax.scatter(
            [bond_lengths[index] for index in hop_indices],
            [dihedrals[index] for index in hop_indices],
            color="blue", edgecolor="white",
            label="Hop geometry",
        )
    ax.set(
        xlabel="Bond length " + "–".join(map(str, bond_length_indices)) + " (Å)",
        ylabel="Dihedral " + "–".join(map(str, dihedral_indices)) + " (°)",
        title=trajectory.path.name,
    )
    ax.grid()
    ax.legend()
    fig.tight_layout()
    return fig, ax
