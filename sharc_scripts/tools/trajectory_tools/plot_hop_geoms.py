"""Plot hop geometries in a trajectory"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parse_sharc_output import SharcTrajectory


def plot_hop_geoms(
    trajectory: SharcTrajectory,
    bond_length_indices: tuple[int, int],
    dihedral_indices: tuple[int, int, int, int],
) -> tuple[Figure, Axes]:
    bond_indices_zero_based = tuple(index - 1 for index in bond_length_indices)
    dihedral_indices_zero_based = tuple(index - 1 for index in dihedral_indices)
    hops = [
        (previous.state, current.state, current)
        for previous, current in zip(trajectory.frames, trajectory.frames[1:])
        if previous.state != current.state
    ]

    fig, ax = plt.subplots()
    if hops:
        for from_state, to_state in sorted({hop[:2] for hop in hops}):
            hop_frames = [
                frame
                for previous_state, current_state, frame in hops
                if (previous_state, current_state) == (from_state, to_state)
            ]
            ax.scatter(
                [frame.geometry.get_distance(*bond_indices_zero_based) for frame in hop_frames],
                [frame.geometry.get_dihedral(*dihedral_indices_zero_based) for frame in hop_frames],
                label=f"S{from_state - 1} → S{to_state - 1}",
            )
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No hops observed.", ha="center", va="center", transform=ax.transAxes)
    ax.set(
        xlabel="Bond length " + "–".join(map(str, bond_length_indices)) + " (Å)",
        ylabel="Dihedral " + "–".join(map(str, dihedral_indices)) + " (°)",
        title=trajectory.path.name,
    )
    ax.grid()
    fig.tight_layout()
    return fig, ax
