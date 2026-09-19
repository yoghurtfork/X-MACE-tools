"""Plot selected bond lengths against time"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parse_sharc_output import SharcTrajectory


def plot_trajectory_bond_lengths(
    trajectory: SharcTrajectory,
    bonds: list[tuple[int, int]],
) -> tuple[Figure, Axes]:
    bonds_zero_based = [
        (first - 1, second - 1)
        for first, second in bonds
    ]
    times = [frame.time_fs for frame in trajectory.frames]

    fig, ax = plt.subplots()
    for (first, second), (first_zero_based, second_zero_based) in zip(
        bonds, bonds_zero_based
    ):
        ax.plot(
            times,
            [
                frame.geometry.get_distance(first_zero_based, second_zero_based)
                for frame in trajectory.frames
            ],
            label=(
                f"{trajectory.frames[0].geometry[first_zero_based].symbol}{first}–"
                f"{trajectory.frames[0].geometry[second_zero_based].symbol}{second}"
            ) if trajectory.frames else f"{first}–{second}",
        )
    ax.set(xlabel="Time (fs)", ylabel="Bond length (Å)", title=trajectory.path.name)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    return fig, ax
