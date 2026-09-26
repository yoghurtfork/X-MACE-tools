"""Plots of X-MACE predicted energy surfaces"""

import matplotlib.pyplot as plt
import numpy as np


def plot_pred_energy_mesh(atoms_list, x_coordinate, y_coordinate, cmap="viridis"):
    """Plot 2D surface of predicted energy"""
    x_values = np.array([x_coordinate(atoms) for atoms in atoms_list])
    y_values = np.array([y_coordinate(atoms) for atoms in atoms_list])
    energies = np.vstack([np.asarray(atoms.info["PRED_energy"]) for atoms in atoms_list])
    x_grid = np.unique(x_values)
    y_grid = np.unique(y_values)

    # if there are multiple geometries with the same (x,y), take the average
    x_indices = np.searchsorted(x_grid, x_values)
    y_indices = np.searchsorted(y_grid, y_values)
    sums = np.zeros((len(y_grid), len(x_grid), energies.shape[1]))
    counts = np.zeros((len(y_grid), len(x_grid)), dtype=int)
    np.add.at(sums, (y_indices, x_indices), energies)
    np.add.at(counts, (y_indices, x_indices), 1)
    grid = np.full_like(sums, np.nan)
    grid[counts > 0] = sums[counts > 0] / counts[counts > 0, None]

    # plot
    fig, ax = plt.subplots(
        1,
        energies.shape[1],
        squeeze=False,
        figsize=(5 * energies.shape[1], 4),
        constrained_layout=True,
    )
    ax = ax.ravel()
    fig.suptitle("Predicted energy surfaces")
    for state, axis in enumerate(ax):
        surface = axis.pcolormesh(x_grid, y_grid, grid[:, :, state], cmap=cmap, shading="nearest")
        axis.set_title(f"S{state}")
        axis.set_xlabel(x_coordinate.label(atoms_list[0]))
        axis.set_ylabel(y_coordinate.label(atoms_list[0]))
        fig.colorbar(surface, ax=axis, label="Energy (eV)")
    return fig, ax
