"""Plots for energy residuals (PRED_energy - REF_energy)"""

import matplotlib.pyplot as plt
import numpy as np


def _get_residuals(atoms_list, absolute):
    predicted = np.vstack([np.asarray(atoms.info["PRED_energy"]) for atoms in atoms_list])
    reference = np.vstack([np.asarray(atoms.info["REF_energy"]) for atoms in atoms_list])
    residuals = predicted - reference
    return reference, np.abs(residuals) if absolute else residuals


def _residual_label(absolute):
    return "Absolute energy residual (eV)" if absolute else "Energy residual (eV)"


def plot_residual_heatmaps(atoms_list, x_coordinate, y_coordinate, absolute=True):
    """Plot residual heatmaps on a 2D surface"""
    _, residuals = _get_residuals(atoms_list, absolute)

    # get x and y axes
    x_values = np.array([x_coordinate(atoms) for atoms in atoms_list])
    y_values = np.array([y_coordinate(atoms) for atoms in atoms_list])
    x_grid = np.unique(x_values)
    y_grid = np.unique(y_values)

    # if there are multiple geometries with the same (x,y), take the average
    x_indices = np.searchsorted(x_grid, x_values)
    y_indices = np.searchsorted(y_grid, y_values)
    sums = np.zeros((len(y_grid), len(x_grid), residuals.shape[1]))
    counts = np.zeros((len(y_grid), len(x_grid)), dtype=int)
    np.add.at(sums, (y_indices, x_indices), residuals)
    np.add.at(counts, (y_indices, x_indices), 1)
    grid = np.full_like(sums, np.nan)
    grid[counts > 0] = sums[counts > 0] / counts[counts > 0, None]

    # plot
    fig, ax = plt.subplots(1, residuals.shape[1], squeeze=False,
                           figsize=(5 * residuals.shape[1], 4), constrained_layout=True)
    ax = ax.ravel()
    fig.suptitle("Energy residual surfaces")
    vmax = np.nanmax(np.abs(grid)) if not absolute else None
    for state, axis in enumerate(ax):
        options = {"cmap": "magma"} if absolute else {"cmap": "coolwarm", "vmin": -vmax, "vmax": vmax}
        surface = axis.pcolormesh(x_grid, y_grid, grid[:, :, state], shading="nearest", **options)
        axis.set_title(f"S{state}")
        axis.set_xlabel(x_coordinate.label(atoms_list[0]))
        axis.set_ylabel(y_coordinate.label(atoms_list[0]))
        fig.colorbar(surface, ax=axis, label=_residual_label(absolute))
    return fig, ax


def plot_residual_vs_coordinate(atoms_list, coordinate, absolute=True):
    """Plot scatterplots of residual vs geometry coordinate"""
    _, residuals = _get_residuals(atoms_list, absolute)

    # get values of coordinate
    coordinate_values = np.array([coordinate(atoms) for atoms in atoms_list])

    # plot
    fig, ax = plt.subplots(1, residuals.shape[1], squeeze=False,
                           figsize=(5 * residuals.shape[1], 4), constrained_layout=True)
    ax = ax.ravel()
    for state, axis in enumerate(ax):
        axis.scatter(coordinate_values, residuals[:, state], s=25, alpha=0.75)
        axis.set_title(f"S{state}")
        axis.set_xlabel(coordinate.label(atoms_list[0]))
        axis.set_ylabel(_residual_label(absolute))
        axis.grid(alpha=0.3)
    return fig, ax


def plot_residual_vs_gap(atoms_list, gap_states=(0, 1), absolute=True):
    """Plot scatterplots of residual vs energy gap"""
    reference, residuals = _get_residuals(atoms_list, absolute)

    # get energy gaps
    lower_state, upper_state = gap_states
    reference_gap = reference[:, upper_state] - reference[:, lower_state]

    # plot
    fig, ax = plt.subplots(1, residuals.shape[1], squeeze=False,
                           figsize=(5 * residuals.shape[1], 4), constrained_layout=True)
    ax = ax.ravel()
    for state, axis in enumerate(ax):
        axis.scatter(reference_gap, residuals[:, state], s=25, alpha=0.75)
        axis.set_title(f"S{upper_state}−S{lower_state}")
        axis.set_xlabel(f"S{upper_state}−S{lower_state} energy gap (eV)")
        axis.set_ylabel(_residual_label(absolute))
        axis.grid(alpha=0.3)
    return fig, ax
