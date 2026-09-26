"""2D projections of dataset with energy information encoded as colour"""

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def _get_axes(atoms_list, x_coordinate, y_coordinate):
    # use callable Coordinates to get x and y axes
    x_values = np.array([x_coordinate(atoms) for atoms in atoms_list])
    y_values = np.array([y_coordinate(atoms) for atoms in atoms_list])
    # get energies from ASE Atoms objects
    energies = np.vstack([atoms.info["REF_energy"] for atoms in atoms_list])
    return x_values, y_values, energies


def _plot_2d_surface(atoms_list, x_coordinate, y_coordinate, n_states, title):
    fig, ax = plt.subplots(
        1,
        n_states,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(5 * n_states, 4),
        constrained_layout=True,
    )
    ax = ax.ravel()
    fig.suptitle(title)
    for state, axis in enumerate(ax):
        axis.set_title(f"S{state}")
        axis.set_xlabel(x_coordinate.label(atoms_list[0]))
        axis.set_ylabel(y_coordinate.label(atoms_list[0]))
    return fig, ax


def plot_energy_scatter(atoms_list, x_coordinate, y_coordinate, cmap="viridis"):
    """Plot 2D scatterplots coloured by energy"""
    x_values, y_values, energies = _get_axes(
        atoms_list, x_coordinate, y_coordinate
    )
    fig, ax = _plot_2d_surface(
        atoms_list,
        x_coordinate,
        y_coordinate,
        energies.shape[1],
        "Scatterplot coloured by energy",
    )
    for state, axis in enumerate(ax):
        points = axis.scatter(
            x_values,
            y_values,
            c=energies[:, state].reshape(-1, 1),
            cmap=cmap,
            s=25,
            alpha=0.75,
            edgecolors="none",
        )
        fig.colorbar(points, ax=axis, label="Energy (eV)")
    return fig, ax


def plot_energy_mesh_triangulate(
    atoms_list, x_coordinate, y_coordinate, cmap="viridis", shading="gouraud"
):
    """Plot triangulated surfaces coloured by energy"""
    x_values, y_values, energies = _get_axes(
        atoms_list, x_coordinate, y_coordinate
    )

    # finds distinct (x,y) pairs and uses the average if there are geometries with the same (x,y)
    unique_coordinates, inverse = np.unique(
        np.column_stack((x_values, y_values)), axis=0, return_inverse=True
    )
    counts = np.bincount(inverse)
    summed_energies = np.zeros((len(unique_coordinates), energies.shape[1]))
    np.add.at(summed_energies, inverse, energies)
    mean_energies = summed_energies / counts[:, None]
    triangulation = mtri.Triangulation(*unique_coordinates.T)

    fig, ax = _plot_2d_surface(
        atoms_list,
        x_coordinate,
        y_coordinate,
        energies.shape[1],
        "Triangulated energy surfaces",
    )
    for state, axis in enumerate(ax):
        surface = axis.tripcolor(
            triangulation, mean_energies[:, state], cmap=cmap, shading=shading
        )
        fig.colorbar(surface, ax=axis, label="Energy (eV)")
    return fig, ax
