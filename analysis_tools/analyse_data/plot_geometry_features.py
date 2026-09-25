"""Plot histograms of geometry features across dataset"""

import matplotlib.pyplot as plt

from analysis_tools.analyse_data.geometry_coordinates import (
    BondAngleCoordinate,
    BondLengthCoordinate,
    DihedralCoordinate,
)


def _plot_distribution(atoms_list, atom_index_sets, coordinate_type, title, bins):
    # create a Coordinate object for each set in atom_index_sets
    coordinates = [coordinate_type(indices) for indices in atom_index_sets]
    # call Coordinate objects for each ASE Atoms object to get values
    values = [coordinate(atoms) for atoms in atoms_list for coordinate in coordinates]
    # plot
    fig, ax = plt.subplots(constrained_layout=True)
    ax.hist(values, bins=bins, color="tab:blue", edgecolor="white")
    ax.set(title=title, xlabel=coordinates[0].label(atoms_list[0]), ylabel="Count")
    ax.grid(axis="y", alpha=0.3)
    return fig, ax


def plot_bond_length_distribution(atoms_list, atom_index_sets, bins=30):
    """Plot distribution of bond lengths specified by atom_index_sets"""
    return _plot_distribution(
        atoms_list,
        atom_index_sets,
        BondLengthCoordinate,
        "Bond length distribution",
        bins,
    )


def plot_bond_angle_distribution(atoms_list, atom_index_sets, bins=30):
    """Plot distribution of angles specified by atom_index_sets"""
    return _plot_distribution(
        atoms_list,
        atom_index_sets,
        BondAngleCoordinate,
        "Bond angle distribution",
        bins,
    )


def plot_dihedral_distribution(atoms_list, atom_index_sets, bins=30):
    """Plot distribution of dihedral angles specified by atom_index_sets"""
    return _plot_distribution(
        atoms_list,
        atom_index_sets,
        DihedralCoordinate,
        "Dihedral angle distribution",
        bins,
    )
