"""Callable geometry coordinate definitions for use by plotting modules"""


def _atom_identities(atoms, indices):
    symbols = atoms.get_chemical_symbols()
    return "-".join(symbols[index - 1] for index in indices)


class BondLengthCoordinate:
    """Evaluate a bond length from 1-based atom indices"""

    def __init__(self, indices):
        self.indices = tuple(indices)

    def __call__(self, atoms):
        first, second = (index - 1 for index in self.indices) # convert to 0-based indices
        return float(atoms.get_distance(first, second))

    def label(self, atoms):
        return f"{_atom_identities(atoms, self.indices)} bond length (Å)"


class BondAngleCoordinate:
    """Evaluate a bond angle from 1-based atom indices"""

    def __init__(self, indices):
        self.indices = tuple(indices)

    def __call__(self, atoms):
        first, second, third = (index - 1 for index in self.indices) # convert to 0-based indices
        return float(atoms.get_angle(first, second, third))

    def label(self, atoms):
        return f"{_atom_identities(atoms, self.indices)} bond angle (°)"


class DihedralCoordinate:
    """Evaluate a dihedral angle from 1-based atom indices"""

    def __init__(self, indices):
        self.indices = tuple(indices)

    def __call__(self, atoms):
        first, second, third, fourth = (index - 1 for index in self.indices) # convert to 0-based indices
        return float(atoms.get_dihedral(first, second, third, fourth))

    def label(self, atoms):
        return f"{_atom_identities(atoms, self.indices)} dihedral angle (°)"
