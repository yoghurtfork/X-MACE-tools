"""Sample geometries near a conical intersection"""

import ase
import numpy as np


def choose_sigma(gaps: np.ndarray, n_samples: int) -> float:
    """
    Near-CI sampling probabilities are proportional to exp(-0.5 * (gap / sigma) ** 2)
    This function chooses a suitable sigma (the smallest sigma that ensures effective_sample_size >= 2 * n_samples)
    """
    candidates = np.array(
        [
            0.01,
            0.02,
            0.03,
            0.05,
            0.075,
            0.10,
            0.125,
            0.15,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
            0.75,
            1.00,
        ]
    )

    for sigma in candidates:
        weights = np.exp(-0.5 * (gaps / sigma) ** 2)
        probabilities = weights / np.sum(weights)
        effective_sample_size = 1.0 / np.sum(probabilities**2)
        if effective_sample_size >= 2 * n_samples:
            print(f"Sigma used: {sigma} eV")
            return float(sigma)

    sigma = candidates[-1]
    print("Warning: no sigma met the diversity target.")
    print(f"Sigma used: {sigma} eV")
    return float(sigma)


def sample_near_CI(
    atoms_list, CI, n_samples, seed=42, sigma="auto"
) -> list[ase.Atoms]:
    """
    Sample geometries with probability concentrated at CIs
    Probabilities are proportional to exp(-0.5 * (gap / sigma) ** 2)
    """
    # first calculate the energy gaps between the states you want to get CIs for
    first_state = int(CI[0][1:])
    second_state = int(CI[1][1:])
    gaps = np.array(
        [
            np.asarray(atoms.info["REF_energy"]).reshape(-1)[second_state]
            - np.asarray(atoms.info["REF_energy"]).reshape(-1)[first_state]
            for atoms in atoms_list
        ],
        dtype=float,
    )
    finite = np.isfinite(gaps)
    gaps = gaps[finite]
    candidates = [atoms for atoms, is_finite in zip(atoms_list, finite) if is_finite]

    # automatically choose a good sigma
    if sigma == "auto":
        sigma = choose_sigma(gaps, n_samples)

    # randomly sample from Gaussian distribution
    weights = np.exp(-0.5 * (gaps / sigma) ** 2)
    probabilities = weights / np.sum(weights)
    indices = np.random.default_rng(seed).choice(
        len(candidates), size=n_samples, replace=False, p=probabilities
    )
    return [candidates[index].copy() for index in indices]
