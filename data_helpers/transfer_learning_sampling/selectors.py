"""
Select transfer learning sample using descriptor vectors
select_atoms(list of descriptors, selector, n samples, kwargs) is the main function
that returns the list of selected geometries
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from ase import Atoms
from skmatter.sample_selection import FPS
from sklearn.cluster import KMeans


def random_sampling(
    descriptor_matrix: np.ndarray,
    n_samples: int,
    seed: int | None = None,
) -> np.ndarray:
    """Select randomly uniformly without replacement"""
    return np.random.default_rng(seed).choice(
        descriptor_matrix.shape[0], size=n_samples, replace=False
    )


def farthest_point_sampling(
    descriptor_matrix: np.ndarray,
    n_samples: int,
    initialize: int | str = 0,
) -> np.ndarray:
    """Start with one geometry, then repeatedly select the geometry whose nearest selected neighbour is as far away as possible"""
    selector = FPS(n_to_select=n_samples, initialize=initialize)
    selector.fit(descriptor_matrix)
    return np.asarray(selector.selected_idx_, dtype=int)


def _allocate_evenly(cluster_sizes: np.ndarray, n_samples: int) -> np.ndarray:
    """Allocate number of samples per cluster for K-means weighted=False"""
    allocation = np.zeros(len(cluster_sizes), dtype=int)
    remaining = n_samples
    while remaining:
        eligible = np.flatnonzero(allocation < cluster_sizes)
        for cluster in eligible:
            if remaining == 0:
                break
            allocation[cluster] += 1
            remaining -= 1
    return allocation


def _allocate_weighted(cluster_sizes: np.ndarray, n_samples: int) -> np.ndarray:
    """Allocate number of samples per cluster for K-means weighted=True"""
    ideal = cluster_sizes / cluster_sizes.sum() * n_samples
    allocation = np.minimum(np.floor(ideal).astype(int), cluster_sizes)
    remaining = n_samples - int(allocation.sum())

    while remaining:
        capacity = cluster_sizes - allocation
        eligible = np.flatnonzero(capacity > 0)
        priorities = ideal[eligible] - allocation[eligible]
        cluster = eligible[int(np.argmax(priorities))]
        allocation[cluster] += 1
        remaining -= 1

    return allocation


def kmeans_sampling(
    descriptor_matrix: np.ndarray,
    n_samples: int,
    seed: int | None = None,
    n_clusters: int | None = None,
    weighted: bool = False,
) -> np.ndarray:
    """Split geometries into clusters, then select points nearest the cluster centroids"""
    cluster_count = n_samples if n_clusters is None else n_clusters
    model = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10)
    labels = model.fit_predict(descriptor_matrix)
    cluster_sizes = np.bincount(labels, minlength=cluster_count)
    allocation = (
        _allocate_weighted(cluster_sizes, n_samples)
        if weighted
        else _allocate_evenly(cluster_sizes, n_samples)
    )

    selected = []
    for label, amount in enumerate(allocation):
        if amount == 0:
            continue
        members = np.flatnonzero(labels == label)
        distances = np.linalg.norm(
            descriptor_matrix[members] - model.cluster_centers_[label], axis=1
        )
        selected.extend(members[np.argsort(distances, kind="stable")[:amount]])
    return np.asarray(selected, dtype=int)


def stratified_sampling(
    descriptor_matrix: np.ndarray,
    n_samples: int,
    fractions: Sequence[float],
    allocations: Sequence[int],
    seed: int | None = None,
) -> np.ndarray:
    """Rank all geometries, then stratify and sample from strata"""
    fractions_array = np.asarray(fractions, dtype=float)
    allocations_array = np.asarray(allocations, dtype=int)
    raw_sizes = fractions_array * descriptor_matrix.shape[0]
    stratum_sizes = np.floor(raw_sizes).astype(int)
    remainder = descriptor_matrix.shape[0] - int(stratum_sizes.sum())
    order = np.argsort(-(raw_sizes - stratum_sizes), kind="stable")
    stratum_sizes[order[:remainder]] += 1

    ranked = np.argsort(-np.max(descriptor_matrix, axis=1), kind="stable")
    boundaries = np.cumsum(stratum_sizes)[:-1]
    strata = np.split(ranked, boundaries)
    rng = np.random.default_rng(seed)
    selected = np.concatenate(
        [
            rng.choice(stratum, size=amount, replace=False)
            for stratum, amount in zip(strata, allocations_array, strict=True)
        ]
    )
    return rng.permutation(selected)


SelectorFunction = Callable[..., np.ndarray]
SELECTOR_REGISTRY: dict[str, SelectorFunction] = {
    "random": random_sampling,
    "fps": farthest_point_sampling,
    "kmeans": kmeans_sampling,
    "stratified": stratified_sampling,
}


def select_atoms(
    described_atoms: Sequence[tuple[Atoms, np.ndarray]],
    selector: str,
    n_samples: int,
    **kwargs: Any,
) -> list[Atoms]:
    """Select and return the geometries as ASE objects"""
    atoms = [atoms for atoms, _ in described_atoms]
    descriptor_matrix = np.vstack([vector for _, vector in described_atoms])
    indices = SELECTOR_REGISTRY[selector](
        descriptor_matrix, n_samples=n_samples, **kwargs
    )
    selected = [atoms[int(index)] for index in indices]

    if len(selected) != n_samples or len({id(item) for item in selected}) != n_samples:
        raise RuntimeError(
            f"Selector '{selector}' did not return {n_samples} unique geometries."
        )
    return selected
