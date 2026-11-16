"""Simulated Annealing with a 2-opt neighbourhood for symmetric TSPs."""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np


def distance_matrix(coordinates: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the Euclidean distance matrix for a set of 2D coordinates."""
    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("coordinates must contain at least three 2D points")
    differences = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=2))


def tour_cost(tour: Sequence[int], distances: np.ndarray) -> float:
    """Calculate the closed-tour cost."""
    route = np.asarray(tour, dtype=int)
    return float(np.sum(distances[route, np.roll(route, -1)]))


def simulated_annealing(
    coordinates: Sequence[Sequence[float]],
    initial_temperature: float,
    seed: int,
    *,
    cooling_rate: float = 0.995,
    minimum_temperature: float = 1e-3,
    iterations_per_temperature: int = 100,
) -> float:
    """Optimize a symmetric Euclidean TSP and return its best cost.

    A local ``random.Random`` instance is used so callers get reproducible
    results without changing the process-wide random-number generator.
    """
    if initial_temperature <= 0:
        raise ValueError("initial_temperature must be positive")
    if not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate must be between zero and one")
    if iterations_per_temperature < 1:
        raise ValueError("iterations_per_temperature must be positive")

    distances = distance_matrix(coordinates)
    rng = random.Random(seed)
    current = list(range(len(distances)))
    rng.shuffle(current)
    current_cost = tour_cost(current, distances)
    best_cost = current_cost
    temperature = float(initial_temperature)

    while temperature > minimum_temperature:
        for _ in range(iterations_per_temperature):
            first = rng.randrange(0, len(current) - 1)
            last = rng.randrange(first + 1, len(current))
            candidate = current[:first] + current[first : last + 1][::-1] + current[last + 1 :]
            candidate_cost = tour_cost(candidate, distances)
            difference = candidate_cost - current_cost
            if difference < 0 or rng.random() < math.exp(-difference / temperature):
                current, current_cost = candidate, candidate_cost
                best_cost = min(best_cost, current_cost)
        temperature *= cooling_rate

    return float(best_cost)
