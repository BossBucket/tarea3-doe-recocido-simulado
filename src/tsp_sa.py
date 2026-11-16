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
    iterations: int = 300,
    final_temperature: float = 1e-3,
) -> float:
    """Optimize a symmetric Euclidean TSP and return its best cost.

    The temperature follows a geometric schedule from ``initial_temperature``
    to ``final_temperature`` over a fixed number of iterations.  A fixed budget
    keeps every design point comparable: changing ``initial_temperature`` does
    not change how many candidate tours are evaluated.

    A local ``random.Random`` instance is used so callers get reproducible
    results without changing the process-wide random-number generator.
    """
    if initial_temperature <= 0:
        raise ValueError("initial_temperature must be positive")
    if final_temperature <= 0:
        raise ValueError("final_temperature must be positive")
    if final_temperature > initial_temperature:
        raise ValueError("final_temperature must not exceed initial_temperature")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    distances = distance_matrix(coordinates)
    rng = random.Random(seed)
    current = list(range(len(distances)))
    rng.shuffle(current)
    current_cost = tour_cost(current, distances)
    best_cost = current_cost
    temperature = float(initial_temperature)
    if iterations > 1:
        cooling = (final_temperature / initial_temperature) ** (1.0 / (iterations - 1))
    else:
        cooling = 1.0

    for _ in range(iterations):
        first = rng.randrange(0, len(current) - 1)
        last = rng.randrange(first + 1, len(current))
        candidate = current[:first] + current[first : last + 1][::-1] + current[last + 1 :]
        candidate_cost = tour_cost(candidate, distances)
        difference = candidate_cost - current_cost
        if difference < 0 or rng.random() < math.exp(-difference / temperature):
            current, current_cost = candidate, candidate_cost
            if current_cost < best_cost:
                best_cost = current_cost
        temperature *= cooling

    return float(best_cost)
