"""Run the factorial DOE for the initial temperature of Simulated Annealing."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

from tsp_sa import simulated_annealing

MATRICULA = 12345678  # Replace with the student's matrícula when submitting.
T_MIN, T_MAX = 10.0, 500.0
SIZES = (5, 10, 15)
REPLICATES = 10
OUTPUT = Path(__file__).resolve().parents[1] / "results" / "resultados_corridas.csv"


def temperature_levels(r: float) -> dict[str, float]:
    """Scale the three Option A levels to the configured temperature range."""
    proportions = {"bajo": 0.3 * r, "medio": 0.3 * r + 0.3, "alto": 0.3 * r + 0.6}
    return {
        level: T_MIN + proportion * (T_MAX - T_MIN)
        for level, proportion in proportions.items()
    }


def fixed_instance(size: int) -> np.ndarray:
    """Create the reproducible instance associated with ``size``."""
    return np.random.default_rng(MATRICULA + size).uniform(0, 100, size=(size, 2))


def run_experiment(output: Path = OUTPUT) -> tuple[float, dict[str, float]]:
    """Run all 90 factorial trials and write their raw observations."""
    random.seed(MATRICULA)
    r = random.random()
    temperatures = temperature_levels(r)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for size in SIZES:
        coordinates = fixed_instance(size)
        for level, temperature in temperatures.items():
            for replicate in range(REPLICATES):
                seed = MATRICULA + size * 1000 + replicate
                cost = simulated_annealing(coordinates, temperature, seed)
                rows.append(
                    {
                        "tamano": size,
                        "temperatura_nivel": level,
                        "temperatura": temperature,
                        "replica": replicate + 1,
                        "semilla": seed,
                        "costo": cost,
                    }
                )

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return r, temperatures


if __name__ == "__main__":
    value, levels = run_experiment()
    print(f"r = {value:.10f}")
    print("Temperaturas evaluadas:", ", ".join(f"{key}={value:.4f}" for key, value in levels.items()))
    print(f"Se escribieron {len(SIZES) * len(levels) * REPLICATES} corridas en {OUTPUT}")
