"""Run the factorial DOE for the initial temperature of Simulated Annealing."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

from tsp_sa import simulated_annealing

MATRICULA = 2225447
T_MIN, T_MAX = 0.1, 50.0
SIZES = (5, 10, 15)
OFFSETS = {5: 0, 10: 6, 15: 3}
REPLICATES = 10
LEVELS = ("bajo", "medio", "alto")
ITERATIONS = 300
FINAL_TEMPERATURE = 1e-3

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = RESULTS / "data"
FIGURES = RESULTS / "figures"
OUTPUT = DATA / "resultados_corridas.csv"
INSTANCES = DATA / "instancias.csv"
DESIGN = DATA / "diseno.json"


def temperature_levels(r: float) -> dict[str, float]:
    """Scale the three Option A levels to the configured temperature range."""
    proportions = {"bajo": 0.3 * r, "medio": 0.3 * r + 0.3, "alto": 0.3 * r + 0.6}
    return {
        level: T_MIN + proportion * (T_MAX - T_MIN)
        for level, proportion in proportions.items()
    }


def temperature_fractions(r: float) -> dict[str, float]:
    """Return the dimensionless fractions before scaling to real temperatures."""
    return {"bajo": 0.3 * r, "medio": 0.3 * r + 0.3, "alto": 0.3 * r + 0.6}


def fixed_instance(size: int) -> np.ndarray:
    """Create the reproducible challenging instance associated with ``size``."""
    offset = OFFSETS[size]
    return np.random.default_rng(MATRICULA + size * 1000 + offset).uniform(
        0, 100, size=(size, 2)
    )


def run_seed(size: int, level_index: int, replicate: int) -> int:
    """Return a deterministic seed that is unique for every design point."""
    return MATRICULA * 1_000_000 + size * 10_000 + level_index * 1_000 + replicate + 1


def write_instances(instances: dict[int, np.ndarray]) -> None:
    """Write the coordinates of every TSP instance used in the experiment."""
    with INSTANCES.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["tamano", "punto", "x", "y"])
        for size, coordinates in instances.items():
            for point, (x, y) in enumerate(coordinates, start=1):
                writer.writerow([size, point, float(x), float(y)])


def write_design(r: float, temperatures: dict[str, float]) -> None:
    """Write the fixed experimental-design metadata for the report."""
    payload = {
        "matricula": MATRICULA,
        "r": r,
        "t_min": T_MIN,
        "t_max": T_MAX,
        "sizes": list(SIZES),
        "offsets": {str(size): offset for size, offset in OFFSETS.items()},
        "replicates": REPLICATES,
        "iterations": ITERATIONS,
        "final_temperature": FINAL_TEMPERATURE,
        "levels": list(LEVELS),
        "fractions": temperature_fractions(r),
        "temperatures": temperatures,
        "total_runs": len(SIZES) * len(LEVELS) * REPLICATES,
    }
    DESIGN.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_experiment(output: Path = OUTPUT) -> tuple[float, dict[str, float]]:
    """Run all 90 factorial trials and write their raw observations."""
    random.seed(MATRICULA)
    r = random.random()
    temperatures = temperature_levels(r)
    output.parent.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    instances = {size: fixed_instance(size) for size in SIZES}
    write_instances(instances)

    rows: list[dict[str, object]] = []
    for size in SIZES:
        coordinates = instances[size]
        for level_index, level in enumerate(LEVELS):
            temperature = temperatures[level]
            for replicate in range(REPLICATES):
                seed = run_seed(size, level_index, replicate)
                cost = simulated_annealing(
                    coordinates,
                    temperature,
                    seed,
                    iterations=ITERATIONS,
                    final_temperature=FINAL_TEMPERATURE,
                )
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

    write_design(r, temperatures)
    return r, temperatures


if __name__ == "__main__":
    value, levels = run_experiment()
    print(f"r = {value:.10f}")
    print(
        "Temperaturas evaluadas:",
        ", ".join(f"{key}={temperature:.4f}" for key, temperature in levels.items()),
    )
    print(f"Se escribieron {len(SIZES) * len(levels) * REPLICATES} corridas en {OUTPUT}")
