"""Statistical analysis and plots for the factorial TSP temperature DOE."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from experimentos import MATRICULA, OUTPUT, temperature_levels

RESULTS = OUTPUT.parent
PLOTS = RESULTS / "graficas"
BOOTSTRAP_SAMPLES = 10_000


def bootstrap_median(values: pd.Series, seed: int = 2024) -> tuple[float, float, float]:
    """Return median and percentile bootstrap confidence limits."""
    observations = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(observations, size=(BOOTSTRAP_SAMPLES, len(observations)), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.median(observations)), float(np.percentile(medians, 2.5)), float(
        np.percentile(medians, 97.5)
    )


def analyze(csv_path: Path = RESULTS / "resultados_corridas.csv") -> pd.DataFrame:
    """Print the ANOVA, create plots, and return the final summary table."""
    data = pd.read_csv(csv_path)
    data["tamano"] = data["tamano"].astype("category")
    data["temperatura_nivel"] = data["temperatura_nivel"].astype("category")
    model = smf.ols("costo ~ C(temperatura_nivel) * C(tamano)", data=data).fit()
    anova = sm.stats.anova_lm(model, typ=2)

    random_state = np.random.default_rng(2024)
    summary_rows: list[dict[str, float | int]] = []
    for size, group in data.groupby("tamano", observed=True):
        medians = group.groupby("temperatura", observed=False)["costo"].median()
        optimal_temperature = float(medians.idxmin())
        median, lower, upper = bootstrap_median(
            group.loc[group["temperatura"] == optimal_temperature, "costo"],
            seed=int(random_state.integers(0, 2**32 - 1)),
        )
        summary_rows.append(
            {
                "Tamaño (n)": int(size),
                "Temperatura Óptima (T*)": optimal_temperature,
                "Mediana Bootstrap": median,
                "IC 95% Inferior": lower,
                "IC 95% Superior": upper,
            }
        )

    PLOTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))
    sns.pointplot(data=data, x="temperatura_nivel", y="costo", hue="tamano", errorbar="ci")
    plt.title("Efectos principales: temperatura y tamaño")
    plt.tight_layout()
    plt.savefig(PLOTS / "efectos_principales.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.pointplot(data=data, x="temperatura_nivel", y="costo", hue="tamano", errorbar="ci")
    plt.title("Interacción temperatura × tamaño")
    plt.tight_layout()
    plt.savefig(PLOTS / "interaccion_temperatura_tamano.png", dpi=300)
    plt.close()

    rng = random.Random(MATRICULA)
    random_value = rng.random()
    print(f"r generado: {random_value:.10f}")
    print("Temperaturas reales evaluadas:")
    for level, value in temperature_levels(random_value).items():
        print(f"  {level:>6}: {value:.6f}")
    print("\nANOVA (tipo II):")
    print(anova.to_string())
    print("\nTabla resumen final:")
    result = pd.DataFrame(summary_rows)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return result


if __name__ == "__main__":
    analyze()
