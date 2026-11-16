"""Statistical analysis and plots for the factorial TSP temperature DOE."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = RESULTS / "data"
FIGURES = RESULTS / "figures"
LEVELS = ("bajo", "medio", "alto")
SIZES = (5, 10, 15)
ALPHA = 0.05
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2024


def load_design() -> dict:
    """Load the fixed design metadata written by ``experimentos.py``."""
    return json.loads((DATA / "diseno.json").read_text(encoding="utf-8"))


def load_data() -> pd.DataFrame:
    """Load the raw trials with ordered categorical factors."""
    data = pd.read_csv(DATA / "resultados_corridas.csv")
    data["tamano"] = pd.Categorical(data["tamano"], categories=SIZES, ordered=True)
    data["temperatura_nivel"] = pd.Categorical(
        data["temperatura_nivel"], categories=LEVELS, ordered=True
    )
    return data


def bootstrap_median(values: pd.Series | np.ndarray) -> tuple[float, float, float]:
    """Return median and percentile bootstrap confidence limits."""
    observations = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(
        observations,
        size=(BOOTSTRAP_SAMPLES, len(observations)),
        replace=True,
    )
    medians = np.median(samples, axis=1)
    return (
        float(np.median(observations)),
        float(np.percentile(medians, 2.5)),
        float(np.percentile(medians, 97.5)),
    )


def mean_ci(values: pd.Series) -> tuple[float, float]:
    """Return the sample mean and the half-width of its 95% normal CI."""
    observations = np.asarray(values, dtype=float)
    n = len(observations)
    mean = float(np.mean(observations))
    if n < 2:
        return mean, 0.0
    half_width = float(1.96 * np.std(observations, ddof=1) / math.sqrt(n))
    return mean, half_width


def eta_squared(table: pd.DataFrame, effect: str) -> float:
    """Return the proportion of total sum of squares explained by ``effect``."""
    total = float(table["sum_sq"].sum())
    return float(table.loc[effect, "sum_sq"] / total) if total else float("nan")


def anova_effect(table: pd.DataFrame, effect: str) -> dict[str, float]:
    """Extract F, p-value and eta squared for one ANOVA row."""
    return {
        "F": float(table.loc[effect, "F"]),
        "p": float(table.loc[effect, "PR(>F)"]),
        "eta2": eta_squared(table, effect),
    }


def fit_main_anova(data: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    """Fit the two-way ANOVA with interaction and save its table."""
    model = smf.ols("costo ~ C(temperatura_nivel) * C(tamano)", data=data).fit()
    table = sm.stats.anova_lm(model, typ=2)
    export = table.reset_index().rename(
        columns={
            "index": "Fuente",
            "sum_sq": "Suma de cuadrados",
            "df": "gl",
            "F": "F",
            "PR(>F)": "p-value",
        }
    )
    export.to_csv(DATA / "anova.csv", index=False, encoding="utf-8")
    return model, table


def fit_robustness(data: pd.DataFrame) -> dict[str, object]:
    """Run the supplementary checks reported in the discussion."""
    without_n5 = data[data["tamano"] != 5].copy()
    without_n5["tamano"] = without_n5["tamano"].cat.remove_unused_categories()
    model = smf.ols("costo ~ C(temperatura_nivel) * C(tamano)", data=without_n5).fit()
    table = sm.stats.anova_lm(model, typ=2)
    table.reset_index().rename(columns={"index": "Fuente"}).to_csv(
        DATA / "anova_sin_n5.csv", index=False, encoding="utf-8"
    )

    one_way_rows = []
    for size in SIZES:
        group = data[data["tamano"] == size]
        model_size = smf.ols("costo ~ C(temperatura_nivel)", data=group).fit()
        table_size = sm.stats.anova_lm(model_size, typ=2)
        residual_ss = float(table_size.loc["Residual", "sum_sq"])
        if np.isclose(residual_ss, 0.0):
            one_way_rows.append(
                {
                    "Tamano": size,
                    "F": float("nan"),
                    "p-value": float("nan"),
                    "eta2": float("nan"),
                    "Nota": "sin varianza residual (todas las corridas convergen)",
                }
            )
        else:
            one_way_rows.append(
                {
                    "Tamano": size,
                    "F": float(table_size.loc["C(temperatura_nivel)", "F"]),
                    "p-value": float(table_size.loc["C(temperatura_nivel)", "PR(>F)"]),
                    "eta2": eta_squared(table_size, "C(temperatura_nivel)"),
                    "Nota": "",
                }
            )
    one_way = pd.DataFrame(one_way_rows)
    one_way.to_csv(DATA / "anova_por_tamano.csv", index=False, encoding="utf-8")

    groups = [
        group["costo"].to_numpy()
        for _, group in data.groupby(["tamano", "temperatura_nivel"], observed=True)
    ]
    levene = stats.levene(*groups, center="median")

    return {
        "sin_n5": {
            "temperatura": anova_effect(table, "C(temperatura_nivel)"),
            "tamano": anova_effect(table, "C(tamano)"),
            "interaccion": anova_effect(table, "C(temperatura_nivel):C(tamano)"),
        },
        "por_tamano": one_way.to_dict(orient="records"),
        "levene_p": float(levene.pvalue),
    }


def descriptive_table(data: pd.DataFrame) -> pd.DataFrame:
    """Summarise every design point."""
    descriptive = (
        data.groupby(["tamano", "temperatura_nivel", "temperatura"], observed=True)["costo"]
        .agg(
            n="size",
            media="mean",
            mediana="median",
            desviacion="std",
            minimo="min",
            maximo="max",
        )
        .reset_index()
        .rename(
            columns={
                "tamano": "Tamano",
                "temperatura_nivel": "Nivel",
                "temperatura": "Temperatura",
                "desviacion": "Desv. est.",
            }
        )
    )
    descriptive.to_csv(DATA / "descriptivos.csv", index=False, encoding="utf-8")
    return descriptive


def select_optimal(data: pd.DataFrame) -> tuple[dict[int, str], pd.DataFrame]:
    """Choose the best level per size (median, then mean as tie-breaker)."""
    selected: dict[int, str] = {}
    rows = []
    for size in SIZES:
        group = data[data["tamano"] == size]
        stats_level = (
            group.groupby("temperatura_nivel", observed=True)["costo"]
            .agg(["median", "mean", "std", "size"])
            .reindex(LEVELS)
        )
        ordered = stats_level.sort_values(["median", "mean"], kind="stable")
        best_level = str(ordered.index[0])
        selected[size] = best_level
        rows.append(
            {
                "Tamano": size,
                "Nivel": best_level,
                "Mediana": float(stats_level.loc[best_level, "median"]),
                "Media": float(stats_level.loc[best_level, "mean"]),
                "Desv. est.": float(stats_level.loc[best_level, "std"]),
            }
        )
    return selected, pd.DataFrame(rows)


def bootstrap_tables(
    data: pd.DataFrame, design: dict, selected: dict[int, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the full and selected bootstrap tables."""
    full_rows = []
    for size in SIZES:
        for level in LEVELS:
            values = data.loc[
                (data["tamano"] == size) & (data["temperatura_nivel"] == level), "costo"
            ]
            median, lower, upper = bootstrap_median(values)
            full_rows.append(
                {
                    "Tamano": size,
                    "Nivel": level,
                    "Temperatura": design["temperatures"][level],
                    "Mediana bootstrap": median,
                    "IC 95% inferior": lower,
                    "IC 95% superior": upper,
                    "n": len(values),
                }
            )
    full_table = pd.DataFrame(full_rows)
    full_table.to_csv(DATA / "bootstrap_por_punto.csv", index=False, encoding="utf-8")

    selected_rows = []
    for size in SIZES:
        level = selected[size]
        row = full_table[(full_table["Tamano"] == size) & (full_table["Nivel"] == level)].iloc[0]
        selected_rows.append(
            {
                "Tamano (n)": size,
                "Nivel": level,
                "Temperatura (T*)": float(row["Temperatura"]),
                "Mediana bootstrap": float(row["Mediana bootstrap"]),
                "IC 95% inferior": float(row["IC 95% inferior"]),
                "IC 95% superior": float(row["IC 95% superior"]),
                "Replicas": int(row["n"]),
            }
        )
    selected_table = pd.DataFrame(selected_rows)
    selected_table.to_csv(
        DATA / "resumen_temperaturas.csv", index=False, encoding="utf-8"
    )
    return full_table, selected_table


def level_labels(design: dict) -> list[str]:
    """Return x-axis labels that show both level name and real temperature."""
    return [
        f"{level}\n{design['temperatures'][level]:.2f}" for level in LEVELS
    ]


def effect_plots(data: pd.DataFrame, design: dict) -> None:
    """Create the main-effects and interaction plots."""
    sns.set_theme(style="whitegrid")
    labels = level_labels(design)
    data = data.copy()
    data["costo_centrado"] = data["costo"] - data.groupby(
        "tamano", observed=True
    )["costo"].transform("mean")

    temperature_stats = data.groupby("temperatura_nivel", observed=True)["costo"].apply(
        mean_ci
    )
    temperature_means = [value[0] for value in temperature_stats]
    temperature_errors = [value[1] for value in temperature_stats]

    centered_stats = data.groupby("temperatura_nivel", observed=True)[
        "costo_centrado"
    ].apply(mean_ci)
    centered_means = [value[0] for value in centered_stats]
    centered_errors = [value[1] for value in centered_stats]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].errorbar(
        range(len(LEVELS)),
        temperature_means,
        yerr=temperature_errors,
        marker="o",
        color="#1f77b4",
        capsize=5,
        linewidth=2,
    )
    axes[0].set_xticks(range(len(LEVELS)))
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Nivel de temperatura")
    axes[0].set_ylabel("Costo medio")
    axes[0].set_title("Efecto principal de temperatura\n(promedio sobre tamaños)")

    axes[1].errorbar(
        range(len(LEVELS)),
        centered_means,
        yerr=centered_errors,
        marker="o",
        color="#d62728",
        capsize=5,
        linewidth=2,
    )
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xticks(range(len(LEVELS)))
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Nivel de temperatura")
    axes[1].set_ylabel("Costo centrado por tamaño")
    axes[1].set_title("Efecto de temperatura\n(después de eliminar el bloque tamaño)")
    fig.tight_layout()
    fig.savefig(FIGURES / "efectos_principales_temperatura.png", dpi=300)
    plt.close(fig)

    size_stats = data.groupby("tamano", observed=True)["costo"].apply(mean_ci)
    size_means = [value[0] for value in size_stats]
    size_errors = [value[1] for value in size_stats]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        range(len(SIZES)),
        size_means,
        yerr=size_errors,
        marker="s",
        color="#2ca02c",
        capsize=5,
        linewidth=2,
    )
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([f"n = {size}" for size in SIZES])
    ax.set_xlabel("Tamaño del problema")
    ax.set_ylabel("Costo medio")
    ax.set_title("Efecto principal del tamaño del problema")
    fig.tight_layout()
    fig.savefig(FIGURES / "efectos_principales_tamano.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    colors = {5: "#1f77b4", 10: "#ff7f0e", 15: "#2ca02c"}
    for size in SIZES:
        group = data[data["tamano"] == size]
        stats_level = group.groupby("temperatura_nivel", observed=True)["costo"].apply(
            mean_ci
        )
        means = [stats_level.loc[level][0] for level in LEVELS]
        errors = [stats_level.loc[level][1] for level in LEVELS]
        axes[0].errorbar(
            range(len(LEVELS)),
            means,
            yerr=errors,
            marker="o",
            capsize=4,
            linewidth=2,
            color=colors[size],
            label=f"n = {size}",
        )
        centered = group.groupby("temperatura_nivel", observed=True)[
            "costo_centrado"
        ].apply(mean_ci)
        c_means = [centered.loc[level][0] for level in LEVELS]
        c_errors = [centered.loc[level][1] for level in LEVELS]
        axes[1].errorbar(
            range(len(LEVELS)),
            c_means,
            yerr=c_errors,
            marker="o",
            capsize=4,
            linewidth=2,
            color=colors[size],
            label=f"n = {size}",
        )
    for ax in axes:
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Nivel de temperatura")
    axes[0].set_ylabel("Costo medio")
    axes[0].set_title("Interacción temperatura × tamaño\n(costos observados)")
    axes[0].legend()
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Costo centrado por tamaño")
    axes[1].set_title("Interacción temperatura × tamaño\n(después de eliminar el bloque tamaño)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "interaccion_temperatura_tamano.png", dpi=300)
    plt.close(fig)


def diagnostic_plots(data: pd.DataFrame, model: object) -> None:
    """Create a boxplot by design point and residual diagnostics."""
    sns.set_theme(style="whitegrid")
    plot_data = data.rename(columns={"tamano": "Tamaño"})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=plot_data,
        x="temperatura_nivel",
        y="costo",
        hue="Tamaño",
        order=LEVELS,
        hue_order=SIZES,
        ax=ax,
    )
    sns.stripplot(
        data=plot_data,
        x="temperatura_nivel",
        y="costo",
        hue="Tamaño",
        order=LEVELS,
        hue_order=SIZES,
        dodge=True,
        alpha=0.45,
        size=3,
        ax=ax,
        legend=False,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Nivel de temperatura")
    ax.set_ylabel("Costo (escala logarítmica)")
    ax.set_title("Distribución del costo por punto de diseño")
    fig.tight_layout()
    fig.savefig(FIGURES / "distribucion_por_punto.png", dpi=300)
    plt.close(fig)

    residuals = np.asarray(model.resid, dtype=float)
    fitted = np.asarray(model.fittedvalues, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(fitted, residuals, alpha=0.7)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Valores ajustados")
    axes[0].set_ylabel("Residuales")
    axes[0].set_title("Residuales vs. valores ajustados")
    sm.qqplot(residuals, line="45", ax=axes[1])
    axes[1].set_title("Gráfico Q-Q de residuales")
    fig.tight_layout()
    fig.savefig(FIGURES / "diagnostico_residuales.png", dpi=300)
    plt.close(fig)


def analyze() -> pd.DataFrame:
    """Run the ANOVA, bootstrap, plots and save every report artifact."""
    design = load_design()
    data = load_data()
    FIGURES.mkdir(parents=True, exist_ok=True)

    model, anova = fit_main_anova(data)
    robustness = fit_robustness(data)
    descriptive_table(data)
    selected, _ = select_optimal(data)
    full_bootstrap, summary = bootstrap_tables(data, design, selected)
    effect_plots(data, design)
    diagnostic_plots(data, model)

    effects = {
        "temperatura": anova_effect(anova, "C(temperatura_nivel)"),
        "tamano": anova_effect(anova, "C(tamano)"),
        "interaccion": anova_effect(anova, "C(temperatura_nivel):C(tamano)"),
    }
    temperature_significant = (
        effects["temperatura"]["p"] < ALPHA or effects["interaccion"]["p"] < ALPHA
    )
    payload = {
        "r": design["r"],
        "alpha": ALPHA,
        "temperatures": design["temperatures"],
        "fractions": design["fractions"],
        "iterations": design["iterations"],
        "final_temperature": design["final_temperature"],
        "replicates": design["replicates"],
        "effects": effects,
        "temperature_significant": temperature_significant,
        "optimal": summary.to_dict(orient="records"),
        "robustness": robustness,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    (DATA / "analisis_resumen.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"r generado: {design['r']:.10f}")
    print("Temperaturas reales evaluadas:")
    for level, value in design["temperatures"].items():
        print(f"  {level:>6}: {value:.6f}")
    print("\nANOVA (tipo II):")
    print(anova.to_string())
    print("\nResumen por tamaño:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nBootstrap completo por punto de diseño:")
    print(full_bootstrap.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nRobustez:")
    print(json.dumps(robustness, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    analyze()
