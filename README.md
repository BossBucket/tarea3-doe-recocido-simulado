# Tarea 3 — Ajuste de temperatura en Recocido Simulado (TSP) mediante DOE

Ajuste del parámetro de temperatura inicial T₀ de un recocido simulado con vecindario
2-opt para el problema del agente viajero (TSP), usando un diseño factorial completo
3 × 3 (Opción A), ANOVA de dos factores, bootstrap y gráficas de efectos.

- **Repositorio:** https://github.com/BossBucket/tarea3-doe-recocido-simulado
- **Reporte PDF:** [report/reporte.pdf](report/reporte.pdf)
- **Versión editable:** [report/reporte_editable.docx](report/reporte_editable.docx)
- **Matrícula:** 2225447

## Resumen del experimento

- Factor: temperatura inicial T₀, con 3 niveles (bajo, medio, alto).
- Bloque: tamaño del TSP, n ∈ {5, 10, 15}, con una instancia por tamaño.
- Réplicas: 10 corridas por punto de diseño, con semillas distintas.
- Total: 90 corridas.
- Número aleatorio fijado por matrícula: `r = 0.3243861753`.
- Rango de temperatura: `[0.1, 50]`.
- Niveles reales:
  - bajo = 4.9561
  - medio = 19.9261
  - alto = 34.8961
- Presupuesto fijo: 300 iteraciones por corrida, enfriamiento geométrico hasta `T_f = 1e-3`.

El rango y las instancias se calibraron con un piloto para evitar la saturación observada
con rangos de temperatura demasiado altos. Para n = 10 y n = 15 se eligieron instancias con
varios óptimos locales 2-opt; n = 5 es trivial y todas las temperaturas alcanzan el mismo
óptimo.

## Resultados principales

| n | Nivel | T* | Mediana bootstrap | IC 95 % |
|---|-------|----|-------------------|---------|
| 5 | bajo | 4.9561 | 124.0639 | [124.0639, 124.0639] |
| 10 | bajo | 4.9561 | 264.1639 | [264.1639, 264.1639] |
| 15 | medio | 19.9261 | 339.3459 | [339.3459, 349.4482] |

ANOVA de dos factores (α = 0.05):

| Fuente | F | p-value | η² |
|--------|---|---------|----|
| Temperatura | 3.16 | 0.0478 | 0.0016 |
| Tamaño | 1897.38 | < 0.001 | 0.9747 |
| Temperatura × Tamaño | 2.78 | 0.0322 | 0.0029 |

La conclusión debe leerse con cautela: n = 5 satura (varianza cero), las pruebas por tamaño
son no concluyentes o borderline y, al excluir n = 5, la interacción deja de ser significativa
al 5 %. El reporte discute estas limitaciones y los análisis de robustez.

## Estructura

```text
.
├── src/
│   ├── tsp_sa.py          # Recocido simulado + 2-opt
│   ├── experimentos.py    # Diseño factorial y 90 corridas
│   └── analisis.py        # ANOVA, bootstrap, gráficas y tablas
├── results/
│   ├── data/              # CSV de corridas, ANOVA, descriptivos y bootstrap
│   └── figures/           # Gráficas de efectos, interacción y diagnósticos
├── report/
│   ├── reporte.pdf        # PDF exportado desde la versión editable
│   ├── reporte_editable.docx
│   └── reporte_editable.md
├── requirements.txt
└── README.md
```

## Reproducir

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python src\experimentos.py
.venv\Scripts\python src\analisis.py
```

Los comandos generan/actualizan:

- `results/data/resultados_corridas.csv`: 90 observaciones crudas.
- `results/data/instancias.csv`: coordenadas de las instancias TSP.
- `results/data/diseno.json`: metadatos del diseño (`r`, niveles, presupuesto y offsets de instancias).
- `results/data/anova.csv`, `descriptivos.csv`, `resumen_temperaturas.csv`,
  `bootstrap_por_punto.csv`, `anova_sin_n5.csv`, `anova_por_tamano.csv`.
- `results/figures/*.png`: efectos principales, interacción, distribución y residuales.

El reporte se entrega en:

- `report/reporte_editable.docx`: versión editable (fuente para la entrega).
- `report/reporte_editable.md`: la misma versión en Markdown.
- `report/reporte.pdf`: PDF exportado desde `reporte_editable.docx`.
