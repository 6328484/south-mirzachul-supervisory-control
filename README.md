# Supervisory gate control of the South Mirzachul main canal under supply shortage

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23494009.svg)](https://doi.org/10.5281/zenodo.23494009)

Archived on Zenodo: https://doi.org/10.5281/zenodo.23494009 (all versions; v1.0.0: https://doi.org/10.5281/zenodo.23494010).

Code, input data, trained controllers and simulation results for the article

> Seytov A.J., Abdujabborov Z.A., Choriyorov N.Q., Yadgarov S.A., Haydarova R.D. **Neural supervisory gate control for sharing a supply shortage in a large irrigation main canal.** Manuscript submitted for publication.

The study tests whether a neural supervisory controller can share a supply shortage more equitably among the offtakes of the 123.7-km South Mirzachul main canal (Uzbekistan), and whether a transparent rule optimized with the same algorithm and budget does as well. Both operate only gates that the canal already has: seven check gates, the tail gate and the regulators of five major branches. The controllers are also tested under roughness uncertainty, measurement noise and equipment faults. Everything runs on a one-dimensional Saint-Venant model of the canal built from its passport. The repository contains everything needed to rebuild Tables I–VI and Figs. 2–9 of the article, either from the stored results (about 1 min), by re-simulating the stored controllers (about 25 min) or by retraining from scratch (about 13 h on two CPU cores).

## Relation to an earlier archive

The hydrodynamic model, the local FF/P layer, the fixed rule, the neural supervisor and the full-control benchmark were first archived with an earlier version of this study ([south-mirzachul-neural-gate-control](https://github.com/6328484/south-mirzachul-neural-gate-control), DOI [10.5281/zenodo.23351117](https://doi.org/10.5281/zenodo.23351117)). This repository contains that code and its results unchanged, together with the analyses added in the article: the optimized staged rule (`src/staged_rule.py`), the gate-effort variants of the neural supervisor, the measurement-noise and fault tests (`src/robustness_tests.py`) and the Molden–Gates indicators (`src/indicators.py`).

## Repository layout

```
data/
  structures.csv               check structures and offtakes from the canal passport (model input, English names)
  passport_structures_ru.csv   the same passport tables as transcribed in Russian (source record)
  canal_route.json             digitised canal route (WGS84 latitude/longitude) used for Fig. 1
  README.md                    description of the data files and their processing
src/
  gated_model.py               Saint-Venant model with gated check structures (Preissmann scheme, Newton iteration)
  canal_setup.py               calibration to the passport depths at Qmax; delivery ratios of the offtake groups
  environment.py               simulation environment with hourly control (inputs measurable by SCADA)
  supervisory.py               local FF/P layer; current practice, rule and neural supervisors
  metrics.py                   level-safety and gate-effort indicators
  objective.py                 training objective J
  scenarios.py                 test scenarios (Table III) and training cases
  train_nominal.py             evolution-strategy training on the nominal canal (100 generations)
  train_continue.py            40 further generations, nominal or with randomised roughness
  staged_rule.py               optimized staged rule (24 parameters), the fair benchmark of the network
  train_extra.py               training of the staged rule and of the gate-effort variants
  indicators.py                Molden-Gates adequacy, dependability, equity and Gini coefficient
  evaluate_extra.py            evaluation of the staged rule and the effort variants
  robustness_tests.py          measurement-noise and fault tests
  benchmark.py                 full-control benchmark (theoretical reference)
  evaluate.py                  evaluation of all strategies, robustness sweep, validation of the policies
  make_fig_canal.py            Fig. 2 and the values of Tables I–II
  fig_architecture.py          Fig. 3
  make_figures.py              Figs. 4–9 and Table IV
  paths.py                     file locations
results/
  policies/                    trained parameters (.npy): neural supervisors and full-control benchmark
  training_logs/               training curves (.csv)
  results.json                 indicators of every strategy and test scenario
  robustness.json              indicators for roughness multipliers n/n0 = 0.8–1.4
  validation.json              objective J of every policy on the 12 supply/roughness training cases
  timeseries.npz               depths and delivery ratios over time
  table_results.csv/.md        Table IV
  table_canal.json             values of Tables I and II
  evaluation_log.txt           console output of src/evaluate.py, including numerical diagnostics
  results_extra.json           staged rule and effort variants in the four test scenarios
  robustness_extra.json        staged rule in the roughness sweep
  validation_extra.json        objective J of the staged policies on the 12 training cases
  timeseries_extra.npz         time series of the staged-rule and effort runs
  indicators.json              Molden-Gates indicators of the runs in timeseries.npz
  robustness_noise_faults.json noise and fault tests
figures/                       Figs. 2–9 (.png, 300 dpi; .pdf)
run_all.sh                     the complete pipeline in order
```

## Where each result of the article comes from

| Article | Produced by | Files |
| --- | --- | --- |
| Table I (reaches), Table II (check structures) | `src/make_fig_canal.py` | `results/table_canal.json`, `data/structures.csv`, `PASSPORT` in `src/gated_model.py` |
| Table III (test scenarios) | — | `src/scenarios.py` |
| Table IV (indicators) | `src/make_figures.py` | `results/table_results.csv`, `.md` (from `results/results.json`) |
| Fig. 1 (location map) | not reproducible here: the Google Earth background imagery is not redistributed | route: `data/canal_route.json` |
| Fig. 2 (canal in the model) | `src/make_fig_canal.py` | `figures/Fig2_canal.*` |
| Fig. 3 (two-layer control) | `src/fig_architecture.py` | `figures/Fig3_architecture.*` |
| Fig. 4 (training) | `src/make_figures.py` | `figures/Fig4_training.*` (from `results/training_logs/`, `results/validation.json`) |
| Fig. 5 (delivery equity) | `src/make_figures.py` | `figures/Fig5_shares.*` (from `results/results.json`) |
| Fig. 6 (water levels) | `src/make_figures.py` | `figures/Fig6_levels.*` (from `results/timeseries.npz`) |
| Fig. 7 (gate effort) | `src/make_figures.py` | `figures/Fig7_gates.*` (from `results/results.json`) |
| Fig. 8 (gate-effort trade-off) | `src/make_figures.py` | `figures/Fig8_effort.*` (from `results/results.json`, `results/results_extra.json`) |
| Fig. 9 (roughness sensitivity) | `src/make_figures.py` | `figures/Fig9_robustness.*` (from `results/robustness.json`, `results/robustness_extra.json`) |
| Table V (Molden–Gates indicators) | `src/evaluate_extra.py` | `results/results_extra.json`, `results/indicators.json` |
| Table VI (noise and faults) | `src/robustness_tests.py` | `results/robustness_noise_faults.json` |
| Table B.1 (Supplementary Material) and the numbers of Appendices A–B of the Supplementary Material (pool travel times, steady error under a gate-coefficient error, set-point step response, delay-stability bound, Courant and Froude numbers) | `src/appendix_b.py` | `results/appendix_b.json` |

## Reproducing the results

Tested with Python 3.13 and the package versions in `requirements.txt`. The figures use the Liberation Sans font (metric-compatible with Arial); without it, matplotlib falls back to another font and the figures differ slightly in appearance.

```bash
git clone https://github.com/6328484/south-mirzachul-supervisory-control.git
cd south-mirzachul-supervisory-control
python -m pip install -r requirements.txt

# 1) Tables and figures from the stored results (about 1 min)
python src/make_fig_canal.py
python src/fig_architecture.py
python src/make_figures.py

# 2) Re-simulate the stored controllers in all scenarios (about 25 min, one core)
python src/evaluate.py > results/evaluation_log.txt
python src/evaluate_extra.py > results/evaluation_extra_log.txt
python src/robustness_tests.py > results/robustness_tests_log.txt

# 3) Retrain everything from scratch (about 13 h on two cores)
bash run_all.sh
```

Step 2 rewrites `results/results.json`, `robustness.json`, `validation.json` and `timeseries.npz` from the stored policies. Step 3 also overwrites the stored policies and training logs.

**Reproducibility checks.** With the package versions in `requirements.txt`:

- Steps 1 and 2 reproduce the stored result files, tables and figures exactly.
- Training uses fixed random seeds and is deterministic. The first generations of `train_nominal.py`, `train_continue.py` and `benchmark.py` reproduce the stored training logs exactly.
- Other versions of NumPy, SciPy or the BLAS library can change the last digits of the simulations and hence the course of the training.

**Solver note.** The nominal training (`train_nominal.py`) and the benchmark optimisation (`benchmark.py`) were run with an earlier version of the solver. That version did not halve a time step when the Newton iteration failed to converge, so these two scripts set `SOLVER_HALVINGS = 0` to reproduce the stored runs. The difference affects only candidate runs in which the Newton iteration fails. All evaluations reported in the article and the continued training (`train_continue.py`) use the current solver with up to four halvings (`GatedCanal.max_halvings`).

## Model and controllers in brief

- **Hydrodynamic model** (`src/gated_model.py`, Sect. IV-A). One-dimensional Saint-Venant equations with lateral outflow, discretised with the four-point implicit Preissmann scheme (θ = 0.6) on 132 nodes (123 segments of about 1 km and eight zero-length check structures) with a 30-min time step and Newton iteration. Each of the ten passport reaches has a prismatic trapezoidal section. Check structures follow the gate equation Q = μ a<sub>e</sub> W √(2g ΔY), a<sub>e</sub> = min(a, h<sub>u</sub>), μ = 0.65. Offtakes draw q = u q⁰ (h/h⁰)<sup>1/2</sup>. The model is calibrated to the passport depths at Qmax = 300 m³/s (`src/canal_setup.py`). One 72-h simulation takes about 0.9 s.
- **Local FF/P layer** (`src/supervisory.py`, Sect. IV-D). Every hour, each of the seven check gates and the tail gate computes a target discharge from the measured water balance of its pool plus a proportional level correction (Tc = 4 h) and converts it into an opening with the gate equation. A gate moves by at most 0.30 m per hour and not at all for changes below 1 cm.
- **Supervisors** (`src/supervisory.py`, `src/staged_rule.py`, Sect. IV-E). *Current practice*: set-points at h<sub>up</sub> − 0.10 m, branches fully open. *Rule*: branch regulators at u = f<sup>1.5</sup>. *Neural*: a multilayer perceptron 11 → 8 → 12 (204 parameters) that sets the seven level set-points and the five branch openings from measured quantities (f, nine depths, tail outflow). It reproduces the rule when its outputs are zero.
- **Training** (`src/train_nominal.py`, `src/train_continue.py`, Sect. IV-F). OpenAI-ES with 8 antithetic pairs per generation (σ = 0.1), centred-rank utilities and Adam (step 0.05). Each candidate is scored on three 72-h episodes with f = 0.7, 0.8 and 0.9 by the objective J in `src/objective.py`. Five seeds × 100 generations, followed by 40 generations either on the nominal canal or with randomised roughness n/n0 ∈ {0.9, 1.0, 1.1, 1.3}.
- **Full-control benchmark** (`src/benchmark.py`, Sect. IV-G). A theoretical reference, **not** a description of the existing gates of the canal. It assumes that all 58 offtakes are adjustable and that the scenario is known in advance. It adds open-loop corrections in three 24-h blocks (195 parameters) to the neural policy and is optimised separately for each scenario without a gate-movement penalty.

## Result files

`results/results.json` holds one record per strategy and test scenario. `scenario` is one of `f=0.75`, `f=0.85`, `variable` and `n x1.2`. `method` is one of:

| `method` | Strategy |
| --- | --- |
| `none` | no control |
| `practice` | current practice |
| `rule` | rule + FF/P |
| `neural_s0` … `neural_s4` | neural supervisor after 100 nominal generations (`policies/nominal_g100_seed*.npy`) |
| `neural_ftnom_s0` … `_s4` | +40 nominal generations, 140 in total (`policies/nominal_g140_seed*.npy`) |
| `neural_ftdr_s0` … `_s4` | +40 generations with randomised roughness (`policies/randomised_g140_seed*.npy`) — **the neural controller reported in the article** |
| `fcb` | full-control benchmark (`policies/fcb_<scenario>.npy`, starting from `nominal_g100_seed0.npy`) |
| `staged_nom_s0` … `_s2` | optimized staged rule, 100 nominal generations (`policies/staged_nominal_g100_seed*.npy`) — in `results_extra.json` |
| `staged_dr_s0` … `_s2` | staged rule, +40 generations with randomised roughness (`policies/staged_randomised_g140_seed*.npy`) — **the staged rule reported in the article** |
| `neural_w5_s0` … `_s2`, `neural_w15_s0` … `_s2` | neural supervisor, +40 nominal generations with the gate-movement weights ×5 or ×15 (`policies/effort_w5_g140_seed*.npy`, `effort_w15_g140_seed*.npy`) |

Main fields:

- **Delivery** (mean over 24–72 h): `shares` (nine group delivery ratios: offtakes of reaches 2–9 and the tail outflow), `cv`, `worst`, `weighted` (total delivery).
- **Level safety** (0–72 h): `violation_hours` (point-hours outside the allowed range), `max_violation_hours`, `max_over_m`, `max_under_m`, `max_drawdown_rate`, `rate_violation_hours`.
- **Gate effort**: `check_travel_m`, `check_moves`, `reg_travel_frac`, `reg_moves`, `n_gates`.
- **Numerical diagnostics**: `max_residual`, `substeps`, `unconverged`.

Runs in which the canal dried out contain `failed_at_h` instead of the indicators. `results/robustness.json` has the same fields with `n_scale` (n/n0) instead of `scenario`.

A neural policy file holds 204 values in the order W1 (8 × 11, row-major), b1 (8), W2 (12 × 8), b2 (12). Outputs 0–6 are the set-points of PK145, PK400, PK623+86, PK746+62, PK923+02, PK1087+23 and PK1177+29; outputs 7–11 are the openings of Right Bayaut, YuR-10, Central, Kurgantepa and YuR-18 (Akbulak). See `neural_policy` in `src/supervisory.py` for the mapping to feasible actions.

A staged-rule policy file holds 24 values: x (5), y (5), c (7), d (7), with the branches and check gates in the same order as above. The exponent of branch j is p = 1.5 exp(x), its lag τ = 24 / (1 + exp(4 − y)) h, and the set-point of check gate k follows from c and d as described in `src/staged_rule.py`.

## Data and limitations

All model data come from the canal passport (compiled around 2000 and reproduced in Seytov 2018); field measurements were not available. The model therefore describes the canal as designed and was not validated against observed levels, discharges or gate positions. The passport gives no sill or bed elevations at the structures, so the bed was taken as continuous and the effective gate widths were enlarged at three structures. The level limits are passport depths at Qmin and Qmax, and the 0.15 m/h drawdown limit is an external benchmark (Wu et al. 2021). See `data/README.md` and Sect. VI-D of the article for details.

## Licence

- Code (`src/`, `run_all.sh`): MIT licence, see `LICENSE`.
- Data, results and figures (`data/`, `results/`, `figures/`): Creative Commons Attribution 4.0 International (CC BY 4.0), see `LICENSE-DATA.md`.

Please cite the article (see `CITATION.cff`) when you use the code or data.

## Use of AI assistance

### Use of AI Assistance

During the preparation of this manuscript, the authors used Claude (Anthropic) solely for language editing and improving grammatical accuracy and clarity. The authors reviewed and approved the final text and take full responsibility for the content of the manuscript.

## Contact

Zafar A. Abdujabborov (corresponding author), National University of Uzbekistan named after Mirzo Ulugbek, Tashkent: abdujabborov_za@nuu.uz
