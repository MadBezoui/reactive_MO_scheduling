# Reproducibility guide — "Conflict-Weighting Variable Selection for Reactive Multi-Objective Job-Shop and RCPSP Scheduling"

This file maps every table and figure of the manuscript (`paper_JOS/main.tex`) to the script and data that produce it.

## Environment

- Java: JDK 11+ (campaign run on Eclipse Temurin OpenJDK 11, Apple M1 Pro, macOS 14, single-threaded)
- Solver: Choco 4.10.14 (bundled via Maven)
- Python: 3.11 — `pip install -r requirements.txt` (numpy, pandas, scipy, scikit-posthocs, pymoo, matplotlib, seaborn)
- Or use the provided `Dockerfile`

## Full pipeline from scratch

```bash
cd choco_solver && mvn package && cd ..   # builds target/dmo-choco.jar
python3 download_orlib.py                  # job-shop instances (OR-Library)
zsh run_large_campaign.sh                  # main campaign (~4-8 h) -> results_campaign/raw_results.csv
python3 recompute_hv2d.py                  # normalised 2D HV  -> results_campaign/hv2d_no_lex/
bash run_weighting_controlled.sh           # controlled experiment -> weighting_controlled.jsonl
python3 paper_JOS/regen_figures.py         # all paper figures
python3 verify_paper_stats.py              # every headline number of the paper
```

Benchmarks: OR-Library (job-shop: la, ft, abz, orb, swv, yn) and PSPLIB (j30, j60). They are public; `download_orlib.py` fetches the job-shop set, PSPLIB sets are in `benchmarks_real/j30`, `j60`. The tuning triple `la06-la08` and the non-standard `jobshop1_full` are excluded from all reported statistics (test set: 79 JSSP + 80 RCPSP).

## Table / figure provenance

| Manuscript item | Produced by | Data |
|---|---|---|
| Table 1 (`tab:dataset`) | dataset accounting, manual | `benchmarks_real/` |
| Table 2 (`tab:names`) | method-label table, manual | — |
| Friedman χ², Table 3 (`tab:stats`) | `verify_paper_stats.py` | `results_campaign/hv2d_no_lex/hv2d_raw.csv` |
| Table 4 (`tab:pairedci`) | `verify_paper_stats.py` (bootstrap, seed 0) | idem |
| Table 5 (`tab:ablation`) | `verify_paper_stats.py` (HD ablation block) | idem |
| Table 6 (`tab:cpu`) | `verify_paper_stats.py` (CPU block) | `results_campaign/raw_no_lex.csv` |
| Table 7 (`tab:controlled`) | `run_weighting_controlled.sh` | `results_campaign/weighting_controlled.jsonl`, `_summary.json`, `_summary_full80.json` |
| Table 8 (`tab:sens`), App. B | `hv_ref_sensitivity.py` + harness sweeps | `results_campaign/sensitivity.jsonl` |
| Table 9 (`tab:claims`) | summary, manual | — |
| Fig. 1 (framework) | `paper_JOS/tikz/tikz_framework.tex` | — |
| Figs. 2–8 (boxplots, CD diagrams, efficiency, Pareto example, stability, snapshots) | `paper_JOS/regen_figures.py` | `hv2d_raw.csv`, `raw_no_lex.csv` |
| App. A (historical ablation) | `ablation.py` (legacy, documented only) | `ablation_results.json` |

## Key implementation files

- `choco_solver/src/main/java/dmowcsp/WeightingSelector.java` — the three increment rules (scope-sharing 2004, per-variable AbsCon, +HD) with scope-freshness instrumentation
- `choco_solver/src/main/java/dmowcsp/ExperimentWeighting.java`, `ExperimentWeightingRCPSP.java` — controlled-experiment harness
- `choco_solver/src/main/java/dmowcsp/ParetoSolver.java` — ε-constraint sweep (budget = total/(κ+2) per subproblem)
- `main_real.py` — campaign orchestrator (snapshots, disruptions, NSGA-II/III baselines)
- `stability.py` — schedule-stability metric

## Seeds and pairing

Campaign seeds 0–4, identical across heuristics per instance (paired design); disruption generator seeded by (instance, snapshot, seed); bootstrap CIs use `numpy default_rng(0)`, 10^4 resamples.
