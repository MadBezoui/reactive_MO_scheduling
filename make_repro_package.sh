#!/usr/bin/env bash
# Builds the reproducibility archive cited in the Data Availability Statement.
# Usage: bash make_repro_package.sh   ->  dmo-wcsp-repro.zip
set -euo pipefail
cd "$(dirname "$0")"

OUT=dmo-wcsp-repro.zip
TMP="$(mktemp -d)/$OUT"
rm -f "$OUT"

zip -r "$TMP" \
  REPRODUCIBILITY.md README.md requirements.txt Dockerfile run_all.sh \
  verify_paper_stats.py recompute_hv2d.py analyze_per_instance.py \
  analyze_by_family.py hv_ref_sensitivity.py stability.py main_real.py \
  choco_runner.py milp_baseline.py ablation.py tune_hyperparams.py \
  download_orlib.py download_psplib.py run_large_campaign.sh \
  run_weighting_controlled.sh run_weighting_experiment.py run_campaign.py \
  choco_solver/pom.xml choco_solver/src \
  results_campaign/raw_results.csv results_campaign/raw_no_lex.csv \
  results_campaign/hv2d_no_lex \
  results_campaign/weighting_controlled.jsonl \
  results_campaign/weighting_controlled_summary.json \
  results_campaign/weighting_controlled_summary.md \
  results_campaign/weighting_controlled_summary_full80.json \
  results_campaign/sensitivity.jsonl \
  paper_JOS/regen_figures.py paper_JOS/tikz \
  -x "*.DS_Store" -x "*__pycache__*" -x "*target*"

cp "$TMP" "$OUT" && rm -rf "$(dirname "$TMP")"
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "Next: upload to Zenodo (metadata in .zenodo.json), note the DOI,"
echo "and insert it in paper_JOS/main.tex (Data availability) + cover letter."
