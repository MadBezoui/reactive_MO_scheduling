#!/usr/bin/env bash
# =============================================================================
# run_all.sh — pipeline DMO-WCSP reproductible de zéro à figures
# =============================================================================
# Usage :
#   bash run_all.sh                # exécution complète
#   bash run_all.sh --skip-build   # ne pas rebuilder le jar (déjà compilé)
#   bash run_all.sh --skip-campaign # rejouer seulement stats + figures
# =============================================================================

set -euo pipefail

SKIP_BUILD=0
SKIP_CAMPAIGN=0
for arg in "$@"; do
  case "$arg" in
    --skip-build)    SKIP_BUILD=1 ;;
    --skip-campaign) SKIP_CAMPAIGN=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
  esac
done

cd "$(dirname "$0")"

CSV=results_campaign/raw_results.csv
JAR=choco_solver/target/dmo-choco.jar

echo "=============================================================="
echo " DMO-WCSP — pipeline reproductible end-to-end"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="

# ── [1/5] Dépendances Python ────────────────────────────────────────────────
echo ""
echo "[1/5] Dépendances Python"
if ! python3 -c "import pymoo" 2>/dev/null; then
  pip3 install --quiet -r requirements.txt
fi
echo "  pymoo / pandas / scipy / scikit-posthocs OK"

# ── [2/5] Build du jar Choco ────────────────────────────────────────────────
echo ""
echo "[2/5] Build du jar Choco"
if (( SKIP_BUILD )); then
  echo "  --skip-build : on conserve $JAR"
elif [[ -f "$JAR" && "$JAR" -nt choco_solver/src/main/java/dmowcsp/MoDynHdCacd.java ]]; then
  echo "  $JAR est plus récent que les sources Java → pas de rebuild"
else
  ( cd choco_solver && mvn -q -DskipTests package )
fi

# Sanity check minimal (échec rapide si jar cassé).
echo "  Sanity check ft06 …"
java -jar "$JAR" \
  --instance benchmarks_real/orlib/ft06.txt \
  --heuristic mo_dyn_hd_cacd --objective pareto --points 5 --timeout 10 \
  > /tmp/dmo_sanity.json
python3 -c "import json,sys;d=json.load(open('/tmp/dmo_sanity.json'));\
front=d.get('front') or [];\
sys.exit(0 if front and min(f[0] for f in front)==55 else 1)" \
  && echo "  ft06 → Cmax 55 ✓" \
  || { echo "  [ERREUR] sanity check ft06 KO. Le jar produit un Cmax incorrect."; exit 2; }

# ── [3/5] Téléchargement des instances OR-Library (idempotent) ─────────────
echo ""
echo "[3/5] Instances OR-Library"
n_orlib=$(ls benchmarks_real/orlib/*.txt 2>/dev/null | wc -l | tr -d ' ')
if (( n_orlib < 40 )); then
  python3 download_orlib.py
else
  echo "  $n_orlib instances job-shop présentes → pas de re-téléchargement"
fi

# ── [4/5] Campagne large ────────────────────────────────────────────────────
echo ""
echo "[4/5] Campagne large"
if (( SKIP_CAMPAIGN )); then
  echo "  --skip-campaign : on garde $CSV"
elif [[ -f "$CSV" ]]; then
  echo "  $CSV existe déjà. Supprimer pour rejouer."
else
  command -v zsh >/dev/null && zsh run_large_campaign.sh || bash run_large_campaign.sh
fi

# ── [5/5] Statistiques + figures ────────────────────────────────────────────
echo ""
echo "[5/5] Statistiques + figures"
if [[ ! -f "$CSV" ]]; then
  echo "  [ERREUR] $CSV introuvable — la campagne a-t-elle bien tourné ?"
  exit 3
fi

python3 run_campaign.py --results "$CSV" --out-dir results_campaign --metric hv
python3 run_campaign.py --results "$CSV" --out-dir results_campaign --metric spacing
python3 run_campaign.py --results "$CSV" --out-dir results_campaign --metric cpu

python3 generate_figures.py --results "$CSV" --out-dir paper_EJOR/figures

echo ""
echo "=============================================================="
echo " Pipeline terminé."
echo "   - CSV brut   : $CSV"
echo "   - Stats      : results_campaign/{friedman,wilcoxon,nemenyi}*.csv"
echo "   - Figures    : paper_EJOR/figures/*.pdf"
echo "=============================================================="
