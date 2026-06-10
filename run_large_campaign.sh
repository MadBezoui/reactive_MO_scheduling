#!/usr/bin/env zsh
# =============================================================================
# run_large_campaign.sh — campagne large DMO-WCSP (EJOR)
#
# Prérequis :
#   1. cd choco_solver && mvn package && cd ..   (jar recompilé post-nettoyage)
#   2. python3 download_orlib.py                 (la11-la40 + abz/orb téléchargés)
#   3. pip install -r requirements.txt
#
# Usage :
#   zsh run_large_campaign.sh             # campagne complète (~plusieurs heures)
#   zsh run_large_campaign.sh --dry-run   # affiche les commandes sans les exécuter
# =============================================================================

set -e

DRY_RUN=0
[[ "${1}" == "--dry-run" ]] && DRY_RUN=1

JAR=choco_solver/target/dmo-choco.jar
OUT=results_campaign
TIMEOUT=30          # secondes par run Choco
SEEDS=5             # seeds par heuristique
SNAPSHOTS=5         # snapshots dynamiques par instance
JOBS=4              # parallélisme Python

HEURISTICS="dom,wdeg,activity,mo_dyn_hd_cacd"
RCPSP_FAMILIES="j30,j60"
MAX_RCPSP=40        # instances RCPSP par famille (40×2 = 80 instances RCPSP)

# ── Vérifications ──────────────────────────────────────────────────────────

echo "=== DMO-WCSP — Campagne large ==="
echo "  Timeout    : ${TIMEOUT}s/run"
echo "  Seeds      : ${SEEDS}"
echo "  Snapshots  : ${SNAPSHOTS}"
echo "  RCPSP      : ${RCPSP_FAMILIES}, max ${MAX_RCPSP}/famille"
echo ""

if [[ ! -f "$JAR" ]]; then
    echo "[ERREUR] $JAR introuvable. Lancer : cd choco_solver && mvn package"
    exit 1
fi

mkdir -p "${OUT}"

n_orlib=$(ls benchmarks_real/orlib/*.txt 2>/dev/null | wc -l | tr -d ' ')
echo "  Instances job-shop : $n_orlib"
if (( n_orlib < 40 )); then
    echo "  [AVERTISSEMENT] Moins de 40 instances job-shop."
    echo "  Lancer : python3 download_orlib.py"
fi

# ── Campagne job-shop ───────────────────────────────────────────────────────

echo ""
echo "[1/2] Campagne job-shop (orlib) …"

CMD_ORLIB=(
    python3 main_real.py
    --instances-dir benchmarks_real
    --rcpsp-families ""
    --out-dir "${OUT}/orlib"
    --timeout $TIMEOUT
    --seeds $SEEDS
    --snapshots $SNAPSHOTS
    --heuristics $HEURISTICS
    --jobs $JOBS
)

if (( DRY_RUN )); then
    echo "  [dry-run] ${CMD_ORLIB[*]}"
else
    "${CMD_ORLIB[@]}" 2>&1 | tee "${OUT}/orlib_run.log"
fi

# ── Campagne RCPSP ─────────────────────────────────────────────────────────

echo ""
echo "[2/2] Campagne RCPSP (j30 + j60) …"

CMD_RCPSP=(
    python3 main_real.py
    --instances-dir benchmarks_real
    --skip-orlib
    --rcpsp-families $RCPSP_FAMILIES
    --max-per-family $MAX_RCPSP
    --out-dir "${OUT}/rcpsp"
    --timeout $TIMEOUT
    --seeds $SEEDS
    --snapshots $SNAPSHOTS
    --heuristics $HEURISTICS
    --jobs $JOBS
)

if (( DRY_RUN )); then
    echo "  [dry-run] ${CMD_RCPSP[*]}"
else
    "${CMD_RCPSP[@]}" 2>&1 | tee "${OUT}/rcpsp_run.log"
fi

# ── Fusion des CSV ──────────────────────────────────────────────────────────

if (( ! DRY_RUN )); then
    echo ""
    echo "[3/3] Fusion raw_results.csv …"
    python3 - <<'PYEOF'
import pandas as pd, glob, os

csvs = glob.glob("results_campaign/**/*.csv", recursive=True)
csvs = [c for c in csvs if "raw_results" in c]
if not csvs:
    print("  Aucun CSV trouvé dans results_campaign/")
    exit(0)

df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
out = "results_campaign/raw_results.csv"
df.to_csv(out, index=False)
print(f"  {len(df)} lignes → {out}")
print(f"  Heuristiques : {sorted(df['heuristic'].unique())}")
print(f"  Instances    : {df['instance'].nunique()}")
PYEOF
fi

echo ""
echo "=== Campagne terminée. Lancer ensuite ==="
echo "  python3 run_campaign.py --results results_campaign/raw_results.csv \\"
echo "      --out-dir results_campaign --metric hv"
