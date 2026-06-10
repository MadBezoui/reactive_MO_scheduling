#!/usr/bin/env bash
# Controlled weighting experiment over the full benchmark.
# JSSP via compiled ExperimentWeighting; RCPSP via single-file source launcher.
set -u
ROOT="/sessions/stoic-admiring-darwin/mnt/CSP_SCHED_V2"
cd "$ROOT" || exit 1
CP="choco_solver/target/classes:choco_solver/target/dmo-choco.jar"
RCPSP_SRC="choco_solver/src/main/java/dmowcsp/ExperimentWeightingRCPSP.java"
TIMEOUT=${TIMEOUT:-6}
POINTS=${POINTS:-3}
OUT="$ROOT/results_campaign/weighting_controlled.jsonl"
PAR=${PAR:-3}
mkdir -p "$(dirname "$OUT")"

# Build the job list: family|instancefile|variant
JOBS="$ROOT/results_campaign/_weight_jobs.txt"
: > "$JOBS"
JSSP=${JSSP_SET:-"la01 la06 la11 la16 la21 la26 la31 la36 ft06 ft10 ft20 abz5 abz7 abz9 orb01 orb05 orb09 swv01 swv06 swv11 swv16 yn1 yn2"}
for v in wdeg2004 abscon hd2004; do
  for i in $JSSP; do
    f="benchmarks_real/orlib/${i}.txt"
    [ -f "$f" ] && echo "jssp|$f|$v" >> "$JOBS"
  done
done
RCPSP=""
for g in 10 11 12 13; do for k in $(seq 1 10); do RCPSP="$RCPSP j30${g}_${k}"; done; done
for g in 10 11 12 13; do for k in $(seq 1 10); do RCPSP="$RCPSP j60${g}_${k}"; done; done
for v in wdeg2004 abscon hd2004; do
  for i in $RCPSP; do
    fam=$([[ "$i" == j30* ]] && echo j30 || echo j60)
    f="benchmarks_real/${fam}/${i}.sm"
    [ -f "$f" ] && echo "rcpsp|$f|$v" >> "$JOBS"
  done
done

run_one() {
  line="$1"; CP="$2"; TIMEOUT="$3"; POINTS="$4"; OUT="$5"; RCPSP_SRC="$6"
  fam="${line%%|*}"; rest="${line#*|}"; f="${rest%%|*}"; v="${rest##*|}"
  # skip if already done
  base=$(basename "$f"); name="${base%.*}"
  if grep -q "\"instance\":\"${name}\",\"variant\":\"${v}\"" "$OUT" 2>/dev/null; then return; fi
  budget=$(( (POINTS + 4) * TIMEOUT + 30 ))
  if [ "$fam" = "jssp" ]; then
    out=$(timeout ${budget}s java -cp "$CP" dmowcsp.ExperimentWeighting --instance "$f" --variant "$v" --timeout "$TIMEOUT" --points "$POINTS" 2>/dev/null | tail -1)
  else
    out=$(timeout ${budget}s java -cp "$CP" "$RCPSP_SRC" --instance "$f" --variant "$v" --timeout "$TIMEOUT" --points "$POINTS" 2>/dev/null | tail -1)
  fi
  [ -z "$out" ] && out="{\"instance\":\"${name}\",\"variant\":\"${v}\",\"error\":\"timeout_or_empty\"}"
  # prepend family
  echo "{\"family\":\"${fam}\",${out#\{}" >> "$OUT"
}
export -f run_one
export CP TIMEOUT POINTS OUT RCPSP_SRC

cat "$JOBS" | xargs -d '\n' -I{} -P "$PAR" bash -c 'run_one "$@"' _ {} "$CP" "$TIMEOUT" "$POINTS" "$OUT" "$RCPSP_SRC"
echo "DONE $(date)" >> "$ROOT/results_campaign/_weight_done.txt"
