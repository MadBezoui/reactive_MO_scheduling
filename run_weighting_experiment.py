#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_weighting_experiment.py — banc d'essai contrôlé des schémas de pondération.

Compare, sur EXACTEMENT le même modèle/budget, trois schémas de pondération des
conflits implémentés dans `WeightingSelector.java` :
    wdeg2004  — dom/wdeg original (poids scalaire par contrainte, partage de portée)
    abscon    — raffinement par variable (Wattez et al. 2019)
    hd2004    — wdeg2004 + terme d'historique de branchement (la seule déviation
                réelle du code du manuscrit vs dom/wdeg2004)

But : exposer toute différence de COMPORTEMENT DE RECHERCHE (nœuds, échecs,
backtracks, taux de clôture/preuve) — pas la qualité de front sous budget, où
les schémas peuvent être à égalité. C'est l'expérience demandée par le rapport
de relecture (séparer 2004 / AbsCon / HD avec des métriques sensibles au
branchement).

Prérequis : jar compilé sur le Mac (`cd choco_solver && mvn package`).

Usage :
    python3 run_weighting_experiment.py                 # toutes les instances job-shop
    python3 run_weighting_experiment.py --timeout 30 --points 5 --jobs 4
"""
import os, sys, csv, json, glob, argparse, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats

JAR = "choco_solver/target/dmo-choco.jar"
VARIANTS = ["wdeg2004", "abscon", "hd2004"]
TUNING = {"la06", "la07", "la08"}   # triplet de réglage, exclu du test


def run_one(args):
    jar, instance, variant, timeout, points = args
    cmd = ["java", "-cp", jar, "dmowcsp.ExperimentWeighting",
           "--instance", instance, "--variant", variant,
           "--timeout", str(timeout), "--points", str(points)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout * (points + 4) + 60)
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "{}"
        rec = json.loads(line)
        rec.setdefault("instance", os.path.splitext(os.path.basename(instance))[0])
        rec.setdefault("variant", variant)
        return rec
    except Exception as e:
        return {"instance": os.path.splitext(os.path.basename(instance))[0],
                "variant": variant, "error": str(e)}


def vargha_a12(x, y):
    x, y = np.asarray(x), np.asarray(y)
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([x, y]))
    return (r[:m].sum() / m - (m + 1) / 2) / n


def paired_test(df, metric, a, b, lower_better=True):
    """Test apparié a vs b sur `metric`, alignés par instance."""
    piv = df.pivot_table(index="instance", columns="variant", values=metric, aggfunc="mean")
    piv = piv.dropna(subset=[a, b])
    if len(piv) < 3:
        return None
    xa, xb = piv[a].values, piv[b].values
    try:
        p = stats.wilcoxon(xa, xb).pvalue
    except ValueError:
        p = 1.0
    a12 = vargha_a12(xa, xb)               # P(a > b) ; selon le sens de la métrique
    return dict(n=len(piv), median_a=float(np.median(xa)), median_b=float(np.median(xb)),
                wilcoxon_p=float(p), a12=float(a12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jar", default=JAR)
    ap.add_argument("--instances-glob", default="benchmarks_real/orlib/*.txt")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--include-tuning", action="store_true",
                    help="inclure la06-08 (exclus par défaut)")
    ap.add_argument("--out-dir", default="results_weighting")
    a = ap.parse_args()

    if not os.path.exists(a.jar):
        sys.exit(f"[ERREUR] jar introuvable : {a.jar}. Lancer `cd choco_solver && mvn package` sur le Mac.")

    instances = sorted(glob.glob(a.instances_glob))
    if not a.include_tuning:
        instances = [i for i in instances
                     if os.path.splitext(os.path.basename(i))[0] not in TUNING]
    if not instances:
        sys.exit(f"[ERREUR] aucune instance pour le motif {a.instances_glob}")
    print(f"{len(instances)} instances × {len(VARIANTS)} variantes "
          f"= {len(instances) * len(VARIANTS)} runs (timeout {a.timeout}s, points {a.points})")

    tasks = [(a.jar, inst, v, a.timeout, a.points) for inst in instances for v in VARIANTS]
    records = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            records.append(rec)
            if k % 10 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)}")

    os.makedirs(a.out_dir, exist_ok=True)
    df = pd.DataFrame(records)
    raw = os.path.join(a.out_dir, "raw.csv")
    df.to_csv(raw, index=False)
    print(f"→ {raw}")

    ok = df[df.get("error").isna()] if "error" in df.columns else df
    if ok.empty:
        sys.exit("[ERREUR] aucun run valide — vérifier le jar / la classe ExperimentWeighting.")

    # Agrégats + tests appariés (le wdeg2004 est la référence).
    lines = ["# Étude contrôlée des schémas de pondération des conflits",
             "",
             f"Instances job-shop : **{ok['instance'].nunique()}** | variantes : {', '.join(VARIANTS)} | "
             f"budget {a.timeout}s/ε-point, {a.points} points.",
             "",
             "## Médianes par variante", "",
             "| Variante | nœuds | échecs | backtracks | CPU (s) | clôture | ε (Prop. 1) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        s = ok[ok["variant"] == v]
        if s.empty:
            continue
        eps = s["epsilon"].mean() if "epsilon" in s.columns else float("nan")
        lines.append(f"| {v} | {s['nodes'].median():.0f} | {s['fails'].median():.0f} | "
                     f"{s['backtracks'].median():.0f} | {s['cpu'].median():.2f} | "
                     f"{s['closure_rate'].mean():.3f} | {eps:.3f} |")
    lines += ["", "## Tests appariés vs `wdeg2004` (référence)", "",
              "Â₁₂ sur nœuds/échecs : <0.5 ⇒ la variante explore MOINS que wdeg2004 (meilleur). "
              "Sur la clôture : >0.5 ⇒ la variante prouve PLUS souvent.", "",
              "| Comparaison | métrique | méd(ref) | méd(autre) | Wilcoxon p | Â₁₂ |",
              "|---|---|---:|---:|---:|---:|"]
    for b in ["abscon", "hd2004"]:
        for metric, lb in [("nodes", True), ("fails", True), ("closure_rate", False)]:
            r = paired_test(ok, metric, "wdeg2004", b, lower_better=lb)
            if r:
                lines.append(f"| wdeg2004 vs {b} | {metric} | {r['median_a']:.0f} | "
                             f"{r['median_b']:.0f} | {r['wilcoxon_p']:.3e} | {r['a12']:.3f} |")
    lines += ["", "## Lecture", "",
              "Si `wdeg2004` et `abscon` sont indistinguables sur les nœuds ET la clôture, "
              "le partage de portée 2004 et le raffinement par variable AbsCon sont équivalents "
              "sur ces propagateurs d'ordonnancement à longue portée — résultat empirique net. "
              "Si `hd2004` diffère significativement de `wdeg2004`, le terme d'historique de "
              "branchement (la seule déviation réelle du manuscrit) est le véritable objet d'étude.",
              ""]
    md = os.path.join(a.out_dir, "stats_weighting.md")
    open(md, "w").write("\n".join(lines))
    print(f"→ {md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
