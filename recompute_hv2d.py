#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recompute_hv2d.py — recalcule un hypervolume 2D *normalisé* et comparable.

Motivation
----------
Le HV stocké dans raw_results.csv n'est pas une base de comparaison valide :
  1. il n'est jamais normalisé (objectifs bruts -> moyenne écrasée par les
     grosses instances) ;
  2. le 3e objectif (robustesse) est codé en dur à 0.0 pour les solveurs CP
     (choco_runner.py L137) alors que NSGA calcule une vraie robustesse :
     l'axe robustesse est donc incomparable entre familles de solveurs.

Ce script recalcule le HV sur les deux objectifs *fiables et comparables*
(makespan x flowtime), normalisés dans [0,1]^2 par (instance x snapshot) avec
un point idéal/nadir PARTAGÉ entre toutes les heuristiques, puis relance
l'analyse statistique (rangs, Friedman, Nemenyi).

Usage :
    python3 recompute_hv2d.py --results results_campaign/raw_results.csv \
        --out-dir results_campaign/hv2d --reference mo_dyn_hd_cacd
"""
import os
import ast
import argparse
import numpy as np
import pandas as pd
from scipy import stats

try:
    import scikit_posthocs as sp
    POSTHOCS_OK = True
except ImportError:
    POSTHOCS_OK = False

REF_MARGIN = 1.1  # point de référence = 1.1 dans l'espace normalisé


# ---------------------------------------------------------------------------
# Parsing des fronts
# ---------------------------------------------------------------------------
def parse_front_2d(s):
    """Retourne un array (n,2) [makespan, flowtime] ou None."""
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        fr = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None
    if not fr:
        return None
    arr = np.array(fr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr[:, :2]


# ---------------------------------------------------------------------------
# HV 2D (minimisation) exact, O(n log n)
# ---------------------------------------------------------------------------
def hv2d(front, ref=(REF_MARGIN, REF_MARGIN)):
    """HV 2D pour minimisation, points et ref dans l'espace normalisé."""
    pts = [(x, y) for x, y in front if x < ref[0] and y < ref[1]]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: (p[0], p[1]))
    nd, best_y = [], float("inf")
    for x, y in pts:
        if y < best_y:
            nd.append((x, y))
            best_y = y
    hv, prev_x = 0.0, ref[0]
    for x, y in reversed(nd):
        hv += (prev_x - x) * (ref[1] - y)
        prev_x = x
    return float(hv)


# ---------------------------------------------------------------------------
# Recalcul
# ---------------------------------------------------------------------------
def recompute(df):
    """Ajoute une colonne hv2d (normalisée par instance x snapshot)."""
    df = df.copy()
    df["_front2d"] = df["pareto_front"].apply(parse_front_2d)

    records = []
    grp_keys = ["instance", "snapshot"]
    for (inst, snap), g in df.groupby(grp_keys):
        # idéal / nadir partagés sur toutes les heuristiques & seeds du groupe
        allpts = [p for fr in g["_front2d"] if fr is not None for p in fr]
        if not allpts:
            continue
        allpts = np.array(allpts)
        ideal = allpts.min(axis=0)
        nadir = allpts.max(axis=0)
        span = nadir - ideal
        span[span == 0] = 1.0  # objectif constant -> évite /0

        for idx, row in g.iterrows():
            fr = row["_front2d"]
            if fr is None:
                hv = 0.0
            else:
                norm = (fr - ideal) / span        # [0,1]^2, 0 = idéal
                hv = hv2d(norm)
            records.append({
                "instance": inst, "snapshot": snap, "seed": row["seed"],
                "heuristic": row["heuristic"], "hv2d": hv,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Stats (repris de run_campaign.py)
# ---------------------------------------------------------------------------
def build_block_matrix(df, metric="hv2d"):
    pivot = df.pivot_table(index=["instance", "snapshot", "seed"],
                           columns="heuristic", values=metric, aggfunc="mean")
    before = len(pivot)
    pivot = pivot.dropna(axis=0, how="any")
    dropped = before - len(pivot)
    if dropped:
        print(f"  [info] {dropped} blocs incomplets écartés ({len(pivot)} retenus)")
    return pivot


def mean_ranks(pivot, higher_better=True):
    ranks = pivot.rank(axis=1, ascending=not higher_better)
    return ranks.mean(axis=0).sort_values()


def vargha_a12(x, y):
    x, y = np.asarray(x), np.asarray(y)
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return float("nan")
    g = sum(1 for a in x for b in y if a > b)
    e = sum(1 for a in x for b in y if a == b)
    return (g + 0.5 * e) / (m * n)


def a12_label(a):
    d = abs(a - 0.5)
    return ("négligeable" if d < 0.06 else "petit" if d < 0.14
            else "moyen" if d < 0.21 else "grand")


def pairwise_vs_ref(pivot, ref, higher_better=True):
    rows = []
    if ref not in pivot.columns:
        return pd.DataFrame()
    r = pivot[ref].values
    for h in pivot.columns:
        if h == ref:
            continue
        o = pivot[h].values
        try:
            _, wp = stats.wilcoxon(r, o)
        except ValueError:
            wp = float("nan")
        a = vargha_a12(r, o)
        if not higher_better:
            a = 1 - a
        rows.append({"vs": h, "median_ref": float(np.median(r)),
                     "median_other": float(np.median(o)), "wilcoxon_p": wp,
                     "A12_ref_better": round(a, 3), "effect": a12_label(a)})
    return pd.DataFrame(rows)


def write_report(out_dir, ranks, fr_stat, fr_p, nem, pair_df, ref, nb, nt):
    os.makedirs(out_dir, exist_ok=True)
    L = [
        "# Analyse statistique — HV 2D normalisé (makespan × flowtime)",
        "",
        "> HV recalculé sur les 2 objectifs comparables, normalisés dans [0,1]² "
        "par (instance × snapshot) avec idéal/nadir partagés entre heuristiques. "
        "L'axe robustesse est exclu (non instrumenté côté solveurs CP — "
        "`choco_runner.py` L137).",
        "",
        f"Blocs (instance × snapshot × seed) : **{nb}** | heuristiques : **{nt}**",
        "",
        "## Rangs moyens (1 = meilleur)",
        "",
        "| Heuristique | Rang moyen |",
        "|-------------|-----------:|",
    ]
    for h, r in ranks.items():
        L.append(f"| {h}{' ⭐' if h == ref else ''} | {r:.3f} |")
    L += ["", "## Test de Friedman", "",
          f"- χ² = **{fr_stat:.3f}**, p = **{fr_p:.3e}**",
          f"- {'Différence globale significative (p < 0.05).' if fr_p < 0.05 else 'Pas de différence significative.'}",
          ""]
    if nem is not None:
        L += ["## Post-hoc de Nemenyi (p-values par paires)", "",
              "```", nem.round(4).to_string(), "```", ""]
    if pair_df is not None and not pair_df.empty:
        L += [f"## {ref} vs autres — Wilcoxon apparié + Â₁₂", "",
              "| vs | médiane(ref) | médiane(autre) | Wilcoxon p | Â₁₂ (ref meilleur) | effet |",
              "|----|----:|----:|----:|----:|----|"]
        for _, x in pair_df.iterrows():
            sig = "**" if (x["wilcoxon_p"] == x["wilcoxon_p"] and x["wilcoxon_p"] < 0.05) else ""
            L.append(f"| {x['vs']} | {x['median_ref']:.4f} | {x['median_other']:.4f} "
                     f"| {sig}{x['wilcoxon_p']:.3e}{sig} | {x['A12_ref_better']:.3f} | {x['effect']} |")
        L += ["", "_p < 0.05 en gras. Â₁₂ > 0.5 ⇒ la référence est meilleure._", ""]
    path = os.path.join(out_dir, "stats_hv2d.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-dir", default="results_campaign/hv2d")
    ap.add_argument("--reference", default="mo_dyn_hd_cacd")
    a = ap.parse_args()

    df = pd.read_csv(a.results)
    rec = recompute(df)
    os.makedirs(a.out_dir, exist_ok=True)
    rec.to_csv(os.path.join(a.out_dir, "hv2d_raw.csv"), index=False)

    pivot = build_block_matrix(rec, "hv2d")
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        raise SystemExit("[ERREUR] Pas assez de blocs/heuristiques.")

    ranks = mean_ranks(pivot, higher_better=True)
    cols = [pivot[c].values for c in pivot.columns]
    fr_stat, fr_p = stats.friedmanchisquare(*cols)
    nem = None
    if POSTHOCS_OK:
        nem = sp.posthoc_nemenyi_friedman(pivot.values)
        nem.index = pivot.columns
        nem.columns = pivot.columns
    pair_df = pairwise_vs_ref(pivot, a.reference, higher_better=True)

    path = write_report(a.out_dir, ranks, fr_stat, fr_p, nem, pair_df,
                        a.reference, pivot.shape[0], pivot.shape[1])
    ranks.to_frame("mean_rank").to_csv(os.path.join(a.out_dir, "ranks_hv2d.csv"))
    print(f"  rangs : {dict(ranks.round(3))}")
    print(f"  Friedman : chi2={fr_stat:.3f}  p={fr_p:.3e}")
    print(f"  HV2D médian par heuristique :")
    print(rec.groupby('heuristic')['hv2d'].median().round(4).to_string())
    print(f"  rapport → {path}")


if __name__ == "__main__":
    main()
