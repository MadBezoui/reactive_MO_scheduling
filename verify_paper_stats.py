#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_paper_stats.py — reproduces every headline number of the JOS manuscript
(paper_JOS/main.tex) from the raw campaign CSVs.

Covers: Friedman chi^2/p (Sec. 6.1), Holm-corrected Wilcoxon + Vargha-Delaney
A12 (Table tab:stats), HD ablation means/Delta/p (Table tab:ablation), median
paired differences with bootstrap CIs (Table tab:pairedci), and per-snapshot
CPU medians (Table tab:cpu).

Usage: python3 verify_paper_stats.py
Inputs: results_campaign/hv2d_no_lex/hv2d_raw.csv, results_campaign/raw_no_lex.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

TUNE = {"la06", "la07", "la08"}           # held-out tuning triple
REF = "mo_dyn_hd_cacd"                     # dom/wdeg2004+HD in the paper
BASELINES = ["dom", "wdeg", "activity", "nsga2", "nsga3"]
LABEL = {"wdeg": "Choco dom/wdeg", "dom": "dom", "activity": "activity",
         "nsga2": "NSGA-II", "nsga3": "NSGA-III"}


def keep_instance(name):
    """Clean test set: drop non-standard jobshop1_full and the tuning triple."""
    n = str(name).lower()
    return ("jobshop" not in n) and (n not in TUNE)


def family(name):
    n = str(name).lower()
    if n.startswith("j") and not n.startswith("jobshop"):
        return "rcpsp"
    return "jobshop"


def a12(x, y):
    """Standard (unpaired) Vargha-Delaney stochastic superiority."""
    x, y = np.asarray(x), np.asarray(y)
    m, n = len(x), len(y)
    r = stats.rankdata(np.concatenate([x, y]))
    return (r[:m].sum() / m - (m + 1) / 2) / n


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[k] = running
    return out


def main():
    hv = pd.read_csv("results_campaign/hv2d_no_lex/hv2d_raw.csv")
    hv = hv[hv["instance"].apply(keep_instance)].copy()
    hv["fam"] = hv["instance"].apply(family)
    agg = hv.groupby(["fam", "heuristic", "instance"])["hv2d"].mean().reset_index()

    rng = np.random.default_rng(0)
    for fam_name in ["jobshop", "rcpsp"]:
        sub = agg[agg.fam == fam_name].pivot(
            index="instance", columns="heuristic", values="hv2d").dropna()
        sub = sub[[c for c in sub.columns if c != "lex_cp_greedy"]]
        chi, p = stats.friedmanchisquare(*[sub[c] for c in sub.columns])
        print(f"\n== {fam_name} | n={len(sub)} | Friedman chi2={chi:.1f} p={p:.2e}")
        print("   means:", {c: round(sub[c].mean(), 3)
                            for c in ["dom", "wdeg", REF, "activity"]})

        ref = sub[REF]
        raw_p = {}
        for b in BASELINES:
            d = ref - sub[b]
            raw_p[b] = 1.0 if (d == 0).all() else stats.wilcoxon(ref, sub[b]).pvalue
        hp = holm(raw_p)
        for b in BASELINES:
            d = (ref - sub[b]).values
            boots = np.median(rng.choice(d, (10000, len(d))), axis=1)
            print(f"   vs {LABEL[b]:<15} Holm p={hp[b]:.2g}  A12={a12(ref, sub[b]):.3f}"
                  f"  medianDelta={np.median(d):+.3f}"
                  f" [{np.percentile(boots, 2.5):+.3f},{np.percentile(boots, 97.5):+.3f}]")

        d = ref - sub["wdeg"]
        if (d != 0).any():
            w = stats.wilcoxon(ref, sub["wdeg"])
            print(f"   HD ablation: Delta={d.mean():+.4f} p={w.pvalue:.3f}")
        else:
            print("   HD ablation: identical per-instance values (exact tie)")

    raw = pd.read_csv("results_campaign/raw_no_lex.csv",
                      usecols=["instance", "heuristic", "cpu"])
    raw = raw[raw["instance"].apply(keep_instance)].copy()
    raw["fam"] = raw["instance"].apply(family)
    print("\n== CPU median per snapshot (s) ==")
    print(raw.groupby(["fam", "heuristic"])["cpu"].median().round(2))


if __name__ == "__main__":
    main()
