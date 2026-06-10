#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_by_family.py — analyse statistique SÉPARÉE job-shop / RCPSP.

- HV2D : repris de results_campaign/hv2d_no_lex/hv2d_raw.csv (HV 2D normalisé,
  déjà recalculé de façon comparable, axe robustesse exclu).
- CPU  : colonne `cpu` de results_campaign/raw_no_lex.csv.

Produit, par famille :
  results_campaign/by_family/{jobshop,rcpsp}/stats_hv2d.md
  results_campaign/by_family/{jobshop,rcpsp}/stats_cpu.md
  paper_EJOR/figures/cd_nemenyi_hv_{jobshop,rcpsp}.pdf
et un récapitulatif machine-lisible by_family/summary.json pour le papier.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = "mo_dyn_hd_cacd"
HV2D = "results_campaign/hv2d_no_lex/hv2d_raw.csv"
RAW  = "results_campaign/raw_no_lex.csv"
OUTROOT = "results_campaign/by_family"
FIGDIR = "paper_EJOR/figures"

JOBSHOP_PREF = ("la", "ft", "abz", "orb", "swv", "yn", "jobshop")

def family(name):
    n = str(name).lower()
    if n.startswith("j") and not n.startswith("jobshop"):
        return "rcpsp"
    if n.startswith(JOBSHOP_PREF):
        return "jobshop"
    return "other"

def vargha_a12(x, y):
    """P(X>Y)+0.5 P(X=Y) ; x = référence. >0.5 => x meilleur (si higher_better)."""
    x, y = np.asarray(x), np.asarray(y)
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return np.nan
    ranks = stats.rankdata(np.concatenate([x, y]))
    r1 = ranks[:m].sum()
    return (r1 / m - (m + 1) / 2) / n

def effect_label(a):
    d = abs(a - 0.5)
    if d < 0.06: return "négligeable"
    if d < 0.14: return "petit"
    if d < 0.21: return "moyen"
    return "grand"

def block_matrix(df, metric):
    piv = df.pivot_table(index=["instance", "snapshot", "seed"],
                         columns="heuristic", values=metric, aggfunc="mean")
    return piv.dropna(axis=0, how="any")

def analyze(piv, higher_better, ref=REF):
    ranks = piv.rank(axis=1, ascending=not higher_better).mean(axis=0).sort_values()
    cols = [piv[c].values for c in piv.columns]
    fr_chi, fr_p = stats.friedmanchisquare(*cols)
    nem = sp.posthoc_nemenyi_friedman(piv.values)
    nem.index = piv.columns; nem.columns = piv.columns
    rows = []
    rv = piv[ref].values
    for h in piv.columns:
        if h == ref: continue
        ov = piv[h].values
        try:
            w_p = stats.wilcoxon(rv, ov, zero_method="wilcox").pvalue
        except ValueError:
            w_p = 1.0
        a = vargha_a12(rv, ov) if higher_better else 1 - vargha_a12(rv, ov)
        rows.append(dict(vs=h, med_ref=float(np.median(rv)), med_other=float(np.median(ov)),
                         wilcoxon_p=float(w_p), a12=float(a), effet=effect_label(a),
                         nemenyi_p=float(nem.loc[ref, h])))
    return ranks, (float(fr_chi), float(fr_p)), nem, rows

def write_md(path, title, note, n_blocks, n_heur, ranks, fried, rows, ref=REF):
    L = [f"# {title}", "", note, "",
         f"Blocs (instance × snapshot × seed) : **{n_blocks}** | heuristiques : **{n_heur}**", "",
         "## Rangs moyens (1 = meilleur)", "", "| Heuristique | Rang moyen |", "|---|---:|"]
    for h, r in ranks.items():
        star = " ⭐" if h == ref else ""
        L.append(f"| {h}{star} | {r:.3f} |")
    L += ["", "## Friedman", "", f"- χ² = **{fried[0]:.3f}**, p = **{fried[1]:.3e}**", "",
          f"## {ref} vs baselines — Wilcoxon apparié + Nemenyi + Â₁₂", "",
          "| vs | méd(ref) | méd(autre) | Wilcoxon p | Nemenyi p | Â₁₂ (ref meilleur) | effet |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        L.append(f"| {r['vs']} | {r['med_ref']:.4g} | {r['med_other']:.4g} | "
                 f"{r['wilcoxon_p']:.3e} | {r['nemenyi_p']:.3f} | {r['a12']:.3f} | {r['effet']} |")
    L.append("")
    L.append("_Â₁₂ > 0.5 ⇒ la référence est meilleure (sens de la métrique pris en compte)._")
    with open(path, "w") as f:
        f.write("\n".join(L))

def cd_diagram(piv, family_name, out_path):
    ranks = piv.rank(axis=1, ascending=False).mean(axis=0).sort_values()
    nem = sp.posthoc_nemenyi_friedman(piv.values)
    nem.index = piv.columns; nem.columns = piv.columns
    plt.figure(figsize=(8, 2.6))
    sp.critical_difference_diagram(
        ranks=ranks, sig_matrix=nem,
        label_fmt_left="{label} ({rank:.2f})", label_fmt_right="{label} ({rank:.2f})")
    plt.title(f"Critical Distance — HV (2D normalisé) — {family_name}", fontsize=11)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def main():
    hv = pd.read_csv(HV2D)
    raw = pd.read_csv(RAW, usecols=["instance", "snapshot", "seed", "heuristic", "cpu"])
    hv["family"] = hv["instance"].apply(family)
    raw["family"] = raw["instance"].apply(family)
    os.makedirs(FIGDIR, exist_ok=True)
    summary = {}
    for fam in ("jobshop", "rcpsp"):
        outdir = os.path.join(OUTROOT, fam); os.makedirs(outdir, exist_ok=True)
        # HV2D
        sub = hv[hv["family"] == fam]
        piv_hv = block_matrix(sub, "hv2d")
        ranks, fried, nem, rows = analyze(piv_hv, higher_better=True)
        write_md(os.path.join(outdir, "stats_hv2d.md"),
                 f"HV 2D normalisé — {fam}",
                 "> HV 2D (makespan × flowtime), normalisé par (instance × snapshot), "
                 "idéal/nadir partagés. Axe robustesse exclu (non instrumenté côté CP).",
                 piv_hv.shape[0], piv_hv.shape[1], ranks, fried, rows)
        cd_diagram(piv_hv, fam, os.path.join(FIGDIR, f"cd_nemenyi_hv_{fam}.pdf"))
        # CPU
        subc = raw[raw["family"] == fam]
        piv_cpu = block_matrix(subc, "cpu")
        ranksc, friedc, nemc, rowsc = analyze(piv_cpu, higher_better=False)
        write_md(os.path.join(outdir, "stats_cpu.md"),
                 f"CPU (s) — {fam}", "> Temps CPU par run (plus petit = meilleur).",
                 piv_cpu.shape[0], piv_cpu.shape[1], ranksc, friedc, rowsc)
        n_inst = sub["instance"].nunique()
        summary[fam] = dict(
            n_instances=int(n_inst),
            hv2d=dict(n_blocks=int(piv_hv.shape[0]), friedman_chi2=fried[0], friedman_p=fried[1],
                      ranks={k: round(v, 3) for k, v in ranks.items()},
                      vs_ref={r["vs"]: dict(p=r["wilcoxon_p"], nemenyi_p=r["nemenyi_p"],
                                            a12=round(r["a12"], 3), effet=r["effet"]) for r in rows}),
            cpu=dict(n_blocks=int(piv_cpu.shape[0]), friedman_chi2=friedc[0], friedman_p=friedc[1],
                     ranks={k: round(v, 3) for k, v in ranksc.items()},
                     median_s={k: float(np.median(piv_cpu[k].values)) for k in piv_cpu.columns},
                     vs_ref={r["vs"]: dict(p=r["wilcoxon_p"], a12=round(r["a12"], 3),
                                           effet=r["effet"]) for r in rowsc}))
        print(f"[{fam}] HV2D blocks={piv_hv.shape[0]} inst={n_inst} | CPU blocks={piv_cpu.shape[0]}")
    os.makedirs(OUTROOT, exist_ok=True)
    with open(os.path.join(OUTROOT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("OK by_family/summary.json")

if __name__ == "__main__":
    main()
