#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_per_instance.py — analyse statistique au niveau INSTANCE (pas bloc).

Répond au reviewer (pseudo-réplication) : les blocs instance×snapshot×seed ne
sont pas indépendants (mesures répétées, seeds appariés). On agrège donc à UNE
valeur par (heuristique, instance) AVANT Friedman/Nemenyi/Wilcoxon/Â₁₂. L'unité
statistique est l'instance (n=83 job-shop, n=80 RCPSP).
"""
import os, json
import numpy as np, pandas as pd
from scipy import stats
import scikit_posthocs as sp

REF = "mo_dyn_hd_cacd"
JS = ("la", "ft", "abz", "orb", "swv", "yn", "jobshop")
def fam(i):
    n = str(i).lower()
    if n.startswith("j") and not n.startswith("jobshop"): return "rcpsp"
    if n.startswith(JS): return "jobshop"
    return "other"

def a12(x, y):
    x, y = np.asarray(x), np.asarray(y); m, n = len(x), len(y)
    r = stats.rankdata(np.concatenate([x, y])); return (r[:m].sum()/m-(m+1)/2)/n
def eff(a):
    d = abs(a-0.5)
    return "negligible" if d < 0.06 else "small" if d < 0.14 else "medium" if d < 0.21 else "large"

def per_instance(df, val, higher=True):
    piv = df.pivot_table(index="instance", columns="heuristic", values=val, aggfunc="mean").dropna(axis=0, how="any")
    ranks = piv.rank(axis=1, ascending=not higher).mean(axis=0).sort_values()
    chi, p = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
    nem = sp.posthoc_nemenyi_friedman(piv.values); nem.index = piv.columns; nem.columns = piv.columns
    rows = {}
    for h in piv.columns:
        if h == REF: continue
        rv, ov = piv[REF].values, piv[h].values
        try: wp = float(stats.wilcoxon(rv, ov).pvalue)
        except ValueError: wp = float("nan")
        A = a12(rv, ov) if higher else 1 - a12(rv, ov)
        rows[h] = dict(wilcoxon_p=wp, nemenyi_p=float(nem.loc[REF, h]), a12=round(float(A), 3), effet=eff(A))
    return dict(n=int(piv.shape[0]), friedman_chi2=float(chi), friedman_p=float(p),
                ranks={k: round(float(v), 3) for k, v in ranks.items()}, vs_ref=rows,
                median={k: float(np.median(piv[k].values)) for k in piv.columns})

def main():
    hv = pd.read_csv("results_campaign/hv2d_no_lex/hv2d_raw.csv")
    raw = pd.read_csv("results_campaign/raw_no_lex.csv", usecols=["instance","snapshot","seed","heuristic","cpu"])
    for d in (hv, raw): d["family"] = d["instance"].apply(fam)
    out = {}
    for f in ("jobshop", "rcpsp"):
        out[f] = dict(
            hv2d=per_instance(hv[hv.family == f], "hv2d", higher=True),
            cpu=per_instance(raw[raw.family == f], "cpu", higher=False))
    os.makedirs("results_campaign/per_instance", exist_ok=True)
    json.dump(out, open("results_campaign/per_instance/summary.json", "w"), indent=2, ensure_ascii=False)
    # Markdown
    L = ["# Statistiques au niveau instance (réponse pseudo-réplication)", "",
         "Unité = instance (moyenne sur snapshots×seeds avant tests). n = 83 job-shop, 80 RCPSP.", ""]
    for f in ("jobshop", "rcpsp"):
        for metric, lab, hi in [("hv2d", "HV 2D normalisé (↑)", True), ("cpu", "CPU s (↓)", False)]:
            r = out[f][metric]
            L += [f"## {f} — {lab} | n={r['n']} | Friedman χ²={r['friedman_chi2']:.1f}, p={r['friedman_p']:.2e}", "",
                  "| Heuristique | rang | médiane |", "|---|---:|---:|"]
            for h, rk in r["ranks"].items():
                star = " ⭐" if h == REF else ""
                L.append(f"| {h}{star} | {rk:.3f} | {r['median'].get(h, float('nan')):.4g} |")
            if r["vs_ref"]:
                L += ["", "| vs | Wilcoxon p | Nemenyi p | Â₁₂ | effet |", "|---|---:|---:|---:|---|"]
                for h, d in r["vs_ref"].items():
                    L.append(f"| {h} | {d['wilcoxon_p']:.2e} | {d['nemenyi_p']:.3f} | {d['a12']} | {d['effet']} |")
            L.append("")
    open("results_campaign/per_instance/stats.md", "w").write("\n".join(L))
    print("→ results_campaign/per_instance/{summary.json,stats.md}")
    print("\n".join(L))

if __name__ == "__main__":
    main()
