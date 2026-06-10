#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensibilité du HV2D au point de référence r (réponse reviewer #3b)."""
import ast, json
import numpy as np, pandas as pd
from scipy import stats

JS = ("la", "ft", "abz", "orb", "swv", "yn", "jobshop")
def fam(i):
    n = str(i).lower()
    if n.startswith("j") and not n.startswith("jobshop"): return "rcpsp"
    if n.startswith(JS): return "jobshop"
    return "other"

def parse(s):
    if not isinstance(s, str) or not s.strip(): return None
    try: fr = ast.literal_eval(s)
    except Exception: return None
    if not fr: return None
    a = np.array(fr, dtype=float)
    return a[:, :2] if a.ndim == 2 and a.shape[1] >= 2 else None

def hv2d(front, r):
    pts = [(x, y) for x, y in front if x < r and y < r]
    if not pts: return 0.0
    pts.sort()
    nd, best = [], float("inf")
    for x, y in pts:
        if y < best: nd.append((x, y)); best = y
    hv, prev = 0.0, r
    for x, y in reversed(nd):
        hv += (prev - x) * (r - y); prev = x
    return hv

R = [1.05, 1.1, 1.2]
df = pd.read_csv("results_campaign/raw_no_lex.csv")
df["_f"] = df["pareto_front"].apply(parse)
rows = {r: [] for r in R}
for (inst, snap), g in df.groupby(["instance", "snapshot"]):
    pts = [p for fr in g["_f"] if fr is not None for p in fr]
    if not pts: continue
    allp = np.array(pts); ideal = allp.min(0); span = allp.max(0) - ideal
    span[span == 0] = 1.0
    for _, row in g.iterrows():
        fr = row["_f"]
        norm = None if fr is None else (fr - ideal) / span
        for r in R:
            h = 0.0 if norm is None else hv2d(norm, r)
            rows[r].append((inst, row["heuristic"], h))

out = {}
for r in R:
    d = pd.DataFrame(rows[r], columns=["instance", "heuristic", "hv"])
    d["family"] = d["instance"].apply(fam)
    out[r] = {}
    for f in ("jobshop", "rcpsp"):
        piv = d[d.family == f].pivot_table(index="instance", columns="heuristic", values="hv", aggfunc="mean").dropna(axis=0, how="any")
        ranks = piv.rank(axis=1, ascending=False).mean(0).sort_values()
        out[r][f] = {k: round(float(v), 3) for k, v in ranks.items()}
print(json.dumps(out, indent=2))
open("results_campaign/per_instance/hv_ref_sensitivity.json", "w").write(json.dumps(out, indent=2))
