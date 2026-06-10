#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regen_figures.py — regenerate the revised paper figures.

Addresses the reviewer's figure remarks:
  * new labels (CACD -> Scope-wdeg2004+HD, wdeg -> Choco dom/wdeg)
  * fig06 split by family + no duplicated legend
  * fig04 with standard-error bands, by family
  * stability normalised by makespan
  * CD diagrams relabelled
Run from the project root:  python3 paper_EJOR/regen_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp

FIG = "paper_JOS/figures"
os.makedirs(FIG, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
})

JOBSHOP_PREF = ("la", "ft", "abz", "orb", "swv", "yn", "jobshop")
TUNE = {"la06", "la07", "la08"}
def keep_instance(name):
    """Clean test set: drop non-standard jobshop1_full and the tuning triple."""
    n = str(name).lower()
    return ("jobshop" not in n) and (n not in TUNE)
def family(name):
    n = str(name).lower()
    if n.startswith("j") and not n.startswith("jobshop"):
        return "RCPSP"
    if n.startswith(JOBSHOP_PREF):
        return "Job-Shop"
    return "other"

# display order and labels (reviewer relabeling)
ORDER = ["activity", "dom", "wdeg", "mo_dyn_hd_cacd", "nsga2", "nsga3"]
LABEL = {"mo_dyn_hd_cacd": "dom/wdeg2004+HD", "nsga2": "NSGA-II",
         "nsga3": "NSGA-III", "wdeg": "Choco dom/wdeg", "dom": "dom",
         "activity": "activity"}
order_lab = [LABEL[h] for h in ORDER]
pal = dict(zip(order_lab, sns.color_palette("Set2", len(order_lab))))

hv = pd.read_csv("results_campaign/hv2d_no_lex/hv2d_raw.csv")
hv = hv[hv["instance"].apply(keep_instance)].copy()
hv["family"] = hv["instance"].apply(family)
hv = hv[hv["family"].isin(["Job-Shop", "RCPSP"])].copy()
hv["Heuristic"] = hv["heuristic"].map(LABEL)

raw = pd.read_csv("results_campaign/raw_no_lex.csv",
                  usecols=["instance", "snapshot", "seed", "heuristic", "cpu"])
raw = raw[raw["instance"].apply(keep_instance)].copy()
raw["family"] = raw["instance"].apply(family)
raw = raw[raw["family"].isin(["Job-Shop", "RCPSP"])].copy()

# ---- fig01 : HV2D boxplot by family ------------------------------------------
g = sns.catplot(data=hv, x="Heuristic", y="hv2d", col="family", kind="box",
                order=order_lab, hue="Heuristic", palette=pal, legend=False,
                height=4, aspect=1.1, sharey=False, showfliers=False)
g.set_titles("{col_name}")
g.set_axis_labels("", "Normalised HV (2D)")
for ax in g.axes.flat:
    ax.tick_params(axis="x", rotation=35)
    for lb in ax.get_xticklabels():
        lb.set_ha("right")
    ax.axvline(3.5, ls=":", c="0.6", lw=1)  # CP selectors | metaheuristics
g.figure.suptitle("Normalised hypervolume (makespan x flowtime), by family", y=1.04)
g.savefig(f"{FIG}/fig01_hv_boxplot.pdf", bbox_inches="tight")
plt.close("all")

# ---- fig04 : HV2D across snapshots, SE bands, by family ----------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
for ax, fam in zip(axes, ["Job-Shop", "RCPSP"]):
    sub = hv[hv["family"] == fam]
    sns.lineplot(data=sub, x="snapshot", y="hv2d", hue="Heuristic",
                 hue_order=order_lab, palette=pal, marker="o", ax=ax,
                 errorbar="se")  # standard-error bands
    ax.set_title(fam); ax.set_xlabel("Snapshot")
    ax.set_ylabel("Mean normalised HV (2D)")
    if ax.get_legend(): ax.get_legend().remove()
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Heuristic",
           bbox_to_anchor=(1.005, 0.9), loc="upper left")
fig.suptitle("Hypervolume across reactive snapshots (+/-1 SE)", y=1.02)
fig.tight_layout()
fig.savefig(f"{FIG}/fig04_hv_over_snapshots.pdf", bbox_inches="tight")
plt.close("all")

# ---- fig06 : CPU vs HV2D, split by family, ONE shared legend (no annotations) -
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharex=False, sharey=False)
sub_titles = {"Job-Shop": "(a) Job-shop", "RCPSP": "(b) RCPSP"}
cp_set = {"dom", "activity", "wdeg", "mo_dyn_hd_cacd"}
for ax, fam in zip(axes, ["Job-Shop", "RCPSP"]):
    hsub = hv[hv["family"] == fam].groupby("heuristic")["hv2d"].mean()
    csub = raw[raw["family"] == fam].groupby("heuristic")["cpu"].mean()
    agg = pd.DataFrame({"hv2d": hsub, "cpu": csub}).reindex(ORDER).reset_index()
    agg["lab"] = agg["heuristic"].map(LABEL)
    for _, r in agg.iterrows():
        if pd.isna(r["cpu"]):
            continue
        c = pal[r["lab"]]
        # distinct marker shapes for CP vs metaheuristics
        mk = "*" if r["heuristic"] == "mo_dyn_hd_cacd" else ("o" if r["heuristic"] in cp_set else "D")
        sz = 720 if r["heuristic"] == "mo_dyn_hd_cacd" else 300
        ax.scatter(r["cpu"], r["hv2d"], s=sz, color=c, marker=mk,
                   edgecolor="black", linewidth=1.0, zorder=3, label=r["lab"])
    ax.set_title(sub_titles[fam], fontsize=16)
    ax.set_xlabel("Mean CPU time per snapshot (s)  -  cost  ->", fontsize=14)
    ax.set_ylabel("Mean normalised HV (2D)  -  quality  ^", fontsize=14)
    ax.tick_params(labelsize=12)
# one deduplicated legend for the whole figure
handles, labels = axes[0].get_legend_handles_labels()
seen, H, L = set(), [], []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l); H.append(h); L.append(l)
fig.legend(H, L, title="Heuristic", bbox_to_anchor=(1.005, 0.92), loc="upper left")
fig.suptitle("Efficiency-quality trade-off, by family", y=1.02)
fig.tight_layout()
fig.savefig(f"{FIG}/fig06_cpu_vs_hv.pdf", bbox_inches="tight")
plt.close("all")

# ---- stability boxplot, normalised by makespan -------------------------------
def stab_plot():
    frames = []
    for fam, path in [("Job-Shop", "results_campaign/orlib/stability.csv"),
                      ("RCPSP", "results_campaign/rcpsp/stability.csv")]:
        d = pd.read_csv(path)
        d["family"] = fam
        frames.append(d)
    st = pd.concat(frames, ignore_index=True)
    st = st[st["instance"].apply(keep_instance)].copy()
    col = "mean_instability_normalized" if "mean_instability_normalized" in st.columns else "mean_instability"
    st = st[st["heuristic"].isin(ORDER)].copy()
    st["Heuristic"] = st["heuristic"].map(LABEL)
    g = sns.catplot(data=st, x="Heuristic", y=col, col="family", kind="box",
                    order=order_lab, hue="Heuristic", palette=pal, legend=False,
                    height=4, aspect=1.1, sharey=False, showfliers=False)
    g.set_titles("{col_name}")
    g.set_axis_labels("", "Start-time deviation / makespan")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=35)
        for lb in ax.get_xticklabels():
            lb.set_ha("right")
    g.figure.suptitle("Schedule stability between snapshots (normalised)", y=1.04)
    g.savefig(f"{FIG}/stability_boxplot.pdf", bbox_inches="tight")
    plt.close("all")
try:
    stab_plot()
except Exception as e:
    print("stability plot skipped:", e)

# ---- CD diagrams, relabelled -------------------------------------------------
def cd(fam_key, fam_disp, out):
    sub = hv[hv["family"] == fam_disp]
    piv = sub.pivot_table(index=["instance", "snapshot", "seed"],
                          columns="heuristic", values="hv2d", aggfunc="mean").dropna()
    # higher HV better -> rank on negative
    ranks = (-piv).rank(axis=1).mean(axis=0)
    ranks.index = [LABEL.get(h, h) for h in ranks.index]
    piv2 = piv.copy(); piv2.columns = [LABEL.get(h, h) for h in piv2.columns]
    nem = sp.posthoc_nemenyi_friedman(piv2.values)
    nem.index = piv2.columns; nem.columns = piv2.columns
    plt.figure(figsize=(8, 2.6))
    sp.critical_difference_diagram(
        ranks, nem,
        label_fmt_left="{label} ({rank:.2f})  ",
        label_fmt_right="  {label} ({rank:.2f})")
    plt.title(f"Critical-distance diagram (Nemenyi), {fam_disp}")
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()

try:
    cd("jobshop", "Job-Shop", f"{FIG}/cd_nemenyi_hv_jobshop.pdf")
    cd("rcpsp", "RCPSP", f"{FIG}/cd_nemenyi_hv_rcpsp.pdf")
except Exception as e:
    print("CD diagram skipped:", e)

# ---- fig10 : Pareto-front example, clean labels, deduplicated legend ---------
def pareto_example():
    import ast
    raw = pd.read_csv("results_campaign/raw_results.csv")
    # prefer la01 at snapshot 0, seed 0
    inst = "la01" if (raw["instance"] == "la01").any() else \
        raw[raw["instance"].str.startswith("la")]["instance"].iloc[0]
    sub = raw[(raw["instance"] == inst) & (raw["snapshot"] == 0) & (raw["seed"] == 0)]
    cp_marker = {"dom": "o", "activity": "P", "wdeg": "s", "mo_dyn_hd_cacd": "*"}
    nsga_marker = {"nsga2": "^", "nsga3": "v"}
    plt.figure(figsize=(8, 5.2))
    drawn = set()
    for h in ORDER:
        row = sub[sub["heuristic"] == h]
        if row.empty:
            continue
        try:
            front = ast.literal_eval(row.iloc[0]["pareto_front"])
        except Exception:
            continue
        if not front:
            continue
        f1 = [s[0] for s in front]; f2 = [s[1] for s in front]
        mk = cp_marker.get(h, nsga_marker.get(h, "o"))
        sz = 230 if h == "mo_dyn_hd_cacd" else 110
        lab = LABEL[h]
        plt.scatter(f1, f2, marker=mk, s=sz, color=pal[lab],
                    edgecolor="black", linewidth=0.6,
                    label=(lab if lab not in drawn else None))
        drawn.add(lab)
    plt.title(f"Pareto front ({inst}, snapshot t=0)")
    plt.xlabel("Makespan  (lower is better)")
    plt.ylabel("Flowtime  (lower is better)")
    # deduplicated legend
    h_, l_ = plt.gca().get_legend_handles_labels()
    seen, H, L = set(), [], []
    for hh, ll in zip(h_, l_):
        if ll and ll not in seen:
            seen.add(ll); H.append(hh); L.append(ll)
    plt.legend(H, L, title="Heuristic", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{FIG}/fig10_pareto_example.pdf", bbox_inches="tight")
    plt.close("all")
try:
    pareto_example()
except Exception as e:
    print("pareto example skipped:", e)

print("Regenerated: fig01, fig04, fig06, fig10, stability_boxplot, cd_nemenyi_*")
