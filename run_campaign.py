#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_campaign.py — analyse statistique d'une campagne DMO-WCSP (cible EJOR)
==========================================================================

Lit le CSV de résultats bruts produit par main_real.py et produit l'analyse
statistique attendue d'un papier OR/scheduling de rang Q1 :

  - rangs moyens par heuristique (sur HV, plus grand = meilleur) ;
  - test de Friedman (différence globale entre heuristiques) ;
  - post-hoc de Nemenyi (comparaisons par paires, diagramme de distance critique) ;
  - Wilcoxon signed-rank apparié de la contribution vs chaque baseline ;
  - taille d'effet Â₁₂ de Vargha-Delaney.

Chaque « bloc » = une (instance × snapshot × seed) ; chaque « traitement » = une
heuristique. Un bloc n'est retenu que si toutes les heuristiques y figurent.

Usage :
    python run_campaign.py --results results_real/raw_results.csv \
        --out-dir results_real --metric hv --reference mo_dyn_hd_cacd
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

try:
    import scikit_posthocs as sp
    POSTHOCS_OK = True
except ImportError:
    POSTHOCS_OK = False


# ---------------------------------------------------------------------------
# Chargement & mise en forme
# ---------------------------------------------------------------------------

def build_block_matrix(df: pd.DataFrame, metric: str,
                       block_keys=("instance", "snapshot", "seed"),
                       treatment="heuristic") -> pd.DataFrame:
    """Pivot blocs × heuristiques pour `metric`. Garde les blocs complets."""
    block_keys = [k for k in block_keys if k in df.columns]
    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    # Moyenne si doublons (sécurité)
    pivot = df.pivot_table(index=block_keys, columns=treatment,
                           values=metric, aggfunc="mean")
    before = len(pivot)
    pivot = pivot.dropna(axis=0, how="any")  # blocs complets uniquement
    dropped = before - len(pivot)
    if dropped:
        print(f"  [info] {dropped} blocs incomplets écartés ({len(pivot)} retenus)")
    return pivot


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

def mean_ranks(pivot: pd.DataFrame, higher_better=True) -> pd.Series:
    """Rang moyen par heuristique (1 = meilleur)."""
    ranks = pivot.rank(axis=1, ascending=not higher_better)
    return ranks.mean(axis=0).sort_values()


def friedman(pivot: pd.DataFrame):
    """Test de Friedman. Retourne (statistique, p-value)."""
    cols = [pivot[c].values for c in pivot.columns]
    stat, p = stats.friedmanchisquare(*cols)
    return stat, p


def nemenyi(pivot: pd.DataFrame):
    """Post-hoc de Nemenyi (matrice de p-values). None si lib absente."""
    if not POSTHOCS_OK:
        return None
    return sp.posthoc_nemenyi_friedman(pivot.values)


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    """Â₁₂ : P(x > y) + 0.5·P(x = y). 0.5 = pas d'effet ; >0.5 = x meilleur."""
    x, y = np.asarray(x), np.asarray(y)
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return float("nan")
    greater = sum(1 for a in x for b in y if a > b)
    equal = sum(1 for a in x for b in y if a == b)
    return (greater + 0.5 * equal) / (m * n)


def pairwise_vs_reference(pivot: pd.DataFrame, reference: str, higher_better=True):
    """Wilcoxon apparié + Â₁₂ de `reference` contre chaque autre heuristique."""
    rows = []
    if reference not in pivot.columns:
        return pd.DataFrame()
    ref = pivot[reference].values
    for h in pivot.columns:
        if h == reference:
            continue
        other = pivot[h].values
        try:
            w_stat, w_p = stats.wilcoxon(ref, other)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        a12 = vargha_delaney_a12(ref, other)
        if not higher_better:
            a12 = 1 - a12
        rows.append({
            "vs": h,
            "median_ref": float(np.median(ref)),
            "median_other": float(np.median(other)),
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
            "A12_ref_better": round(a12, 3),
            "effect": _a12_label(a12),
        })
    return pd.DataFrame(rows)


def _a12_label(a12: float) -> str:
    d = abs(a12 - 0.5)
    if d < 0.06:
        return "négligeable"
    if d < 0.14:
        return "petit"
    if d < 0.21:
        return "moyen"
    return "grand"


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

def write_report(out_dir, metric, ranks, fr_stat, fr_p, nem, pair_df,
                 reference, n_blocks, n_treat):
    os.makedirs(out_dir, exist_ok=True)
    lines = [
        f"# Analyse statistique — campagne DMO-WCSP (métrique : {metric})",
        "",
        f"Blocs (instance × snapshot × seed) : **{n_blocks}** | heuristiques : **{n_treat}**",
        "",
        "## Rangs moyens (1 = meilleur)",
        "",
        "| Heuristique | Rang moyen |",
        "|-------------|-----------:|",
    ]
    for h, r in ranks.items():
        mark = " ⭐" if h == reference else ""
        lines.append(f"| {h}{mark} | {r:.3f} |")

    lines += [
        "",
        "## Test de Friedman",
        "",
        f"- χ² = **{fr_stat:.3f}**, p = **{fr_p:.3e}**",
        f"- {'Différence globale significative (p < 0.05).' if fr_p < 0.05 else 'Pas de différence globale significative (p ≥ 0.05).'}",
        "",
    ]

    if nem is not None:
        lines += ["## Post-hoc de Nemenyi (p-values par paires)", "",
                  "```", nem.round(4).to_string(), "```", ""]
    else:
        lines += ["## Post-hoc de Nemenyi", "",
                  "_scikit-posthocs non installé — `pip install scikit-posthocs`._", ""]

    if pair_df is not None and not pair_df.empty:
        lines += [
            f"## {reference} vs baselines — Wilcoxon apparié + taille d'effet Â₁₂",
            "",
            "| vs | médiane(ref) | médiane(autre) | Wilcoxon p | Â₁₂ (ref meilleur) | effet |",
            "|----|----:|----:|----:|----:|----|",
        ]
        for _, row in pair_df.iterrows():
            sig = "**" if (row["wilcoxon_p"] == row["wilcoxon_p"] and row["wilcoxon_p"] < 0.05) else ""
            lines.append(
                f"| {row['vs']} | {row['median_ref']:.1f} | {row['median_other']:.1f} "
                f"| {sig}{row['wilcoxon_p']:.3e}{sig} | {row['A12_ref_better']:.3f} | {row['effect']} |"
            )
        lines += ["", "_p < 0.05 en gras. Â₁₂ > 0.5 ⇒ la référence est meilleure._", ""]

    path = os.path.join(out_dir, f"stats_{metric}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Entrée principale
# ---------------------------------------------------------------------------

def analyze(results_csv, out_dir, metric="hv", reference="mo_dyn_hd_cacd",
            higher_better=True, exclude=None):
    df = pd.read_csv(results_csv)
    if exclude:
        df = df[~df["heuristic"].isin(exclude)]
    pivot = build_block_matrix(df, metric)
    if pivot.shape[1] < 2 or pivot.shape[0] < 2:
        raise SystemExit("[ERREUR] Pas assez de blocs/heuristiques pour les tests.")

    ranks = mean_ranks(pivot, higher_better)
    fr_stat, fr_p = friedman(pivot)
    nem = nemenyi(pivot)
    if nem is not None:
        nem.index = pivot.columns
        nem.columns = pivot.columns
    pair_df = pairwise_vs_reference(pivot, reference, higher_better)

    path = write_report(out_dir, metric, ranks, fr_stat, fr_p, nem, pair_df,
                        reference, pivot.shape[0], pivot.shape[1])
    # CSV des rangs
    ranks.to_frame("mean_rank").to_csv(os.path.join(out_dir, f"ranks_{metric}.csv"))
    print(f"  rangs moyens : {dict(ranks.round(3))}")
    print(f"  Friedman : chi2={fr_stat:.3f}  p={fr_p:.3e}")
    print(f"  rapport → {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="Analyse statistique campagne DMO-WCSP")
    ap.add_argument("--results", required=True, help="CSV de résultats bruts")
    ap.add_argument("--out-dir", default="results_campaign")
    ap.add_argument("--metric", default="hv", help="hv | spacing | front_size | cpu")
    ap.add_argument("--reference", default="mo_dyn_hd_cacd",
                    help="Heuristique de référence (la contribution)")
    ap.add_argument("--lower-better", action="store_true",
                    help="Métrique où plus petit = meilleur (ex: spacing, cpu)")
    ap.add_argument("--exclude", default=None,
                    help="Heuristiques à exclure (CSV), ex: lex_cp_greedy")
    a = ap.parse_args()
    exclude = [x.strip() for x in a.exclude.split(",")] if a.exclude else None
    analyze(a.results, a.out_dir, a.metric, a.reference,
            higher_better=not a.lower_better, exclude=exclude)


if __name__ == "__main__":
    main()
