#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation.py — étude d'ablation des termes de MO-DYN-HD-CACD (hd, cacd, mo, dyn)
==============================================================================

Pour attribuer le gain de la contribution à chaque composante du score
    score = base·mod,  base = β·cacd + α·hd,  mod = 1 + γ·mo + δ·dyn
on désactive un terme à la fois (poids à 0) et on mesure l'effet sur le HV.

Mécanique (sans recompilation Java) : on impose le même vecteur de poids
(α,β,γ,δ) aux trois presets de perturbation via `weights_dict`, et on n'évalue
que les snapshots **perturbés** (où ces presets s'appliquent). Le HV est rapporté
en ratio à wdeg (apparié, même snapshot) pour être comparable entre instances.

Variantes :
  full   = (¼,¼,¼,¼)         toutes les composantes
  -hd    = (0, ⅓, ⅓, ⅓)
  -cacd  = (⅓, 0, ⅓, ⅓)
  -mo    = (⅓, ⅓, 0, ⅓)
  -dyn   = (⅓, ⅓, ⅓, 0)
(NB : le terme dyn vaut 0 dans le score actuel ; -dyn ≈ full le confirmera —
résultat honnête qui montre que dyn reste à implémenter.)

Régime d'évaluation : instances de taille moyenne + timeout serré, pour que la
qualité du front dépende réellement du guidage (sur instances où le solveur
converge, tous les poids donnent le même front → ablation plate).
"""

import os
import json
import logging
from typing import Dict, List

from main_real import PerturbationGenerator, DynamicScheduler, parse_orlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VARIANTS = {
    "full":  [0.25, 0.25, 0.25, 0.25],
    "-hd":   [0.0,  1/3,  1/3,  1/3],
    "-cacd": [1/3,  0.0,  1/3,  1/3],
    "-mo":   [1/3,  1/3,  0.0,  1/3],
    "-dyn":  [1/3,  1/3,  1/3,  0.0],
}
PERTURB_TYPES = ["job_arrival", "priority_change", "machine_breakdown"]


def weights_dict_for(vec: List[float]) -> Dict[str, List[float]]:
    """Même vecteur (α,β,γ,δ) pour les trois presets de perturbation."""
    return {p: list(vec) for p in PERTURB_TYPES}


def _perturbed_hvs(instance_path: str, heuristic: str, seeds: List[int],
                   T: int, timeout: float, weights_dict=None) -> List[float]:
    """HV sur les snapshots PERTURBÉS uniquement (où les poids s'appliquent)."""
    inst = parse_orlib(instance_path)
    hvs: List[float] = []
    for seed in seeds:
        pg = PerturbationGenerator(seed=seed, lam=0.5, mtbf=5)
        snaps = pg.generate_sequence(inst, T=T)
        sched = DynamicScheduler(timeout=timeout, seeds=[seed],
                                 heuristics=[heuristic], run_nsga2_flag=False,
                                 weights_dict=weights_dict)
        for t, (s_inst, p_applied) in enumerate(snaps):
            if not p_applied:           # ignorer les snapshots sans perturbation
                continue
            for res in sched.run_snapshot(s_inst, t, p_applied):
                if res["heuristic"] == heuristic and res["status"] in ("OPTIMUM", "FEASIBLE", "SAT"):
                    hvs.append(res.get("hv", 0.0))
        sched.cleanup()
    return hvs


def run_ablation(instances: List[str], seeds=(42,), T=3, timeout=5.0,
                 out_json="ablation_results.json") -> Dict:
    seeds = list(seeds)
    # Baseline wdeg (constant vis-à-vis des poids), apparié par instance.
    base_hvs = {p: _perturbed_hvs(p, "wdeg", seeds, T, timeout) for p in instances}

    results = {}
    for name, vec in VARIANTS.items():
        wd = weights_dict_for(vec)
        ratios = []
        for p in instances:
            cand = _perturbed_hvs(p, "mo_dyn_hd_cacd", seeds, T, timeout, weights_dict=wd)
            for hv_c, hv_b in zip(cand, base_hvs[p]):
                if hv_b > 0:
                    ratios.append(hv_c / hv_b)
        mean = sum(ratios) / max(1, len(ratios))
        results[name] = {"mean_hv_ratio_vs_wdeg": round(mean, 4), "n": len(ratios)}
        logging.info(f"{name:6s} : ratio HV moyen = {mean:.4f}  (n={len(ratios)})")

    # Contribution de chaque terme = full − variante (baisse = terme utile)
    full = results["full"]["mean_hv_ratio_vs_wdeg"]
    for name in VARIANTS:
        if name != "full":
            results[name]["delta_vs_full"] = round(full - results[name]["mean_hv_ratio_vs_wdeg"], 4)

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Résultats → {out_json}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ablation MO-DYN-HD-CACD")
    ap.add_argument("--instances", default="benchmarks_real/orlib/la06.txt,benchmarks_real/orlib/la07.txt")
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--snapshots", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=5.0)
    a = ap.parse_args()
    insts = [s.strip() for s in a.instances.split(",") if s.strip()]
    seeds = [int(s) for s in a.seeds.split(",")]
    run_ablation(insts, seeds=seeds, T=a.snapshots, timeout=a.timeout)
