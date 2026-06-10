#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tune_hyperparams.py — réglage des poids (α,β,γ,δ) de MO-DYN-HD-CACD
===================================================================

Recherche aléatoire des poids par type de perturbation, évaluée sur un ensemble
de VALIDATION disjoint du test final.

Score normalisé (corrige le biais d'échelle du HV brut) :
    pour chaque (instance, snapshot), on mesure le ratio
        HV(mo_dyn_hd_cacd) / HV(wdeg)
    sur LE MÊME snapshot perturbé, puis on en prend la moyenne.
Un ratio > 1 signifie que la contribution domine le baseline wdeg. Le ratio est
sans dimension, donc comparable entre instances de tailles différentes — alors
que sommer des HV bruts favorise mécaniquement les instances au plus gros volume.

NB : ceci ne « prouve » pas l'optimalité des poids ; cela sélectionne la meilleure
configuration trouvée par recherche aléatoire sur le set de validation. À
documenter comme tel dans le papier, et à garder strictement séparé du set de test.
"""

import os
import json
import random
import logging
from typing import Dict, List, Tuple

from main_real import PerturbationGenerator, DynamicScheduler, parse_orlib

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Ensemble de VALIDATION — doit rester disjoint du set de TEST du papier.
VAL_INSTANCES = [
    "benchmarks_real/orlib/la06.txt",
    "benchmarks_real/orlib/la07.txt",
    "benchmarks_real/orlib/la08.txt",
]

PERTURB_TYPES = ["job_arrival", "priority_change", "machine_breakdown"]

# Paramètres d'évaluation
T_SNAPSHOTS = 3
EVAL_SEEDS = [42, 7, 19]      # plusieurs graines pour réduire le bruit
SOLVER_TIMEOUT = 5.0

_BASELINE_CACHE: Dict[str, List[float]] = {}   # instance -> [hv wdeg par (seed,snapshot)]


def generate_random_weights() -> List[float]:
    """4 poids (α,β,γ,δ) positifs sommant à 1."""
    w = [random.random() for _ in range(4)]
    s = sum(w)
    return [round(x / s, 4) for x in w]


def _hvs_for(instance_path: str, heuristic: str,
             weights_dict=None) -> List[float]:
    """HV de `heuristic` sur la séquence dynamique de l'instance, pour toutes les
    graines × snapshots. Séquence régénérée à graine fixe → snapshots identiques
    pour le baseline et le candidat (comparaison appariée)."""
    inst = parse_orlib(instance_path)
    hvs: List[float] = []
    for seed in EVAL_SEEDS:
        pg = PerturbationGenerator(seed=seed, lam=0.5, mtbf=5)
        snapshots = pg.generate_sequence(inst, T=T_SNAPSHOTS)
        sched = DynamicScheduler(
            timeout=SOLVER_TIMEOUT, seeds=[seed],
            heuristics=[heuristic], run_nsga2_flag=False,
            weights_dict=weights_dict,
        )
        for t, (s_inst, p_applied) in enumerate(snapshots):
            for res in sched.run_snapshot(s_inst, t, p_applied):
                if res["heuristic"] != heuristic:
                    continue
                if res["status"] in ("OPTIMUM", "FEASIBLE", "SAT"):
                    hvs.append(res.get("hv", 0.0))
                else:
                    logging.warning(f"{os.path.basename(instance_path)} t={t} "
                                    f"seed={seed} échec ({res.get('status')})")
                    hvs.append(0.0)
        sched.cleanup()
    return hvs


def baseline_hvs(instance_path: str) -> List[float]:
    """HV de wdeg (mis en cache : indépendant des poids testés)."""
    if instance_path not in _BASELINE_CACHE:
        _BASELINE_CACHE[instance_path] = _hvs_for(instance_path, "wdeg")
    return _BASELINE_CACHE[instance_path]


def evaluate_weights(weights_dict: Dict[str, List[float]]) -> float:
    """Score = moyenne des ratios HV(mo_dyn_hd_cacd)/HV(wdeg) sur le set de validation."""
    ratios: List[float] = []
    for path in VAL_INSTANCES:
        if not os.path.exists(path):
            logging.warning(f"Instance absente : {path}")
            continue
        base = baseline_hvs(path)
        cand = _hvs_for(path, "mo_dyn_hd_cacd", weights_dict=weights_dict)
        for hv_c, hv_b in zip(cand, base):
            if hv_b > 0:
                ratios.append(hv_c / hv_b)
    return sum(ratios) / max(1, len(ratios))


if __name__ == "__main__":
    n_iters = int(os.environ.get("TUNE_ITERS", "30"))
    best_score = -1.0
    best_weights = None

    logging.info(f"Recherche aléatoire ({n_iters} itérations), "
                 f"validation = {VAL_INSTANCES}")
    logging.info("Calcul des HV baseline (wdeg)...")
    for path in VAL_INSTANCES:
        if os.path.exists(path):
            baseline_hvs(path)

    for i in range(n_iters):
        wd = {p: generate_random_weights() for p in PERTURB_TYPES}
        score = evaluate_weights(wd)
        logging.info(f"It {i+1}/{n_iters} — ratio HV moyen = {score:.4f}")
        if score > best_score:
            best_score = score
            best_weights = wd
            logging.info(f"  ↑ nouveau meilleur ({score:.4f})")

    logging.info("=== TUNING TERMINÉ ===")
    logging.info(f"Meilleur ratio HV (vs wdeg) = {best_score:.4f} "
                 f"({'bat wdeg' if best_score > 1 else 'ne bat pas wdeg'})")
    logging.info(f"Meilleurs poids = {json.dumps(best_weights, indent=2)}")

    with open("tuning_results.json", "w") as f:
        json.dump({"best_hv_ratio_vs_wdeg": best_score,
                   "best_weights": best_weights,
                   "validation_instances": VAL_INSTANCES,
                   "eval_seeds": EVAL_SEEDS}, f, indent=2)
    logging.info("Résultats → tuning_results.json")
