#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stability.py — métrique de stabilité du planning (nervousness) pour DMO-WCSP
============================================================================

EJOR / ordonnancement dynamique : au-delà de la qualité du front de Pareto, on
mesure le **coût de réordonnancement** entre snapshots consécutifs — à quel point
le planning « bouge » quand l'instance est perturbée. Une heuristique qui produit
des plannings stables est préférable en pratique (moins de perturbation pour les
opérateurs, les ateliers, la logistique).

Définition (espace de décision) : pour deux snapshots successifs t-1 et t,
l'**instabilité** est la déviation absolue moyenne des temps de début, sur les
opérations communes aux deux plannings (les opérations nouvelles, issues d'un
job_arrival, sont ignorées) :

    instab(t-1, t) = ( Σ_{(j,k) ∈ commun} |start_t(j,k) − start_{t-1}(j,k)| )
                     / |commun|

La stabilité d'une séquence = moyenne des instab sur les transitions. Plus c'est
petit, plus le planning est stable. On peut aussi normaliser par le makespan.
"""

from typing import Dict, List, Tuple, Optional, Any


def starts_to_map(rep_ops: List[List[int]],
                  rep_starts: List[int]) -> Dict[Tuple[int, int], int]:
    """Convertit (rep_ops, rep_starts) en dict {(j,k): start}."""
    m: Dict[Tuple[int, int], int] = {}
    for (j, k), s in zip(rep_ops, rep_starts):
        m[(int(j), int(k))] = int(s)
    return m


def instability(prev: Dict[Tuple[int, int], int],
                curr: Dict[Tuple[int, int], int],
                normalizer: Optional[float] = None) -> Optional[float]:
    """Déviation absolue moyenne des débuts sur les opérations communes.

    Retourne None s'il n'y a aucune opération commune. Si `normalizer` (ex. le
    makespan courant) est fourni et > 0, le résultat est divisé par lui.
    """
    common = set(prev) & set(curr)
    if not common:
        return None
    total = sum(abs(curr[op] - prev[op]) for op in common)
    val = total / len(common)
    if normalizer and normalizer > 0:
        val /= normalizer
    return val


def sequence_stability(snapshots: List[Dict[Tuple[int, int], int]],
                       makespans: Optional[List[float]] = None) -> Dict[str, Any]:
    """Agrège l'instabilité sur une séquence de plannings (un par snapshot).

    snapshots : liste de dicts {(j,k): start}, dans l'ordre temporel.
    makespans : makespan par snapshot (pour la variante normalisée), optionnel.

    Retourne : nb de transitions, instabilité moyenne brute et normalisée,
    et la liste des instabilités par transition.
    """
    raw: List[float] = []
    norm: List[float] = []
    for t in range(1, len(snapshots)):
        nm = makespans[t] if makespans and t < len(makespans) else None
        v = instability(snapshots[t - 1], snapshots[t])
        if v is not None:
            raw.append(v)
            if nm:
                vn = instability(snapshots[t - 1], snapshots[t], normalizer=nm)
                if vn is not None:
                    norm.append(vn)
    return {
        "n_transitions": len(raw),
        "mean_instability": (sum(raw) / len(raw)) if raw else None,
        "mean_instability_normalized": (sum(norm) / len(norm)) if norm else None,
        "per_transition": raw,
    }


def stability_from_results(results_by_snapshot: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcule la stabilité à partir de résultats de runner Choco successifs.

    results_by_snapshot : liste (ordonnée par snapshot) de dicts contenant
    'rep_ops', 'rep_starts' et éventuellement un front pour récupérer le makespan.
    Les snapshots sans solution représentante sont ignorés (rompent la chaîne).
    """
    snaps: List[Dict[Tuple[int, int], int]] = []
    makespans: List[float] = []
    for r in results_by_snapshot:
        if r.get("rep_ops") and r.get("rep_starts") is not None:
            snaps.append(starts_to_map(r["rep_ops"], r["rep_starts"]))
            # Le pipeline stocke le front sous 'pareto_front' ; le runner sous 'front'.
            front = r.get("front") or r.get("pareto_front") or []
            makespans.append(min(p[0] for p in front) if front else 0.0)
    return sequence_stability(snaps, makespans)


def aggregate_stability(all_results: List[Dict[str, Any]],
                        out_csv: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calcule la stabilité par (instance, heuristique, seed) sur les snapshots.

    all_results : tous les dicts de résultats (avec instance, snapshot, heuristic,
    seed, rep_ops, rep_starts, pareto_front). Regroupe par (instance, heuristic,
    seed), ordonne par snapshot, et calcule la stabilité de la séquence.
    Écrit optionnellement un CSV. Retourne la liste des lignes agrégées.
    """
    import csv as _csv
    from collections import defaultdict

    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in all_results:
        key = (r.get("instance"), r.get("heuristic"), r.get("seed", 0))
        groups[key].append(r)

    rows: List[Dict[str, Any]] = []
    for (inst, heur, seed), items in groups.items():
        items.sort(key=lambda x: x.get("snapshot", 0))
        st = stability_from_results(items)
        rows.append({
            "instance": inst, "heuristic": heur, "seed": seed,
            "n_transitions": st["n_transitions"],
            "mean_instability": st["mean_instability"],
            "mean_instability_normalized": st["mean_instability_normalized"],
        })

    if out_csv and rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


# =============================================================================
# Robustesse non circulaire — variance du makespan sur scénarios perturbés (D1)
# =============================================================================
#
# Le proxy historique « slack moyen » est calculé sur le planning courant : il
# pénalise un planning serré sans dire si ce planning *résiste* aux perturbations
# qu'on craint. On définit ici une métrique de robustesse non circulaire :
#
#     robustness(planning π, instance I, K) = − var(Cmax(π appliqué à I_k))
#                                              pour k = 1..K perturbations tirées
#
# Plus la variance est faible → plus le planning est robuste. Le signe négatif
# permet de traiter cette mesure comme un objectif à *maximiser*, cohérent avec
# les autres objectifs HV-positifs.
#
# Implémentation : la fonction prend en entrée une liste de Cmax observés sur K
# scénarios de perturbation appliqués au *même* planning de référence (ré-évalué
# sous chaque scénario). La logique de simulation des K scénarios et de la
# ré-évaluation reste à l'appelant (pipeline main_real.py ou ablation.py) qui
# dispose du PerturbationGenerator et du ScheduleEvaluator.


def makespan_variance_robustness(cmax_scenarios: List[float]) -> Optional[float]:
    """Robustesse = − variance(Cmax) sur K scénarios de perturbation.

    cmax_scenarios : liste des Cmax obtenus en ré-évaluant le *même* planning
    de référence sous K instances perturbées (K=10 conseillé pour MC stable).
    Retourne None si K < 2.

    Convention : valeur négative — plus elle est proche de 0, plus le planning
    est robuste. Le signe permet de l'agréger avec d'autres objectifs à maximiser.
    """
    if not cmax_scenarios or len(cmax_scenarios) < 2:
        return None
    n = len(cmax_scenarios)
    mean = sum(cmax_scenarios) / n
    var = sum((c - mean) ** 2 for c in cmax_scenarios) / (n - 1)  # sans biais
    return -var


def makespan_range_robustness(cmax_scenarios: List[float]) -> Optional[float]:
    """Alternative discriminante : − (max − min) du Cmax sur K scénarios.

    Moins sensible aux outliers que la variance pour K petit (K=5..10).
    """
    if not cmax_scenarios or len(cmax_scenarios) < 2:
        return None
    return -(max(cmax_scenarios) - min(cmax_scenarios))


def evaluate_robustness_montecarlo(
    base_instance: Any,
    rep_ops: List[List[int]],
    rep_starts: List[int],
    perturbation_gen: Any,
    K: int = 10,
    perturbation_types: Optional[List[str]] = None,
    evaluator_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Robustesse Monte-Carlo d'un planning représentant donné.

    base_instance : l'instance non perturbée (JobShop ou RCPSP).
    rep_ops, rep_starts : planning représentant à évaluer (sorti du front
        Pareto, p. ex. la solution makespan-minimale).
    perturbation_gen : un PerturbationGenerator (cf. main_real.py).
    K : nombre de scénarios Monte-Carlo (10 recommandé).
    perturbation_types : sous-ensemble parmi {job_arrival, machine_breakdown,
        priority_change}. Par défaut : tous.
    evaluator_fn : callable(instance_perturbée, rep_ops, rep_starts) -> Cmax.
        Si None, on tente d'utiliser ScheduleEvaluator du module main_real.

    Retourne : {"K", "cmax_scenarios", "var_robustness", "range_robustness",
                "mean_cmax", "std_cmax"}.
    """
    cmaxs: List[float] = []
    perturbation_types = perturbation_types or [
        "job_arrival", "machine_breakdown", "priority_change"
    ]

    for k in range(K):
        # On délègue au PerturbationGenerator. Chaque appel doit produire UN
        # snapshot perturbé indépendant (graine dérivée du seed_base interne).
        try:
            perturbed = perturbation_gen.apply_random_perturbations(
                base_instance, perturbation_types
            )
        except AttributeError:
            # Repli : si l'API exacte n'existe pas dans PerturbationGenerator,
            # on lève une erreur explicite pour ne pas masquer un défaut d'API.
            raise NotImplementedError(
                "PerturbationGenerator doit exposer apply_random_perturbations("
                "instance, perturbation_types) -> instance perturbée. À câbler "
                "dans main_real.PerturbationGenerator."
            )
        if evaluator_fn is None:
            try:
                from main_real import ScheduleEvaluator, JobShopInstance
                if isinstance(perturbed, JobShopInstance):
                    sch = ScheduleEvaluator.evaluate_jobshop(perturbed)
                else:
                    sch = ScheduleEvaluator.evaluate_rcpsp(perturbed)
                cmaxs.append(float(sch.makespan))
            except Exception as e:
                raise RuntimeError(
                    f"Impossible d'évaluer le scénario {k} : {e}. "
                    f"Fournir `evaluator_fn` explicite."
                )
        else:
            cmaxs.append(float(evaluator_fn(perturbed, rep_ops, rep_starts)))

    var_r = makespan_variance_robustness(cmaxs)
    rng_r = makespan_range_robustness(cmaxs)
    mean = sum(cmaxs) / len(cmaxs) if cmaxs else None
    std = None
    if mean is not None and len(cmaxs) >= 2:
        std = (sum((c - mean) ** 2 for c in cmaxs) / (len(cmaxs) - 1)) ** 0.5
    return {
        "K": len(cmaxs),
        "cmax_scenarios": cmaxs,
        "var_robustness": var_r,
        "range_robustness": rng_r,
        "mean_cmax": mean,
        "std_cmax": std,
    }


if __name__ == "__main__":
    # Démonstration : deux plannings, une op décalée de 3, une op nouvelle ignorée.
    p = {(0, 0): 0, (0, 1): 5, (1, 0): 2}
    c = {(0, 0): 0, (0, 1): 8, (1, 0): 2, (2, 0): 10}  # (2,0) = job arrivé
    print("instab =", instability(p, c), "(attendu 1.0 = 3/3 communes)")

    # Démo robustesse : K=5 scénarios fictifs.
    cmaxs_demo = [55.0, 58.0, 56.0, 60.0, 57.0]
    print("var_robustness  =", makespan_variance_robustness(cmaxs_demo))
    print("range_robustness =", makespan_range_robustness(cmaxs_demo))
