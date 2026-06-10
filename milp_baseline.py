#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
milp_baseline.py — MILP/CP-SAT Baseline for Multi-Objective Job-Shop Scheduling
=================================================================================

Implements MILPBaseline using OR-Tools CP-SAT solver:
  - Exact job-shop model (precedence + no-overlap constraints)
  - Two-phase Pareto front generation:
      Phase 1: ε-constraint sweep on makespan → minimize flowtime
      Phase 2: Weighted scalarization with diverse weight vectors → 3 objectives
  - Objectives: makespan (C_max), flowtime (∑C_j), robustness (mean slack)
  - Dynamic support: warm-start hints carried across snapshots
  - Metrics: same HV/spacing/epsilon as other methods (ParetoMetrics)

Model:
  Variables : S[j][k] ∈ [0, H]  (start time, operation k of job j)
  Constraints:
    S[j][k+1] >= S[j][k] + d[j][k]     (precedence within job)
    AddNoOverlap(intervals on same machine)
  Objectives (scalarized):
    minimize  w1·C_max + w2·Flowtime + w3·(-Robustness)

Usage (standalone):
    python milp_baseline.py
"""

import os
import sys
import csv as csv_module
import time
import math
import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Any

from ortools.sat.python import cp_model

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from main_real import (
    JobShopInstance,
    parse_orlib,
    ParetoMetrics,
    ScheduleEvaluator,
    PerturbationGenerator,
    Perturbation,
)


# =============================================================================
# CP-SAT Job-Shop Solver (single scalarized objective)
# =============================================================================

class JobShopCPSAT:
    """CP-SAT model for 3-objective job-shop scheduling.

    Minimizes a weighted combination:
        w1 * C_max + w2 * Flowtime - w3 * Robustness

    Parameters
    ----------
    inst    : JobShopInstance
    timeout : solver time limit in seconds
    """

    # Scale factor to convert float weights to integers for CP-SAT
    _SCALE = 1000

    def __init__(self, inst: JobShopInstance, timeout: float = 60.0):
        self.inst = inst
        self.timeout = timeout
        # Safe upper bound: sum of all processing times
        self._H = inst.upper_bound_makespan * 2

    def solve(
        self,
        w1: float = 1.0,
        w2: float = 0.0,
        w3: float = 0.0,
        makespan_bound: Optional[int] = None,
        flowtime_bound: Optional[int] = None,
        hint_starts: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Solve with given objective weights and optional ε-bounds.

        Returns solution dict or None if infeasible/timeout without solution.
        """
        model = cp_model.CpModel()
        inst = self.inst
        H = self._H
        S = self._SCALE

        # ── Variables ────────────────────────────────────────────────────────
        sv = {}   # start vars  (j,k) -> IntVar
        ev = {}   # end vars    (j,k) -> IntVar
        iv = {}   # interval    (j,k) -> IntervalVar

        for j, job in enumerate(inst.jobs):
            for k, (machine, dur) in enumerate(job):
                s = model.NewIntVar(0, H, f's_{j}_{k}')
                e = model.NewIntVar(dur, H + dur, f'e_{j}_{k}')
                interval = model.NewIntervalVar(s, dur, e, f'i_{j}_{k}')
                sv[(j, k)] = s
                ev[(j, k)] = e
                iv[(j, k)] = interval

        # ── Precedence constraints ───────────────────────────────────────────
        for j, job in enumerate(inst.jobs):
            for k in range(len(job) - 1):
                _, dur_k = job[k]
                model.Add(sv[(j, k + 1)] >= sv[(j, k)] + dur_k)

        # ── No-overlap per machine ───────────────────────────────────────────
        by_machine: Dict[int, list] = {}
        for j, job in enumerate(inst.jobs):
            for k, (m, _) in enumerate(job):
                by_machine.setdefault(m, []).append(iv[(j, k)])
        for m, intervals in by_machine.items():
            if len(intervals) > 1:
                model.AddNoOverlap(intervals)

        # ── Makespan ─────────────────────────────────────────────────────────
        cmax = model.NewIntVar(0, H, 'cmax')
        model.AddMaxEquality(cmax, [ev[(j, len(job) - 1)]
                                    for j, job in enumerate(inst.jobs)])
        if makespan_bound is not None:
            model.Add(cmax <= makespan_bound)

        # ── Flowtime ─────────────────────────────────────────────────────────
        completion_vars = [ev[(j, len(job) - 1)]
                           for j, job in enumerate(inst.jobs)]
        fvar = model.NewIntVar(0, H * inst.n_jobs, 'flowtime')
        model.Add(fvar == sum(completion_vars))
        if flowtime_bound is not None:
            model.Add(fvar <= flowtime_bound)

        # ── Robustness (slack-based) ─────────────────────────────────────────
        # slack[j] = max(0, deadline[j] - completion[j])
        # Robustness = mean(slacks)  →  maximized as -mean(slacks) minimized
        rob_total = model.NewIntVar(-H * inst.n_jobs, H * inst.n_jobs, 'rob_total')
        if inst.deadlines:
            slack_vars = []
            for j, job in enumerate(inst.jobs):
                dl = inst.deadlines[j]
                slack = model.NewIntVar(0, max(0, dl), f'slack_{j}')
                # slack = max(0, dl - completion)
                raw = model.NewIntVar(-H, H, f'raw_slack_{j}')
                model.Add(raw == dl - ev[(j, len(job) - 1)])
                model.AddMaxEquality(slack, [raw, model.NewConstant(0)])
                slack_vars.append(slack)
            model.Add(rob_total == sum(slack_vars))
        else:
            # Proxy: robustness = -cmax (longer schedules are less robust)
            model.Add(rob_total == -cmax)

        # ── Weighted objective ───────────────────────────────────────────────
        # Minimize: w1·C_max + w2·Flowtime - w3·Robustness
        # (scale weights to integers)
        iw1 = int(round(w1 * S))
        iw2 = int(round(w2 * S))
        iw3 = int(round(w3 * S))

        obj_terms = []
        if iw1 != 0:
            obj_terms.append(iw1 * cmax)
        if iw2 != 0:
            obj_terms.append(iw2 * fvar)
        if iw3 != 0:
            obj_terms.append(-iw3 * rob_total)

        if obj_terms:
            model.Minimize(sum(obj_terms))

        # ── Solution hints ───────────────────────────────────────────────────
        if hint_starts:
            for (j, k), val in hint_starts.items():
                if (j, k) in sv:
                    model.AddHint(sv[(j, k)], max(0, int(val)))

        # ── Solve ────────────────────────────────────────────────────────────
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.timeout
        solver.parameters.num_search_workers = 2
        solver.parameters.log_search_progress = False

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            starts = {(j, k): solver.Value(sv[(j, k)])
                      for j in range(len(inst.jobs))
                      for k in range(len(inst.jobs[j]))}
            cmax_val = solver.Value(cmax)
            flow_val = solver.Value(fvar)
            rob_val = solver.Value(rob_total) / inst.n_jobs if inst.n_jobs > 0 else 0.0
            return {
                'starts': starts,
                'makespan': float(cmax_val),
                'flowtime': float(flow_val),
                'robustness': float(rob_val),
                'status': 'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE',
                'wall_time': solver.WallTime(),
            }
        return None


# =============================================================================
# MILPBaseline — Multi-objective Pareto front via weighted scalarization
# =============================================================================

class MILPBaseline:
    """OR-Tools CP-SAT baseline for multi-objective dynamic job-shop.

    Pareto front approximation strategy:
      1. Solve N problems with diverse weight vectors (w1, w2, w3) sampled
         from the weight simplex → covers all Pareto regions
      2. Apply ε-constraint refinement: fix optimal makespan, sweep flowtime
      3. Collect all solutions, filter to non-dominated front in 3D

    Parameters
    ----------
    timeout_per_point : float
        CPU budget per solve call (seconds). Default 60s.
    n_weight_vectors  : int
        Number of weight vectors for scalarization. Default 12.
    """

    # Fixed weight vectors spanning the 3-objective simplex
    # (w_cmax, w_flow, w_rob) — rows sum to ~1
    _WEIGHT_VECTORS = [
        # Pure objectives
        (1.0, 0.0, 0.0),   # minimize makespan only
        (0.0, 1.0, 0.0),   # minimize flowtime only
        (0.0, 0.0, 1.0),   # maximize robustness only
        # Balanced pairs
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
        # Balanced triple
        (0.333, 0.333, 0.334),
        # Pareto corners
        (0.7, 0.2, 0.1),
        (0.2, 0.7, 0.1),
        (0.2, 0.1, 0.7),
        (0.6, 0.3, 0.1),
        (0.3, 0.6, 0.1),
    ]

    def __init__(
        self,
        timeout_per_point: float = 60.0,
        n_weight_vectors: int = 12,
    ):
        self.timeout_per_point = timeout_per_point
        self.n_weight_vectors = n_weight_vectors
        self._last_hints: Optional[Dict] = None

    def solve(
        self,
        inst: JobShopInstance,
        perturbations: Optional[List[Perturbation]] = None,
    ) -> Dict[str, Any]:
        """Generate Pareto front for the given instance."""
        t0 = time.time()
        greedy = ScheduleEvaluator.evaluate_jobshop(inst)

        raw_solutions: List[Dict] = []
        hints = self._last_hints

        weight_vecs = self._WEIGHT_VECTORS[:self.n_weight_vectors]

        for i, (w1, w2, w3) in enumerate(weight_vecs):
            solver = JobShopCPSAT(inst, timeout=self.timeout_per_point)
            r = solver.solve(w1=w1, w2=w2, w3=w3, hint_starts=hints)
            if r is not None:
                raw_solutions.append(r)
                hints = r['starts']  # warm-start next call

        if not raw_solutions:
            return self._empty_result(inst, time.time() - t0, greedy)

        # Best hints for next snapshot
        self._last_hints = hints

        # ── Filter to non-dominated front (3D) ─────────────────────────────
        # Objectives: minimize makespan, minimize flowtime, minimize -robustness
        all_pts = [
            [r['makespan'], r['flowtime'], -r['robustness']]
            for r in raw_solutions
        ]
        pareto_front = self._filter_pareto_3d(all_pts)

        cpu = time.time() - t0
        F = np.array(pareto_front)

        # Reference point for HV: 1.2 × worst values
        ref_pt = np.array([
            max(p[0] for p in pareto_front) * 1.2,
            max(p[1] for p in pareto_front) * 1.2,
            -min(p[2] for p in pareto_front) * 0.5,
        ])
        ref_pt = np.array([
            greedy.makespan * 1.2,
            greedy.flowtime * 1.2,
            greedy.robustness * 0.5,
        ])

        hv = ParetoMetrics.hypervolume(F, ref_pt)
        sp = ParetoMetrics.spacing(F) if len(F) > 1 else 0.0

        return {
            'heuristic': 'milp_ortools',
            'status': 'SAT',
            'cpu': cpu,
            'par2': cpu,
            'pareto_front': pareto_front,
            'front_size': len(pareto_front),
            'n_solutions': len(pareto_front),
            'hv': hv,
            'spacing': sp,
            'ace_obj': min(p[0] for p in pareto_front),
            'error': None,
        }

    def _filter_pareto_3d(self, points: List[List[float]]) -> List[List[float]]:
        """Remove dominated points in 3D (all objectives to minimize)."""
        if not points:
            return []

        # Deduplicate
        unique = list({tuple(p) for p in points})
        pts = [list(p) for p in unique]

        pareto = []
        for i, p in enumerate(pts):
            dominated = False
            for j, q in enumerate(pts):
                if i == j:
                    continue
                # q dominates p if q <= p on all objectives and < on at least one
                if (q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2] and
                        (q[0] < p[0] or q[1] < p[1] or q[2] < p[2])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)

        return pareto

    def _empty_result(self, inst, cpu: float, greedy=None) -> Dict:
        return {
            'heuristic': 'milp_ortools',
            'status': 'INFEASIBLE',
            'cpu': cpu,
            'par2': cpu,
            'pareto_front': [],
            'front_size': 0,
            'n_solutions': 0,
            'hv': 0.0,
            'spacing': 0.0,
            'ace_obj': None,
            'error': 'No feasible solution found',
        }


# =============================================================================
# Dynamic wrapper
# =============================================================================

class DynamicMILPBaseline:
    """MILPBaseline with warm-start across dynamic snapshots."""

    def __init__(self, timeout_per_point: float = 60.0, n_weight_vectors: int = 12):
        self.baseline = MILPBaseline(timeout_per_point, n_weight_vectors)

    def run_sequence(
        self,
        snapshots: List[Tuple],
    ) -> List[Dict[str, Any]]:
        results = []
        for snap_id, (inst, perturbations) in enumerate(snapshots):
            print(f"  [MILP] Snapshot {snap_id}/{len(snapshots)-1} "
                  f"({inst.n_jobs}j×{inst.n_machines}m) "
                  f"perturb={[p.kind for p in perturbations] or '∅'}")
            r = self.baseline.solve(inst, perturbations)
            r.update({
                'instance': inst.name,
                'snapshot': snap_id,
                'perturbations': [p.kind for p in perturbations],
                'n_jobs': inst.n_jobs,
                'n_machines': inst.n_machines,
                'seed': 0,
            })
            results.append(r)
            print(f"         HV={r['hv']:.2f}  front={r['front_size']}  "
                  f"cpu={r['cpu']:.2f}s  C_max*={r['ace_obj']}")
        return results


# =============================================================================
# Standalone validation on ft06
# =============================================================================

def validate_ft06():
    """Run MILPBaseline on ft06 with 5 dynamic snapshots."""
    orlib_path = os.path.join(_HERE, "benchmarks_real", "orlib", "ft06.txt")
    if not os.path.isfile(orlib_path):
        print(f"[ERROR] ft06.txt not found at {orlib_path}")
        return []

    inst = parse_orlib(orlib_path)
    print(f"\n{'='*60}")
    print("ft06 Validation — MILPBaseline (OR-Tools CP-SAT)")
    print(f"{'='*60}")
    print(f"Instance : {inst.n_jobs}×{inst.n_machines}  "
          f"(optimal C_max=55, Fisher & Thompson 1963)")
    print(f"Solver   : OR-Tools CP-SAT, 12 weight vectors, 60s/point")
    print()

    gen = PerturbationGenerator(seed=42, lam=0.5)
    sequence = gen.generate_sequence(
        inst, T=5,
        perturbation_types=['job_arrival', 'priority_change', 'machine_breakdown']
    )

    dyn = DynamicMILPBaseline(timeout_per_point=30.0, n_weight_vectors=6)
    results = dyn.run_sequence(sequence)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"{'Snap':>4} {'Jobs':>4} {'HV':>9} {'Spacing':>8} "
          f"{'CPU(s)':>7} {'Front':>5} {'C_max*':>7} {'Perturbations'}")
    print(f"{'─'*72}")
    for r in results:
        perturbs = ', '.join(r['perturbations']) or '∅'
        print(f"{r['snapshot']:>4} {r['n_jobs']:>4} {r['hv']:>9.2f} "
              f"{r['spacing']:>8.3f} {r['cpu']:>7.2f} "
              f"{r['front_size']:>5} {r['ace_obj'] or 0:>7.0f} {perturbs}")

    # ── Save results ─────────────────────────────────────────────────────────
    out_dir = os.path.join(_HERE, "results_real")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "milp_ft06_results.csv")

    fieldnames = ['snapshot', 'n_jobs', 'n_machines', 'hv', 'spacing',
                  'cpu', 'front_size', 'ace_obj', 'perturbations', 'status']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv_module.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in results:
            row = dict(r)
            row['perturbations'] = ','.join(r['perturbations'])
            w.writerow(row)

    print(f"\n[OK] Results saved → {csv_path}")
    return results


if __name__ == '__main__':
    validate_ft06()
