#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
choco_runner.py — wrapper Python du solveur Choco (remplaçant de run_ace_mo)
============================================================================

Appelle le jar `dmo-choco.jar` en subprocess, lui passe une instance
d'ordonnancement (job-shop .txt ou RCPSP .sm) et une heuristique de branchement,
et renvoie le front de Pareto (ou la solution mono-objectif) sous forme de dict.

Le jar est construit depuis ./choco_solver (voir choco_solver/README.md).
Chemin résolu via la variable d'environnement DMO_CHOCO_JAR, sinon
./choco_solver/target/dmo-choco.jar.

Schéma de sortie (mode pareto) :
    {
      "heuristic": str, "status": str, "front": [[cmax, flow, -rob], ...],
      "front_size": int, "nodes": int, "cpu": float, "error": str|None
    }
"""

import os
import json
import shutil
import subprocess
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))

# Heuristiques exposées par le runner Choco (Heuristics.java)
CHOCO_HEURISTICS = ["dom", "wdeg", "activity", "input", "mo_dyn_hd_cacd"]


def _resolve_java() -> str:
    """Localise l'exécutable Java (cohérent avec main_real)."""
    env = os.environ.get("DMO_JAVA_EXE")
    if env and os.path.isfile(env):
        return env
    return shutil.which("java") or "java"


def _resolve_choco_jar() -> str:
    """Localise dmo-choco.jar : env DMO_CHOCO_JAR sinon target standard."""
    env = os.environ.get("DMO_CHOCO_JAR")
    if env and os.path.isfile(env):
        return env
    return os.path.join(_HERE, "choco_solver", "target", "dmo-choco.jar")


JAVA_EXE = _resolve_java()
CHOCO_JAR = _resolve_choco_jar()


def choco_available() -> bool:
    """True si le jar Choco est présent et exécutable."""
    return os.path.isfile(CHOCO_JAR)


def run_choco(
    instance_path: str,
    heuristic: str = "wdeg",
    timeout: float = 30.0,
    objective: str = "pareto",   # "pareto" | "cmax"
    points: int = 5,
    perturbations: Optional[List[str]] = None,
    weights_dict: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Lance le runner Choco sur une instance et renvoie un dict de résultats.

    `perturbations` : liste des types de perturbation du snapshot courant,
    transmise à mo_dyn_hd_cacd pour l'adaptation des poids (ignorée par les
    heuristiques natives).

    En cas d'absence du jar ou d'erreur, renvoie un dict avec status d'erreur
    plutôt que de lever — pour rester robuste dans une campagne batch.
    """
    if not choco_available():
        return _err(heuristic, "JAR_MISSING",
                    f"jar Choco introuvable : {CHOCO_JAR} "
                    f"(construire via `cd choco_solver && mvn package`)")

    cmd = [
        JAVA_EXE, "-jar", CHOCO_JAR,
        "--instance", instance_path,
        "--heuristic", heuristic,
        "--objective", objective,
        "--points", str(points),
        "--timeout", str(int(timeout)),
    ]
    if perturbations:
        cmd += ["--perturb", ",".join(perturbations)]
    if weights_dict:
        w_parts = []
        for k, v in weights_dict.items():
            w_parts.append(f"{k}:{','.join(map(str, v))}")
        cmd += ["--weights", ";".join(w_parts)]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout + 60
        )
    except subprocess.TimeoutExpired:
        return _err(heuristic, "TO", "timeout subprocess Choco")
    except Exception as e:  # pragma: no cover
        return _err(heuristic, "ERR", str(e))

    out = (proc.stdout or "").strip()
    parsed = _parse_last_json(out)
    if parsed is None:
        tail = (proc.stderr or out)[-500:]
        return _err(heuristic, "PARSE_ERR", f"sortie non-JSON : {tail}")

    # Normalisation du schéma
    front = parsed.get("front")
    result: Dict[str, Any] = {
        "heuristic": parsed.get("heuristic", heuristic),
        "status": parsed.get("status", "UNKNOWN"),
        "nodes": parsed.get("nodes", 0),
        "cpu": parsed.get("cpu", 0.0),
        "error": parsed.get("error"),
    }
    # Solution représentante (pour la stabilité inter-snapshots), si présente.
    if parsed.get("rep_ops") is not None and parsed.get("rep_starts") is not None:
        result["rep_ops"] = parsed["rep_ops"]
        result["rep_starts"] = parsed["rep_starts"]

    if front is not None:
        result["front"] = front
        result["front_size"] = parsed.get("front_size", len(front))
    else:
        # Garde défensive : depuis le correctif Main.java, les chemins mono-objectif
        # émettent eux aussi un "front" [cmax, flowtime, -robustesse]. Cette branche
        # ne devrait donc plus être atteinte pour une sortie Choco normale ; si elle
        # l'est (sortie ancienne/dégradée), la robustesse est inconnue et laissée à
        # 0.0 faute de schedule pour la recalculer côté Python.
        cmax = parsed.get("cmax")
        flow = parsed.get("flowtime")
        if cmax is not None:
            result["front"] = [[float(cmax),
                                float(flow) if flow is not None else float(cmax),
                                0.0]]
            result["front_size"] = 1
        else:
            result["front"] = []
            result["front_size"] = 0
        result["cmax"] = cmax
        result["flowtime"] = flow
    return result


def _parse_last_json(text: str) -> Optional[Dict[str, Any]]:
    """Récupère le dernier objet JSON sur stdout (le runner émet une ligne JSON)."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _err(heuristic: str, status: str, msg: str) -> Dict[str, Any]:
    return {
        "heuristic": heuristic, "status": status, "front": [],
        "front_size": 0, "nodes": 0, "cpu": 0.0, "error": msg,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Test du wrapper Choco")
    ap.add_argument("--instance", required=True)
    ap.add_argument("--heuristic", default="wdeg")
    ap.add_argument("--objective", default="pareto")
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=30.0)
    a = ap.parse_args()
    res = run_choco(a.instance, a.heuristic, a.timeout, a.objective, a.points)
    print(json.dumps(res, indent=2, ensure_ascii=False))
