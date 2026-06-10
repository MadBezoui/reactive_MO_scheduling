#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_orlib.py — télécharge les instances OR-Library job-shop manquantes.

Télécharge la11–la40, abz5–abz9, orb01–orb10 depuis la source OR-Library
(people.brunel.ac.uk) dans benchmarks_real/orlib/.

Usage :
    python3 download_orlib.py [--out-dir benchmarks_real/orlib]
"""

import os
import time
import argparse
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

OUT_DIR = "benchmarks_real/orlib"

# Base URL OR-Library job-shop
BASE = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files"

# Instances à télécharger (les la01-la10 + ft06/ft10 sont déjà présents)
INSTANCES = (
    # la-series complète (la01-la10 déjà présents)
    [f"la{i:02d}" for i in range(11, 41)]   # la11–la40
    + [f"abz{i}" for i in range(5, 10)]      # abz5–abz9
    + [f"orb{i:02d}" for i in range(1, 11)]  # orb01–orb10
)

# Correspondance nom → fichier source (OR-Library regroupe parfois plusieurs instances)
# La série la est dans jobshop1.txt (la01-la40 groupées) — on la parse nous-mêmes.
# abz et orb ont chacun leur fichier.

LA_SOURCE = f"{BASE}/jobshop1.txt"  # contient la01-la40 groupées


def download_jobshop1(out_dir: str):
    """Télécharge jobshop1.txt et en extrait la11–la40 en fichiers individuels."""
    raw_path = os.path.join(out_dir, "jobshop1_full.txt")
    if not os.path.exists(raw_path):
        print(f"Téléchargement {LA_SOURCE} …")
        urllib.request.urlretrieve(LA_SOURCE, raw_path)
        print(f"  → {raw_path}")
    else:
        print(f"  (déjà présent) {raw_path}")

    # Parser : chaque instance commence par "instance la<N>"
    with open(raw_path) as f:
        content = f.read()

    blocks = {}
    current_name = None
    current_lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("instance"):
            if current_name:
                blocks[current_name] = "\n".join(current_lines)
            parts = stripped.split()
            current_name = parts[1] if len(parts) > 1 else None
            current_lines = []
        elif stripped.startswith("+"):
            continue  # lignes séparateurs
        elif stripped:
            current_lines.append(stripped)
    if current_name:
        blocks[current_name] = "\n".join(current_lines)

    written = 0
    for name, body in blocks.items():
        out_path = os.path.join(out_dir, f"{name}.txt")
        if not os.path.exists(out_path):
            with open(out_path, "w") as f:
                f.write(body + "\n")
            written += 1
    print(f"  {written} nouvelles instances la-series écrites dans {out_dir}")


def download_individual(name: str, out_dir: str):
    """Télécharge une instance individuelle (abz, orb, etc.)."""
    out_path = os.path.join(out_dir, f"{name}.txt")
    if os.path.exists(out_path):
        return
    url = f"{BASE}/{name}.txt"
    try:
        urllib.request.urlretrieve(url, out_path)
        print(f"  ✓ {name}")
        time.sleep(0.2)  # politesse
    except Exception as e:
        print(f"  ✗ {name} : {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # la11-la40 via jobshop1.txt
    download_jobshop1(args.out_dir)

    # abz et orb individuellement
    print("Téléchargement abz/orb …")
    for name in [f"abz{i}" for i in range(5, 10)] + [f"orb{i:02d}" for i in range(1, 11)]:
        download_individual(name, args.out_dir)

    # Bilan
    files = [f for f in os.listdir(args.out_dir) if f.endswith(".txt")]
    print(f"\nTotal instances job-shop disponibles : {len(files)}")
    print("  " + "  ".join(sorted(files)))


if __name__ == "__main__":
    main()
