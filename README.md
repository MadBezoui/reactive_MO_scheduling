# DMO-WCSP — Ordonnancement dynamique multi-objectif sous contraintes

Banc d'essai comparant des heuristiques de **sélection de variables CP** (`dom`,
`wdeg`, `activity`, `input`, et la contribution `mo_dyn_hd_cacd`) et des
métaheuristiques (NSGA-II, NSGA-III) sur des instances réelles
d'ordonnancement (**job-shop OR-Library** et **RCPSP PSPLIB**) traitées comme
des **DMO-WCSP** : problèmes multi-objectifs résolus sur une séquence de
snapshots perturbés au fil du temps.

Objectifs Pareto : **makespan** ($C_{max}$), **flowtime** (somme des temps de
complétion) et **robustesse** (slack moyen vis-à-vis d'un majorant relaxé du
makespan ; voir `paper_EJOR/main.tex` §3.3 pour la définition non circulaire et
les limites du proxy).

**Contribution centrale.** *CACD — Constraint-Aware Conflict-Directed
variable selection.* Score d'une variable $x$ :
$\text{cacd}(x) = \frac{\sum_{c \in \mathcal{C}(x)} w(c)}{|D(x)|}$,
où $w(c)$ est le poids d'échec accumulé par le propagateur $c$ et $|D(x)|$ la
taille de domaine courante. Contrairement à `dom/wdeg`, CACD crédite **toute la
portée** du propagateur fautif, pas seulement la variable failee. Implémenté
comme `VariableSelector` Choco dans `choco_solver/src/main/java/dmowcsp/MoDynHdCacd.java`
et accessible via la clé `mo_dyn_hd_cacd`.

Voir `CONTRIBUTION_REFRAMING.md` pour le recadrage scientifique (CACD pur ; les
termes auxiliaires HD/MO/DYN et le sélecteur de valeur slack-preserving sont
documentés comme non contributifs et reportés en travaux futurs).

---

## Installation

```bash
pip install -r requirements.txt
```

Python ≥ 3.10 recommandé. Dépendances : numpy, pymoo, ortools, scipy,
scikit-posthocs, pandas, matplotlib, seaborn.

### Solveur Choco

La contribution CACD est compilée en un fat-jar Choco (Java 11+, Maven).

```bash
cd choco_solver && mvn package && cd ..
# → choco_solver/target/dmo-choco.jar
```

Variables d'environnement (sinon valeurs par défaut détectées) :

```bash
export DMO_JAVA_EXE=/chemin/vers/java
export DMO_CHOCO_JAR=$(pwd)/choco_solver/target/dmo-choco.jar
```

### Image Docker reproductible

```bash
docker build -t dmo-wcsp .
docker run --rm -v "$PWD/results_campaign:/app/results_campaign" dmo-wcsp \
    bash run_all.sh
```

---

## Structure du projet

| Fichier / dossier | Rôle |
|---|---|
| `main_real.py` | Pipeline Python : parsing, perturbations, runners Choco/NSGA-II/NSGA-III/CP-SAT, métriques HV/spacing/stability, rapport. |
| `choco_solver/` | Modèle CP Choco (job-shop + RCPSP) et heuristiques de branchement (`dom`, `wdeg`, `activity`, `input`, `mo_dyn_hd_cacd`). |
| `choco_runner.py` | Wrapper subprocess du fat-jar Choco. |
| `stability.py` | Métrique de stabilité (déviation des temps de début entre snapshots) et robustesse Monte-Carlo non circulaire. |
| `run_campaign.py` | Analyse statistique : Friedman + Nemenyi + Wilcoxon + Â₁₂ sur le CSV bruts. |
| `generate_figures.py` | Production de toutes les figures du paper (boxplots, CD plots Nemenyi job-shop/RCPSP, stability boxplot, fronts de Pareto). |
| `ablation.py` | Ablation des termes (`hd`/`cacd`/`mo`/`dyn`) sans recompilation. |
| `tune_hyperparams.py` | Random search des poids sur set de validation (la06/la07/la08). |
| `milp_baseline.py` | Baseline CP-SAT (OR-Tools). |
| `download_orlib.py` | Téléchargement la01-la40 + abz5-9 + orb01-10. |
| `download_psplib.py` | Téléchargement PSPLIB j30/j60/j90/j120 (déjà présent). |
| `run_large_campaign.sh` | Campagne complète : 30 seeds × 5 snapshots × ~80 instances job-shop+RCPSP. |
| `run_all.sh` | **Pipeline reproductible end-to-end** : build → sanity → download → campagne → stats × 3 métriques → figures. |
| `Dockerfile` | Image reproductible (eclipse-temurin:11 + maven + python + zsh). |
| `benchmarks_real/` | `orlib/` (job-shop), `j30/j60/j90/j120/` (RCPSP). |
| `paper_EJOR/` | Brouillon paper EJOR (main.tex, references.bib, figures/, tikz/). |
| `PROJECT_STATE.md` | État du projet & document de reprise (résumé Δ par session). |
| `TODO.md` | Feuille de route vers soumission EJOR (chemin critique + déjà fait + incohérences). |
| `CONTRIBUTION_REFRAMING.md` | Recadrage scientifique (CACD pur, ablation honnête). |

---

## Utilisation

### Pipeline reproductible complet

```bash
bash run_all.sh
# Variantes :
#   bash run_all.sh --skip-build     (jar déjà compilé)
#   bash run_all.sh --skip-campaign  (rejouer stats + figures seulement)
```

### Étape par étape

```bash
# 1. Build du jar
cd choco_solver && mvn package && cd ..

# 2. Sanity check ft06 (Cmax 55 attendu)
java -jar choco_solver/target/dmo-choco.jar \
    --instance benchmarks_real/orlib/ft06.txt \
    --heuristic mo_dyn_hd_cacd --objective pareto --points 5 --timeout 10

# 3. Téléchargement instances job-shop (idempotent)
python3 download_orlib.py

# 4. Campagne large (~4-8 h sur machine locale)
zsh run_large_campaign.sh

# 5. Statistiques
python3 run_campaign.py --results results_campaign/raw_results.csv \
    --out-dir results_campaign --metric hv
python3 run_campaign.py --results results_campaign/raw_results.csv \
    --out-dir results_campaign --metric spacing
python3 run_campaign.py --results results_campaign/raw_results.csv \
    --out-dir results_campaign --metric cpu

# 6. Figures EJOR (boxplots + CD plots Nemenyi + stability)
python3 generate_figures.py \
    --results results_campaign/raw_results.csv \
    --out-dir paper_EJOR/figures
```

### Mode test rapide

```bash
python3 main_real.py --instances-dir ./benchmarks_real \
    --out-dir ./results_test --timeout 10 --seeds 1 --test
```

---

## Reproductibilité

- **Seeds appariées.** Chaque heuristique reçoit les mêmes seeds $[0..29]$ sur
  chaque instance, et toutes les heuristiques voient les mêmes snapshots
  perturbés (`PerturbationGenerator(seed=42)` partagé). Design apparié pour
  Friedman / Wilcoxon.
- **Split train/val/test.** La06, la07, la08 sont réservés à la
  validation/ablation et exclus du set de test.
- **NSGA-III.** `pop_size = 91` (Das-Dennis 3 objectifs, $n_{\text{partitions}} = 12$).
- **Dockerfile** : image reproductible avec build du jar intégré.
- Voir `PROJECT_STATE.md` §11 et `TODO.md` §5.

---

## État scientifique et publication

Cible revue : **European Journal of Operational Research (EJOR)**.

Le brouillon `paper_EJOR/main.tex` est aligné sur la thèse CACD scope-based
pure ; les chiffres sont marqués `\placeholder{TBD}` en attendant la campagne
large. Ablation honnête (Option A) : les termes auxiliaires HD/MO/DYN et le
sélecteur de valeur slack-preserving sont rapportés comme non contributifs et
reportés en travaux futurs.

Voir `TODO.md` §1 pour le chemin critique restant.
