# =============================================================================
# DMO-WCSP — image reproductible pour soumission EJOR
# =============================================================================
# Usage :
#   docker build -t dmo-wcsp .
#   docker run --rm -v "$PWD/results_campaign:/app/results_campaign" dmo-wcsp \
#       zsh run_large_campaign.sh
#
# Image légère basée sur openjdk:11 (Choco + ε-constraint) avec Python pour le
# pipeline d'orchestration, NSGA-II/III, métriques HV, statistiques.
# =============================================================================
FROM eclipse-temurin:11-jdk-jammy

# Évite les questions interactives pendant apt-get.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DMO_JAVA_EXE=java \
    DMO_CHOCO_JAR=/app/choco_solver/target/dmo-choco.jar

# Outils système : Maven (build du jar Choco), Python, zsh (scripts campagne),
# git pour les références bibliographiques optionnelles.
RUN apt-get update && apt-get install -y --no-install-recommends \
        maven \
        python3 \
        python3-pip \
        python3-venv \
        zsh \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Dépendances Python — copiées en premier pour bénéficier du cache de couches.
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# 2. Source — copiée ensuite pour invalider le cache uniquement quand le code change.
COPY . /app

# 3. Build du jar fat. Empêche les rebuilds inutiles : si le pom et les .java
#    n'ont pas changé, Docker réutilisera cette couche.
RUN cd choco_solver && mvn -q -DskipTests package && cd ..

# 4. Sanity check à la construction (échec rapide si la chaîne est cassée).
RUN java -jar /app/choco_solver/target/dmo-choco.jar \
        --instance /app/benchmarks_real/orlib/ft06.txt \
        --heuristic mo_dyn_hd_cacd --objective pareto --points 5 --timeout 10 \
    && echo "[sanity] ft06 OK"

# 5. Par défaut : pipeline reproductible complet (build → campagne → stats → figs).
CMD ["bash", "run_all.sh"]
