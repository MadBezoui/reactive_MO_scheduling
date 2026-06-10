package dmowcsp;

import org.chocosolver.solver.search.strategy.Search;
import org.chocosolver.solver.search.strategy.strategy.AbstractStrategy;
import org.chocosolver.solver.variables.IntVar;

/**
 * Heuristiques de branchement comparées.
 *
 * Les heuristiques CSP classiques (dom, wdeg, activity, input) sont fournies
 * nativement par Choco via {@link Search}. La contribution CACD scope-based
 * est branchée via la clé "mo_dyn_hd_cacd" et déléguée à {@link MoDynHdCacd}.
 * Voir DECISION_01_solveur.md et CONTRIBUTION_REFRAMING.md.
 */
public final class Heuristics {

    private Heuristics() {}

    /** Liste des clés supportées (pour message d'aide). */
    public static final String SUPPORTED = "dom, wdeg, activity, input, mo_dyn_hd_cacd";

    /**
     * Stratégie pour un modèle job-shop, incluant la contribution custom.
     * Si name == "mo_dyn_hd_cacd", branche le VariableSelector personnalisé
     * (et son moniteur d'échecs) ; sinon, heuristique native sur flatStart.
     */
    public static AbstractStrategy<IntVar> strategyFor(String name, JobShopModel jm,
                                                       String perturbations, String weights) {
        if (name != null && name.equalsIgnoreCase("mo_dyn_hd_cacd")) {
            return MoDynHdCacd.strategy(jm, perturbations, weights);
        }
        return forName(name, jm.flatStart);
    }

    public static AbstractStrategy<IntVar> forName(String name, IntVar[] vars) {
        if (name == null) name = "wdeg";
        switch (name.toLowerCase()) {
            case "dom":
                // Plus petit domaine, borne inférieure d'abord.
                return Search.minDomLBSearch(vars);
            case "wdeg":
                // dom/wdeg : pondération des variables par les échecs de contraintes.
                return Search.domOverWDegSearch(vars);
            case "wdeg_es":
                // wdeg + earliest-start explicitement
                // Notons que dans les versions récentes de Choco, domOverWDegSearch utilise déjà IntDomainMin (qui est getLB).
                // Nous l'ajoutons pour clarifier la sémantique et isoler l'effet du sélecteur de valeur.
                return Search.domOverWDegSearch(vars);
            case "activity":
            case "abs":
                // Activity-based search (proche en esprit de CACD).
                return Search.activityBasedSearch(vars);
            case "input":
                return Search.inputOrderLBSearch(vars);
            default:
                // Inconnu (ex. mo_dyn_hd_cacd pas encore branché) → défaut wdeg.
                return Search.domOverWDegSearch(vars);
        }
    }
}
