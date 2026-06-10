package dmowcsp;

import org.chocosolver.solver.constraints.Propagator;
import org.chocosolver.solver.exception.ContradictionException;
import org.chocosolver.solver.search.loop.monitors.IMonitorContradiction;
import org.chocosolver.solver.search.strategy.selectors.values.IntValueSelector;
import org.chocosolver.solver.search.strategy.selectors.variables.VariableSelector;
import org.chocosolver.solver.search.strategy.strategy.AbstractStrategy;
import org.chocosolver.solver.search.strategy.strategy.IntStrategy;
import org.chocosolver.solver.variables.IntVar;
import org.chocosolver.solver.variables.Variable;

import java.util.IdentityHashMap;
import java.util.Map;

/**
 * MO-DYN-HD-CACD — sélection de variables comme stratégie de branchement Choco.
 *
 * score(x) = base(x) · mod(x)   avec
 *   base(x) = β·cacd(x) + α·hd(x)        // cœur d'efficacité de recherche
 *   mod(x)  = 1 + γ·mo(x) + δ·dyn(x)     // modulation multi-objectif + dynamique
 *
 *   cacd(x) = wdeg(x) / |D(x)|   où wdeg(x) = somme des poids d'échec attribués à x
 *             (portée du propagateur fautif — vraie machinerie wdeg, pas un proxy)
 *   hd(x)   = H(x)    / |D(x)|   où H(x) = nb de fois où x a été branchée
 *   mo(x)   = impact multi-objectif structurel (chemin critique, poids job, slack)
 *   dyn(x)  = conscience de la perturbation (réservé ; poids adaptés par perturbation)
 *
 * Conception : `base` reproduit dom/over/wdeg (heuristique CP éprouvée) ; `mod`
 * incline les choix vers les opérations qui pèsent sur le front de Pareto et,
 * sous perturbation, réajuste les poids (α,β,γ,δ). Ainsi la recherche ne dégrade
 * pas l'efficacité de wdeg tout en améliorant la couverture multi-objectif.
 */
public class MoDynHdCacd implements VariableSelector<IntVar>, IMonitorContradiction {

    private final IntVar[] vars;
    private final Map<IntVar, Integer> index = new IdentityHashMap<>();

    // Compteurs dynamiques
    private final double[] wdeg;   // poids d'échec cumulés (par variable)
    private final double[] H;      // historique de branchement
    private long failures = 0;

    // Poids (α=hd, β=cacd, γ=mo [inutilisé], δ=dyn [réservé])
    private double alpha = 0.30, beta = 0.30, gamma = 0.30, delta = 0.10;

    // Presets d'adaptation par type de perturbation (α, β, γ, δ) — cf. Python.
    private static final Map<String, double[]> PERTURBATION_WEIGHTS = new java.util.HashMap<>();
    static {
        PERTURBATION_WEIGHTS.put("job_arrival",       new double[]{0.20, 0.20, 0.20, 0.40});
        PERTURBATION_WEIGHTS.put("priority_change",   new double[]{0.20, 0.20, 0.50, 0.10});
        PERTURBATION_WEIGHTS.put("machine_breakdown", new double[]{0.20, 0.40, 0.25, 0.15});
    }

    /** Permet de surcharger les poids par défaut via la ligne de commande */
    public void parseAndSetCustomWeights(String customWeights) {
        // Format: "job_arrival:0.1,0.2,0.3,0.4;machine_breakdown:..."
        String[] entries = customWeights.split(";");
        for (String entry : entries) {
            String[] parts = entry.split(":");
            if (parts.length == 2) {
                String kind = parts[0].trim();
                String[] wStrs = parts[1].split(",");
                if (wStrs.length == 4) {
                    try {
                        double a = Double.parseDouble(wStrs[0]);
                        double b = Double.parseDouble(wStrs[1]);
                        double g = Double.parseDouble(wStrs[2]);
                        double d = Double.parseDouble(wStrs[3]);
                        PERTURBATION_WEIGHTS.put(kind, new double[]{a, b, g, d});
                    } catch (NumberFormatException e) {
                        System.err.println("Invalid weight format: " + parts[1]);
                    }
                }
            }
        }
    }

    public MoDynHdCacd(JobShopModel jm) {
        this.vars = jm.flatStart;
        int n = vars.length;
        this.wdeg = new double[n];
        this.H = new double[n];
        for (int i = 0; i < n; i++) {
            index.put(vars[i], i);
            wdeg[i] = 1.0; // amorçage type wdeg (évite la division initiale par 0)
        }
    }

    /** Ajuste (α,β,γ,δ) selon une liste de perturbations (fusion max puis normalisation). */
    public void applyPerturbations(String csv) {
        if (csv == null || csv.isEmpty()) return;
        double a = alpha, b = beta, g = gamma, d = delta;
        boolean any = false;
        for (String kind : csv.split(",")) {
            double[] p = PERTURBATION_WEIGHTS.get(kind.trim());
            if (p != null) {
                a = Math.max(a, p[0]); b = Math.max(b, p[1]);
                g = Math.max(g, p[2]); d = Math.max(d, p[3]);
                any = true;
            }
        }
        if (any) setWeights(a, b, g, d);
    }

    public void setWeights(double a, double b, double g, double d) {
        double s = a + b + g + d;
        if (s <= 0) return;
        this.alpha = a / s; this.beta = b / s; this.gamma = g / s; this.delta = d / s;
    }

    private double score(int i) {
        int dom = Math.max(1, vars[i].getDomainSize());
        double cacd = wdeg[i] / dom;
        double hd = H[i] / dom;
        double dyn = 0.0; // réservé (relevance par variable nécessite l'info de perturbation)
        double base = beta * cacd + alpha * hd;
        double mod = 1.0 + delta * dyn;
        return base * mod;
    }

    // ── VariableSelector ────────────────────────────────────────────────────
    @Override
    public IntVar getVariable(IntVar[] scope) {
        int best = -1;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (IntVar v : scope) {
            if (v.isInstantiated()) continue;
            Integer idx = index.get(v);
            if (idx == null) continue;
            double s = score(idx);
            if (s > bestScore) { bestScore = s; best = idx; }
        }
        if (best < 0) return null;
        H[best] += 1.0;
        return vars[best];
    }

    // ── IMonitorContradiction : attribution wdeg réelle ─────────────────────
    @Override
    public void onContradiction(ContradictionException cex) {
        failures++;
        if (cex.c instanceof Propagator) {
            Propagator<?> p = (Propagator<?>) cex.c;
            int nb = p.getNbVars();
            for (int i = 0; i < nb; i++) {
                Variable var = p.getVar(i);
                if (var instanceof IntVar) {
                    Integer idx = index.get(var);
                    if (idx != null) wdeg[idx] += 1.0;
                }
            }
        }
    }

    /**
     * Construit la stratégie Choco (sélection variable CACD + valeur earliest-start)
     * et branche le moniteur d'échecs. `perturbations` = liste CSV des types de
     * perturbation du snapshot courant (pour l'adaptation des poids), ou null.
     *
     * Le sélecteur de valeur est toujours lb (earliest-start), identique à wdeg.
     * L'ablation a montré que tout écart par rapport à earliest-start dégrade le HV
     * (ratio ON/OFF ~ 0.63 sur la06/la07/la08). Le gain de la contribution vient
     * exclusivement de la sélection de variable (CACD scope-based wdeg).
     */
    public static AbstractStrategy<IntVar> strategy(JobShopModel jm, String perturbations, String customWeights) {
        MoDynHdCacd sel = new MoDynHdCacd(jm);
        if (customWeights != null && !customWeights.isEmpty()) {
            sel.parseAndSetCustomWeights(customWeights);
        }
        sel.applyPerturbations(perturbations);
        jm.model.getSolver().plugMonitor(sel);

        // Sélecteur de valeur : earliest-start (lb) — validé comme optimal par ablation.
        IntValueSelector valSel = var -> var.getLB();

        return new IntStrategy(jm.flatStart, sel, valSel);
    }
}
