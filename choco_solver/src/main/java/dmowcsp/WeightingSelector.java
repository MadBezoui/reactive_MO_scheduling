package dmowcsp;

import org.chocosolver.solver.constraints.Constraint;
import org.chocosolver.solver.constraints.Propagator;
import org.chocosolver.solver.exception.ContradictionException;
import org.chocosolver.solver.search.loop.monitors.IMonitorContradiction;
import org.chocosolver.solver.search.strategy.selectors.variables.VariableSelector;
import org.chocosolver.solver.variables.IntVar;
import org.chocosolver.solver.variables.Variable;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * WeightingSelector — un SEUL sélecteur de variables paramétré par le schéma de
 * pondération des conflits, pour la comparaison contrôlée (papier Path A).
 *
 * La SEULE chose qui varie entre les trois modes est la règle d'incrément du
 * poids et la façon de le sommer ; le sélecteur de valeur, le modèle, les
 * variables et le budget sont identiques. Ceci isole l'effet du schéma de
 * pondération, conformément à Wattez et al. (2019), §IV.
 *
 *  - WDEG_2004 : dom/wdeg original (Boussemart, Hemery, Lecoutre & Saïs, ECAI
 *                2004). Un poids SCALAIRE par contrainte, +1 à la contrainte
 *                fautive ; wdeg(x) = Σ_{c∋x, |fut(c)|>1} w(c). Score = wdeg(x)/|D(x)|.
 *  - ABSCON    : raffinement par variable (Wattez, Lecoutre, Paparrizou &
 *                Tabary, "Refining Constraint Weighting", ICTAI 2019).
 *                +1 à chaque variable FUTURE de la contrainte fautive.
 *                wdeg(x) = Σ_{c∋x} w(c,x). Score = wdeg(x)/|D(x)|.
 *  - HD_2004   : WDEG_2004 augmenté du terme HD (historique de branchement) du
 *                manuscrit : score = (β·wdeg2004(x) + α·H(x)) / |D(x)|, où H(x)
 *                = nb de fois où x a été choisie pour brancher. Isole la SEULE
 *                déviation réelle du code original par rapport à dom/wdeg2004.
 *
 * Valeur de branchement : earliest-start (getLB), identique pour tous les modes
 * et identique au domOverWDegSearch de Choco (IntDomainMin).
 */
public class WeightingSelector implements VariableSelector<IntVar>, IMonitorContradiction {

    public enum Mode { WDEG_2004, ABSCON, HD_2004 }

    private final Mode mode;
    private final IntVar[] vars;
    private final Map<IntVar, Integer> vIndex = new IdentityHashMap<>();

    // Indexation des propagateurs.
    private final Map<Propagator<?>, Integer> pIndex = new IdentityHashMap<>();
    private final List<Propagator<?>> props = new ArrayList<>();
    private double[] cw;                       // WDEG_2004 : poids scalaire par contrainte
    private final List<int[]> propScope = new ArrayList<>();   // propScope[p] = indices de variables (IntVar connues) dans la portée de p

    // ABSCON : poids par (propagateur, variable). Indexé [pId] -> double[scopeLen].
    private final List<double[]> cwVar = new ArrayList<>();

    // HD : historique de branchement par variable.
    private final double[] H;

    // Mesure de la fraîcheur de portée aux échecs (test de la Proposition 1) :
    // à chaque wipeout, fraction des variables de la portée du propagateur fautif
    // encore non instanciées. Sa moyenne estime b(c) ; ε ≈ 1 − moyenne.
    private double freshSum = 0.0;
    private long freshCount = 0;
    public double getFreshSum()   { return freshSum; }
    public long   getFreshCount() { return freshCount; }
    public double meanScopeFreshness() { return freshCount == 0 ? Double.NaN : freshSum / freshCount; }

    // Poids du terme HD (mode HD_2004 uniquement). Par défaut α=β (le manuscrit utilise α=β=0.30).
    private double alpha = 0.5, beta = 0.5;

    public WeightingSelector(JobShopModel jm, Mode mode) {
        this(jm.flatStart, jm.model.getCstrs(), mode);
    }

    public WeightingSelector(IntVar[] decisionVars, Constraint[] allCstrs, Mode mode) {
        this.mode = mode;
        this.vars = decisionVars;
        this.H = new double[decisionVars.length];
        for (int i = 0; i < decisionVars.length; i++) vIndex.put(decisionVars[i], i);

        // Recense les propagateurs et, pour chacun, les indices des variables de décision dans sa portée.
        for (Constraint c : allCstrs) {
            for (Propagator<?> p : c.getPropagators()) {
                if (pIndex.containsKey(p)) continue;
                int pid = props.size();
                pIndex.put(p, pid);
                props.add(p);
                List<Integer> sc = new ArrayList<>();
                for (int i = 0; i < p.getNbVars(); i++) {
                    Variable v = p.getVar(i);
                    Integer vi = (v instanceof IntVar) ? vIndex.get(v) : null;
                    if (vi != null) sc.add(vi);
                }
                int[] arr = new int[sc.size()];
                for (int i = 0; i < arr.length; i++) arr[i] = sc.get(i);
                propScope.add(arr);
                cwVar.add(new double[arr.length]);
                java.util.Arrays.fill(cwVar.get(pid), 1.0); // amorçage type wdeg
            }
        }
        this.cw = new double[props.size()];
        java.util.Arrays.fill(this.cw, 1.0);
    }

    /** Permet de fixer α,β pour le mode HD_2004 (normalisés). */
    public void setHdWeights(double a, double b) {
        double s = a + b;
        if (s > 0) { this.alpha = a / s; this.beta = b / s; }
    }

    // ── Score selon le mode ─────────────────────────────────────────────────
    private double wdeg2004(int vi, IntVar v) {
        // Σ_{p ∋ v, |fut(p)|>1} cw[p]
        double w = 0.0;
        for (int pid = 0; pid < props.size(); pid++) {
            int[] sc = propScope.get(pid);
            boolean contains = false;
            int futur = 0;
            for (int idx : sc) {
                if (idx == vi) contains = true;
                if (!vars[idx].isInstantiated()) futur++;
            }
            if (contains && futur > 1) w += cw[pid];
        }
        return w;
    }

    private double abscon(int vi) {
        // Σ_{p ∋ v} cwVar[p][pos(v)]
        double w = 0.0;
        for (int pid = 0; pid < props.size(); pid++) {
            int[] sc = propScope.get(pid);
            for (int k = 0; k < sc.length; k++) {
                if (sc[k] == vi) { w += cwVar.get(pid)[k]; break; }
            }
        }
        return w;
    }

    private double score(int vi, IntVar v) {
        int dom = Math.max(1, v.getDomainSize());
        switch (mode) {
            case ABSCON:
                return abscon(vi) / dom;
            case HD_2004:
                return (beta * wdeg2004(vi, v) + alpha * H[vi]) / dom;
            case WDEG_2004:
            default:
                return wdeg2004(vi, v) / dom;
        }
    }

    // ── VariableSelector ────────────────────────────────────────────────────
    @Override
    public IntVar getVariable(IntVar[] scope) {
        int best = -1;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (IntVar v : scope) {
            if (v.isInstantiated()) continue;
            Integer vi = vIndex.get(v);
            if (vi == null) continue;
            double s = score(vi, v);
            if (s > bestScore) { bestScore = s; best = vi; }
        }
        if (best < 0) return null;
        H[best] += 1.0;
        return vars[best];
    }

    // ── IMonitorContradiction : règle d'incrément spécifique au mode ─────────
    @Override
    public void onContradiction(ContradictionException cex) {
        if (!(cex.c instanceof Propagator)) return;
        Propagator<?> p = (Propagator<?>) cex.c;
        Integer pid = pIndex.get(p);
        if (pid == null) return;

        // Fraîcheur de portée : part des variables du propagateur fautif encore libres.
        int nbv = p.getNbVars(), futur = 0;
        for (int i = 0; i < nbv; i++) {
            Variable v = p.getVar(i);
            if (v instanceof IntVar && !v.isInstantiated()) futur++;
        }
        if (nbv > 0) { freshSum += (double) futur / nbv; freshCount++; }

        if (mode == Mode.ABSCON) {
            // +1 à chaque variable FUTURE (non instanciée) de la portée
            int[] sc = propScope.get(pid);
            double[] wv = cwVar.get(pid);
            for (int k = 0; k < sc.length; k++) {
                if (!vars[sc[k]].isInstantiated()) wv[k] += 1.0;
            }
        } else {
            // WDEG_2004 / HD_2004 : +1 au poids scalaire de la contrainte fautive
            cw[pid] += 1.0;
        }
    }
}
