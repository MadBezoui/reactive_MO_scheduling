package dmowcsp;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solution;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.search.strategy.selectors.values.IntValueSelector;
import org.chocosolver.solver.search.strategy.strategy.IntStrategy;
import org.chocosolver.solver.variables.IntVar;

import java.util.HashMap;
import java.util.Map;

/**
 * ExperimentWeighting — banc d'essai CONTRÔLÉ pour le papier (Path A).
 *
 * Compare trois schémas de pondération des conflits (dom/wdeg2004, AbsCon par
 * variable, HD-augmenté) sur EXACTEMENT le même modèle, le même balayage
 * ε-contraint et le même budget. Émet, par instance, des statistiques d'arbre
 * de recherche SENSIBLES au branchement — nœuds, échecs, backtracks, et taux de
 * clôture (preuve d'optimalité dans le budget) — plus le temps CPU.
 *
 * Le but n'est PAS de comparer la qualité de front sous budget (où les schémas
 * peuvent être à égalité) mais d'exposer toute différence de comportement de
 * recherche, comme demandé par Wattez et al. (2019), §IV.
 *
 * Usage :
 *   java -cp dmo-choco.jar dmowcsp.ExperimentWeighting \
 *        --instance benchmarks_real/orlib/la06.txt \
 *        --variant wdeg2004|abscon|hd2004 --timeout 30 --points 5
 *
 * Sortie : une ligne JSON sur stdout.
 */
public class ExperimentWeighting {

    private final JobShopInstance inst;
    private final WeightingSelector.Mode mode;
    private final int nPoints;
    private final int perSolveTimeout;

    // Accumulateurs (sur l'ensemble des solves du balayage ε).
    long nodes = 0, fails = 0, backtracks = 0, totalSolves = 0, closedSolves = 0;
    double cpu = 0.0;
    int bestCmax = Integer.MAX_VALUE;
    double freshSum = 0.0; long freshCount = 0;   // fraîcheur de portée aux échecs (Proposition 1)

    public ExperimentWeighting(JobShopInstance inst, WeightingSelector.Mode mode, int nPoints, int perSolveTimeout) {
        this.inst = inst; this.mode = mode; this.nPoints = nPoints; this.perSolveTimeout = perSolveTimeout;
    }

    private static WeightingSelector.Mode parseMode(String v) {
        switch (v.toLowerCase()) {
            case "abscon":   return WeightingSelector.Mode.ABSCON;
            case "hd2004":
            case "hd":       return WeightingSelector.Mode.HD_2004;
            case "wdeg2004":
            case "wdeg":
            default:         return WeightingSelector.Mode.WDEG_2004;
        }
    }

    /** Un sous-problème ε : minimise Cmax sous flowtime ≤ epsFlow (epsFlow<0 = libre). */
    private double[] solveOne(int epsFlow, boolean minimizeFlow) {
        JobShopModel jm = new JobShopModel(inst);
        Model model = jm.model;
        if (epsFlow >= 0) model.arithm(jm.flowtime, "<=", epsFlow).post();

        WeightingSelector sel = new WeightingSelector(jm, mode);
        IntValueSelector valSel = var -> var.getLB();          // earliest-start, commun à tous les modes
        Solver solver = model.getSolver();
        solver.plugMonitor(sel);
        solver.setSearch(new IntStrategy(jm.flatStart, sel, valSel));
        solver.limitTime(perSolveTimeout + "s");

        IntVar objective = minimizeFlow ? jm.flowtime : jm.cmax;
        Solution sol = solver.findOptimalSolution(objective, Model.MINIMIZE);

        // Stats de recherche sensibles au branchement.
        nodes += solver.getMeasures().getNodeCount();
        fails += solver.getMeasures().getFailCount();
        backtracks += solver.getMeasures().getBackTrackCount();
        cpu += solver.getMeasures().getTimeCount();
        totalSolves++;
        if (!solver.isStopCriterionMet()) closedSolves++;       // clôture = preuve dans le budget
        freshSum += sel.getFreshSum(); freshCount += sel.getFreshCount();

        if (sol == null) return null;
        int cmax = sol.getIntVal(jm.cmax);
        int flow = sol.getIntVal(jm.flowtime);
        if (cmax < bestCmax) bestCmax = cmax;
        return new double[]{cmax, flow};
    }

    public void run() {
        double[] atMinFlow = solveOne(-1, true);
        double[] atMinCmax = solveOne(-1, false);
        if (atMinFlow != null && atMinCmax != null) {
            int flowLo = (int) Math.round(atMinFlow[1]);
            int flowHi = (int) Math.round(atMinCmax[1]);
            if (flowHi < flowLo) { int t = flowLo; flowLo = flowHi; flowHi = t; }
            for (int i = 0; i < nPoints; i++) {
                int eps = nPoints == 1 ? flowHi
                        : flowLo + (int) Math.round((double) (flowHi - flowLo) * i / (nPoints - 1));
                solveOne(eps, false);
            }
        }
    }

    private String json(String variant) {
        double closureRate = totalSolves == 0 ? 0.0 : (double) closedSolves / totalSolves;
        double freshness = freshCount == 0 ? Double.NaN : freshSum / freshCount;  // mean b(c)
        double epsilon = Double.isNaN(freshness) ? Double.NaN : 1.0 - freshness;  // Proposition 1
        return String.format(java.util.Locale.US,
            "{\"instance\":\"%s\",\"variant\":\"%s\",\"points\":%d,"
            + "\"total_solves\":%d,\"closed_solves\":%d,\"closure_rate\":%.4f,"
            + "\"nodes\":%d,\"fails\":%d,\"backtracks\":%d,\"cpu\":%.4f,"
            + "\"scope_freshness\":%.4f,\"epsilon\":%.4f,\"best_cmax\":%d}",
            inst.name, variant, nPoints, totalSolves, closedSolves, closureRate,
            nodes, fails, backtracks, cpu, freshness, epsilon,
            (bestCmax == Integer.MAX_VALUE ? -1 : bestCmax));
    }

    public static void main(String[] args) {
        Map<String, String> opt = new HashMap<>();
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].startsWith("--")) opt.put(args[i].substring(2), args[i + 1]);
        }
        String instancePath = opt.get("instance");
        String variant = opt.getOrDefault("variant", "wdeg2004");
        int timeout = Integer.parseInt(opt.getOrDefault("timeout", "30"));
        int points = Integer.parseInt(opt.getOrDefault("points", "5"));
        if (instancePath == null) {
            System.err.println("Usage: --instance <path> --variant wdeg2004|abscon|hd2004 [--timeout 30] [--points 5]");
            System.exit(2);
        }
        try {
            JobShopInstance inst = OrLibParser.parse(instancePath);
            ExperimentWeighting exp = new ExperimentWeighting(inst, parseMode(variant), points, timeout);
            exp.run();
            System.out.println(exp.json(variant));
        } catch (Exception e) {
            System.out.println("{\"status\":\"ERR\",\"error\":\"" + e.getMessage().replace("\"", "'") + "\"}");
            System.exit(1);
        }
    }
}
