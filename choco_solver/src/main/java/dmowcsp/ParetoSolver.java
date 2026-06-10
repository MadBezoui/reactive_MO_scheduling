package dmowcsp;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solution;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.variables.IntVar;

import java.util.ArrayList;
import java.util.List;

/**
 * Approximation du front de Pareto par méthode ε-constraint.
 *
 * Objectifs : (Cmax, flowtime) minimisés via CP ; la robustesse est dérivée du
 * schedule obtenu (slack moyen), exactement comme côté Python.
 *
 * Schéma :
 *   1. min flowtime  → borne basse du flowtime (et son Cmax) ;
 *   2. min Cmax      → borne basse du Cmax (et son flowtime) ;
 *   3. pour ε ∈ linspace(flow_min, flow_max) : min Cmax s.c. flowtime ≤ ε ;
 *   4. filtrage des points dominés.
 *
 * Chaque sous-problème reconstruit un modèle frais (les contraintes Choco ne se
 * retirent pas aisément), avec l'heuristique de branchement demandée.
 */
public class ParetoSolver {

    public static class Result {
        public final List<double[]> front = new ArrayList<>(); // [cmax, flowtime, -robustness]
        public long nodes = 0;
        public float cpu = 0f;
        public String status = "UNKNOWN";
        // Solution représentative (Cmax minimal) : temps de début par opération,
        // pour la métrique de stabilité inter-snapshots.
        public int repCmax = Integer.MAX_VALUE;
        public int[] repStarts = null;     // repStarts[i] = début de l'op flatVarOps[i]
        public int[][] repOps = null;      // repOps[i] = {j, k}
    }

    private final JobShopInstance inst;
    private final String heuristic;
    private final String perturbations;
    private final String weights;
    private final int nPoints;
    private final int perSolveTimeout; // secondes par sous-problème

    public ParetoSolver(JobShopInstance inst, String heuristic, int nPoints,
                        int totalTimeout, String perturbations, String weights) {
        this.inst = inst;
        this.heuristic = heuristic;
        this.perturbations = perturbations;
        this.weights = weights;
        this.nPoints = Math.max(2, nPoints);
        this.perSolveTimeout = Math.max(1, totalTimeout / (this.nPoints + 2));
    }

    /** Un sous-problème : minimise Cmax, éventuellement sous flowtime ≤ epsFlow (epsFlow<0 = sans). */
    private double[] solveOne(int epsFlow, boolean minimizeFlow, Result acc) {
        JobShopModel jm = new JobShopModel(inst);
        Model model = jm.model;
        if (epsFlow >= 0) {
            model.arithm(jm.flowtime, "<=", epsFlow).post();
        }
        Solver solver = model.getSolver();
        solver.setSearch(Heuristics.strategyFor(heuristic, jm, perturbations, weights));
        solver.limitTime(perSolveTimeout + "s");

        IntVar objective = minimizeFlow ? jm.flowtime : jm.cmax;
        Solution sol = solver.findOptimalSolution(objective, Model.MINIMIZE);

        acc.nodes += solver.getMeasures().getNodeCount();
        acc.cpu += solver.getMeasures().getTimeCount();
        if (sol == null) return null;

        int cmax = sol.getIntVal(jm.cmax);
        int flow = sol.getIntVal(jm.flowtime);
        int[] completion = new int[inst.nJobs];
        for (int j = 0; j < inst.nJobs; j++) {
            completion[j] = sol.getIntVal(jm.jobCompletion[j]);
        }
        double rob = jm.robustness(completion);

        // Mémoriser la solution de plus petit Cmax comme représentante.
        if (cmax < acc.repCmax) {
            acc.repCmax = cmax;
            int m = jm.flatStart.length;
            int[] starts = new int[m];
            for (int i = 0; i < m; i++) starts[i] = sol.getIntVal(jm.flatStart[i]);
            acc.repStarts = starts;
            acc.repOps = jm.flatVarOps;
        }
        return new double[]{cmax, flow, -rob};
    }

    public Result solve() {
        Result res = new Result();
        List<double[]> raw = new ArrayList<>();

        // Bornes du flowtime
        double[] atMinFlow = solveOne(-1, true, res);   // min flowtime
        double[] atMinCmax = solveOne(-1, false, res);  // min cmax
        if (atMinFlow != null) raw.add(atMinFlow);
        if (atMinCmax != null) raw.add(atMinCmax);

        if (atMinFlow != null && atMinCmax != null) {
            int flowLo = (int) Math.round(atMinFlow[1]);
            int flowHi = (int) Math.round(atMinCmax[1]);
            if (flowHi < flowLo) { int t = flowLo; flowLo = flowHi; flowHi = t; }

            // Balayage ε sur le flowtime
            for (int i = 0; i < nPoints; i++) {
                int eps = nPoints == 1 ? flowHi
                        : flowLo + (int) Math.round((double) (flowHi - flowLo) * i / (nPoints - 1));
                double[] pt = solveOne(eps, false, res);
                if (pt != null) raw.add(pt);
            }
        }

        res.front.addAll(nonDominated(raw));
        res.status = res.front.isEmpty() ? "UNKNOWN" : "SAT";
        return res;
    }

    /** Filtre les points dominés (minimisation sur les 3 coordonnées). */
    static List<double[]> nonDominated(List<double[]> pts) {
        List<double[]> nd = new ArrayList<>();
        for (int i = 0; i < pts.size(); i++) {
            double[] a = pts.get(i);
            boolean dominated = false;
            for (int j = 0; j < pts.size(); j++) {
                if (i == j) continue;
                double[] b = pts.get(j);
                boolean le = b[0] <= a[0] && b[1] <= a[1] && b[2] <= a[2];
                boolean lt = b[0] < a[0] || b[1] < a[1] || b[2] < a[2];
                if (le && lt) { dominated = true; break; }
            }
            if (!dominated && !containsEqual(nd, a)) nd.add(a);
        }
        return nd;
    }

    private static boolean containsEqual(List<double[]> list, double[] p) {
        for (double[] q : list) {
            if (q[0] == p[0] && q[1] == p[1] && q[2] == p[2]) return true;
        }
        return false;
    }
}
