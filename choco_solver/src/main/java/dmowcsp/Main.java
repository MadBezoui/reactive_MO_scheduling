package dmowcsp;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solution;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.search.strategy.strategy.AbstractStrategy;
import org.chocosolver.solver.variables.IntVar;

import java.util.HashMap;
import java.util.Map;

/**
 * Point d'entrée CLI du runner Choco.
 *
 * Usage :
 *   java -jar dmo-choco.jar --instance benchmarks_real/orlib/ft06.txt \
 *        --heuristic wdeg --objective cmax --timeout 30
 *
 * Sortie : une ligne JSON sur stdout (consommable par le pipeline Python).
 *
 * Étape J1 (courante) : modèle job-shop + minimisation du makespan, pour valider
 * le modèle (ft06 → optimum C_max = 55). Le multi-objectif (ε-constraint) et
 * MO-DYN-HD-CACD viendront ensuite.
 */
public class Main {

    public static void main(String[] args) {
        Map<String, String> opt = parseArgs(args);
        String instancePath = opt.get("instance");
        String heuristic = opt.getOrDefault("heuristic", "wdeg");
        int timeout = Integer.parseInt(opt.getOrDefault("timeout", "30"));
        String objective = opt.getOrDefault("objective", "cmax"); // "cmax" | "pareto"
        int points = Integer.parseInt(opt.getOrDefault("points", "5"));
        String perturb = opt.get("perturb"); // CSV des perturbations (pour mo_dyn_hd_cacd)
        String weights = opt.get("weights"); // Poids personnalisés pour mo_dyn_hd_cacd

        if (instancePath == null) {
            System.err.println("Erreur : --instance <chemin> requis.");
            System.err.println("Heuristiques supportées : " + Heuristics.SUPPORTED);
            System.exit(2);
        }

        try {
            // ── RCPSP (.sm) : mono-objectif makespan ────────────────────────
            if (instancePath.toLowerCase().endsWith(".sm")) {
                runRcpsp(instancePath, heuristic, timeout);
                return;
            }

            JobShopInstance inst = OrLibParser.parse(instancePath);

            // ── Mode multi-objectif : front de Pareto par ε-constraint ──────
            if (objective.equalsIgnoreCase("pareto")) {
                ParetoSolver.Result r =
                        new ParetoSolver(inst, heuristic, points, timeout, perturb, weights).solve();
                System.out.println(frontJson(inst.name, heuristic, r));
                return;
            }

            // ── Mode mono-objectif : minimisation du makespan ───────────────
            JobShopModel jm = new JobShopModel(inst);
            Model model = jm.model;

            AbstractStrategy<IntVar> search = Heuristics.strategyFor(heuristic, jm, perturb, weights);
            Solver solver = model.getSolver();
            solver.setSearch(search);
            solver.limitTime(timeout + "s");

            Solution sol = solver.findOptimalSolution(jm.cmax, Model.MINIMIZE);

            boolean interrupted = solver.isStopCriterionMet();
            String status;
            Integer cmax = null, flowtime = null;
            double rob = 0.0;
            if (sol != null) {
                cmax = sol.getIntVal(jm.cmax);
                flowtime = sol.getIntVal(jm.flowtime);
                // Robustesse réelle dérivée du schedule (slack moyen vs deadlines),
                // exactement comme ScheduleEvaluator côté Python.
                int[] completion = new int[inst.nJobs];
                for (int j = 0; j < inst.nJobs; j++) {
                    completion[j] = sol.getIntVal(jm.jobCompletion[j]);
                }
                rob = jm.robustness(completion);
                status = interrupted ? "FEASIBLE" : "OPTIMUM";
            } else {
                status = interrupted ? "UNKNOWN" : "UNSAT";
            }

            long nodes = solver.getMeasures().getNodeCount();
            float cpu = solver.getMeasures().getTimeCount();

            System.out.println(monoJson(inst.name, heuristic, status, cmax, flowtime, rob, nodes, cpu));
        } catch (Exception e) {
            System.out.println("{\"status\":\"ERR\",\"error\":\""
                    + e.getMessage().replace("\"", "'") + "\"}");
            System.exit(1);
        }
    }

    private static void runRcpsp(String path, String heuristic, int timeout) throws Exception {
        RCPSPInstance inst = PsplibParser.parse(path);
        RCPSPModel rm = new RCPSPModel(inst);
        Solver solver = rm.model.getSolver();
        solver.setSearch(Heuristics.forName(heuristic, rm.flatStart));
        solver.limitTime(timeout + "s");

        Solution sol = solver.findOptimalSolution(rm.cmax, Model.MINIMIZE);
        boolean interrupted = solver.isStopCriterionMet();
        String status;
        Integer cmax = null, flow = null;
        double rob = 0.0;
        if (sol != null) {
            cmax = sol.getIntVal(rm.cmax);
            flow = sol.getIntVal(rm.flowtime);
            // Robustesse proxy = cmax × 0.1 — identique au proxy NSGA/Python en RCPSP
            // (pas de deadlines explicites). Rend l'axe robustesse comparable.
            rob = rm.robustness(cmax);
            status = interrupted ? "FEASIBLE" : "OPTIMUM";
        } else {
            status = interrupted ? "UNKNOWN" : "UNSAT";
        }
        long nodes = solver.getMeasures().getNodeCount();
        float cpu = solver.getMeasures().getTimeCount();
        System.out.println(monoJson(inst.name, heuristic, status, cmax, flow, rob, nodes, cpu));
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> m = new HashMap<>();
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].startsWith("--")) {
                m.put(args[i].substring(2), args[i + 1]);
            }
        }
        return m;
    }

    private static String frontJson(String inst, String heur, ParetoSolver.Result r) {
        StringBuilder sb = new StringBuilder("{");
        sb.append("\"instance\":\"").append(inst).append("\",");
        sb.append("\"heuristic\":\"").append(heur).append("\",");
        sb.append("\"status\":\"").append(r.status).append("\",");
        sb.append("\"front_size\":").append(r.front.size()).append(",");
        sb.append("\"nodes\":").append(r.nodes).append(",");
        sb.append("\"cpu\":").append(String.format(java.util.Locale.US, "%.4f", r.cpu)).append(",");
        sb.append("\"front\":[");
        for (int i = 0; i < r.front.size(); i++) {
            double[] p = r.front.get(i);
            sb.append(String.format(java.util.Locale.US, "[%.1f,%.1f,%.4f]", p[0], p[1], p[2]));
            if (i < r.front.size() - 1) sb.append(",");
        }
        sb.append("]");
        // Solution représentante (Cmax min) : ops + temps de début, pour la stabilité.
        if (r.repStarts != null && r.repOps != null) {
            sb.append(",\"rep_ops\":[");
            for (int i = 0; i < r.repOps.length; i++) {
                sb.append("[").append(r.repOps[i][0]).append(",").append(r.repOps[i][1]).append("]");
                if (i < r.repOps.length - 1) sb.append(",");
            }
            sb.append("],\"rep_starts\":[");
            for (int i = 0; i < r.repStarts.length; i++) {
                sb.append(r.repStarts[i]);
                if (i < r.repStarts.length - 1) sb.append(",");
            }
            sb.append("]");
        }
        sb.append("}");
        return sb.toString();
    }

    /**
     * Sortie mono-objectif homogénéisée : émet aussi un front mono-point
     * [cmax, flowtime, -robustesse] pour que le 3e objectif (robustesse) soit
     * toujours renseigné — fini le 0.0 reconstruit côté Python.
     */
    private static String monoJson(String inst, String heur, String status,
                                   Integer cmax, Integer flow, double rob,
                                   long nodes, float cpu) {
        StringBuilder sb = new StringBuilder("{");
        sb.append("\"instance\":\"").append(inst).append("\",");
        sb.append("\"heuristic\":\"").append(heur).append("\",");
        sb.append("\"status\":\"").append(status).append("\",");
        sb.append("\"cmax\":").append(cmax == null ? "null" : cmax).append(",");
        sb.append("\"flowtime\":").append(flow == null ? "null" : flow).append(",");
        if (cmax != null) {
            double f = (flow == null) ? (double) cmax : (double) flow;
            sb.append("\"front_size\":1,");
            sb.append("\"front\":[")
              .append(String.format(java.util.Locale.US, "[%.1f,%.1f,%.4f]",
                      (double) cmax, f, -rob))
              .append("],");
        } else {
            sb.append("\"front_size\":0,\"front\":[],");
        }
        sb.append("\"nodes\":").append(nodes).append(",");
        sb.append("\"cpu\":").append(String.format(java.util.Locale.US, "%.4f", cpu));
        sb.append("}");
        return sb.toString();
    }
}
