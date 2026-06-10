package dmowcsp;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.variables.IntVar;
import org.chocosolver.solver.variables.Task;

import java.util.ArrayList;
import java.util.List;

/**
 * Modèle CP correct du job-shop pour Choco.
 *
 * Contrairement à l'ancien encodage XCSP3/ACE (disjonctions binaires + makespan
 * approximé), ce modèle utilise :
 *   - des variables start/end par opération liées par end = start + durée ;
 *   - les précédences intra-job (end[k] <= start[k+1]) ;
 *   - une contrainte globale `cumulative` de capacité 1 par machine
 *     (machine unaire = noOverlap exact) ;
 *   - un vrai makespan : cmax = max(fin du dernier op de chaque job) ;
 *   - le flowtime : somme des fins des derniers op de chaque job.
 */
public class JobShopModel {

    public final Model model;
    public final JobShopInstance inst;
    public final IntVar[][] start;   // start[j][k]
    public final IntVar[][] end;     // end[j][k]
    public final IntVar cmax;        // makespan
    public final IntVar flowtime;    // somme des temps de complétion des jobs
    public final IntVar[] flatStart; // toutes les variables start (ordre d'aplatissement)
    public final IntVar[] jobCompletion; // fin du dernier op de chaque job
    public final int[] deadlines;    // deadline par job (1.5 × durée totale, comme côté Python)
    public final int[][] flatVarOps; // flatVarOps[i] = {j, k} pour flatStart[i]

    public JobShopModel(JobShopInstance inst) {
        this.inst = inst;
        this.model = new Model("jobshop-" + inst.name);
        int horizon = inst.upperBoundMakespan();

        // Deadlines : miroir de parse_orlib (deadline_j = 1.5 × somme des durées du job j)
        this.deadlines = new int[inst.nJobs];
        for (int j = 0; j < inst.nJobs; j++) {
            int tot = 0;
            for (int[] op : inst.jobs.get(j)) tot += op[1];
            deadlines[j] = (int) (tot * 1.5);
        }

        this.start = new IntVar[inst.nJobs][];
        this.end = new IntVar[inst.nJobs][];

        List<IntVar> flat = new ArrayList<>();
        List<int[]> flatOps = new ArrayList<>();
        IntVar[] jobCompletion = new IntVar[inst.nJobs];

        // ── Variables + liaison end = start + durée ──────────────────────────
        for (int j = 0; j < inst.nJobs; j++) {
            int[][] ops = inst.jobs.get(j);
            start[j] = new IntVar[ops.length];
            end[j] = new IntVar[ops.length];
            for (int k = 0; k < ops.length; k++) {
                int dur = ops[k][1];
                start[j][k] = model.intVar("s_" + j + "_" + k, 0, horizon);
                end[j][k] = model.intVar("e_" + j + "_" + k, 0, horizon);
                // Task pose la contrainte start + dur = end
                new Task(start[j][k], dur, end[j][k]);
                flat.add(start[j][k]);
                flatOps.add(new int[]{j, k});
            }
            // Précédences intra-job
            for (int k = 0; k < ops.length - 1; k++) {
                model.arithm(end[j][k], "<=", start[j][k + 1]).post();
            }
            jobCompletion[j] = end[j][ops.length - 1];
        }

        // ── Contrainte machine : cumulative capacité 1 (noOverlap exact) ─────
        for (int m = 0; m < inst.nMachines; m++) {
            List<Task> tasks = new ArrayList<>();
            for (int j = 0; j < inst.nJobs; j++) {
                int[][] ops = inst.jobs.get(j);
                for (int k = 0; k < ops.length; k++) {
                    if (ops[k][0] == m) {
                        tasks.add(new Task(start[j][k], ops[k][1], end[j][k]));
                    }
                }
            }
            if (tasks.size() > 1) {
                Task[] tarr = tasks.toArray(new Task[0]);
                IntVar[] heights = new IntVar[tarr.length];
                for (int i = 0; i < heights.length; i++) heights[i] = model.intVar(1);
                model.cumulative(tarr, heights, model.intVar(1)).post();
            }
        }

        // ── Objectifs ────────────────────────────────────────────────────────
        this.cmax = model.intVar("cmax", 0, horizon);
        model.max(cmax, jobCompletion).post();

        this.flowtime = model.intVar("flowtime", 0, horizon * inst.nJobs);
        model.sum(jobCompletion, "=", flowtime).post();

        this.flatStart = flat.toArray(new IntVar[0]);
        this.jobCompletion = jobCompletion;
        this.flatVarOps = flatOps.toArray(new int[0][]);
    }

    /**
     * Robustesse d'une solution = slack moyen vis-à-vis des deadlines.
     * slack_j = max(0, deadline_j - completion_j). Miroir de ScheduleEvaluator Python.
     */
    public double robustness(int[] completion) {
        double sum = 0.0;
        for (int j = 0; j < inst.nJobs; j++) {
            sum += Math.max(0, deadlines[j] - completion[j]);
        }
        return inst.nJobs > 0 ? sum / inst.nJobs : 0.0;
    }
}
