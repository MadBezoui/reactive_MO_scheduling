package dmowcsp;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.variables.IntVar;
import org.chocosolver.solver.variables.Task;

import java.util.ArrayList;
import java.util.List;

/**
 * Modèle CP du RCPSP pour Choco.
 *
 *   - variables start/end par job, liées par end = start + durée ;
 *   - précédences (start[succ] >= end[job]) ;
 *   - une contrainte globale `cumulative` par ressource renouvelable ;
 *   - makespan = max des fins des jobs réels (durée > 0) ;
 *   - flowtime = somme des fins des jobs réels (miroir de evaluate_rcpsp Python).
 *
 * Horizon = somme des durées (borne supérieure toujours valide, robuste aux
 * fichiers dont l'horizon déclaré serait trop serré).
 */
public class RCPSPModel {

    public final Model model;
    public final RCPSPInstance inst;
    public final IntVar[] start;
    public final IntVar[] end;
    public final IntVar cmax;
    public final IntVar flowtime;
    public final IntVar[] flatStart;        // variables de décision (jobs réels)
    public final IntVar[] jobCompletion;    // fins des jobs réels
    public final double robustnessFactor = 0.1; // proxy (pas de deadlines explicites)

    public RCPSPModel(RCPSPInstance inst) {
        this.inst = inst;
        this.model = new Model("rcpsp-" + inst.name);
        int n = inst.nJobs;
        int horizon = inst.upperBoundMakespan();

        this.start = new IntVar[n];
        this.end = new IntVar[n];
        List<IntVar> decision = new ArrayList<>();
        List<IntVar> realEnds = new ArrayList<>();

        // ── Variables + liaison end = start + durée ─────────────────────────
        for (int j = 0; j < n; j++) {
            int dur = inst.jobs.get(j).dur;
            start[j] = model.intVar("s_" + j, 0, horizon);
            end[j] = model.intVar("e_" + j, 0, horizon);
            new Task(start[j], dur, end[j]); // pose start + dur = end
            if (dur > 0) {
                decision.add(start[j]);
                realEnds.add(end[j]);
            }
        }

        // ── Précédences ──────────────────────────────────────────────────────
        for (int j = 0; j < n; j++) {
            for (int s : inst.jobs.get(j).succs) {
                if (s >= 0 && s < n) {
                    model.arithm(start[s], ">=", end[j]).post();
                }
            }
        }

        // ── Cumulative par ressource renouvelable ───────────────────────────
        for (int r = 0; r < inst.nResources; r++) {
            List<Task> tasks = new ArrayList<>();
            List<Integer> heights = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                RCPSPInstance.Job job = inst.jobs.get(j);
                if (job.dur > 0 && r < job.res.length && job.res[r] > 0) {
                    tasks.add(new Task(start[j], job.dur, end[j]));
                    heights.add(job.res[r]);
                }
            }
            if (!tasks.isEmpty()) {
                Task[] tarr = tasks.toArray(new Task[0]);
                IntVar[] h = new IntVar[heights.size()];
                for (int k = 0; k < h.length; k++) h[k] = model.intVar(heights.get(k));
                model.cumulative(tarr, h, model.intVar(inst.capacities[r])).post();
            }
        }

        // ── Objectifs (sur les jobs réels) ──────────────────────────────────
        IntVar[] ends = realEnds.toArray(new IntVar[0]);
        this.cmax = model.intVar("cmax", 0, horizon);
        if (ends.length > 0) {
            model.max(cmax, ends).post();
        } else {
            model.arithm(cmax, "=", 0).post();
        }
        this.flowtime = model.intVar("flowtime", 0, horizon * Math.max(1, ends.length));
        if (ends.length > 0) {
            model.sum(ends, "=", flowtime).post();
        } else {
            model.arithm(flowtime, "=", 0).post();
        }

        this.flatStart = decision.isEmpty()
                ? new IntVar[]{cmax} : decision.toArray(new IntVar[0]);
        this.jobCompletion = ends;
    }

    /** Robustesse proxy = makespan × 0.1 (pas de deadlines en RCPSP standard). */
    public double robustness(int cmaxValue) {
        return cmaxValue * robustnessFactor;
    }
}
