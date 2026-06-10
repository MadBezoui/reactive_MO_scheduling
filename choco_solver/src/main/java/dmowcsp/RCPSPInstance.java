package dmowcsp;

import java.util.List;

/**
 * Instance RCPSP au format PSPLIB (.sm).
 *
 * jobs.get(j) = {dur, res[], succs[]} ; indices de jobs 0-based.
 * job 0 = supersource, job (nJobs-1) = supersink (durées nulles).
 */
public class RCPSPInstance {

    public static class Job {
        public final int dur;
        public final int[] res;     // demande par ressource renouvelable
        public final int[] succs;   // successeurs (0-based)
        public Job(int dur, int[] res, int[] succs) {
            this.dur = dur; this.res = res; this.succs = succs;
        }
    }

    public final String name;
    public final int nJobs;
    public final int nResources;
    public final int horizon;
    public final List<Job> jobs;
    public final int[] capacities;

    public RCPSPInstance(String name, int nJobs, int nResources, int horizon,
                         List<Job> jobs, int[] capacities) {
        this.name = name;
        this.nJobs = nJobs;
        this.nResources = nResources;
        this.horizon = horizon;
        this.jobs = jobs;
        this.capacities = capacities;
    }

    /** Borne supérieure sûre du makespan = somme de toutes les durées. */
    public int upperBoundMakespan() {
        int total = 0;
        for (Job j : jobs) total += j.dur;
        return total;
    }
}
