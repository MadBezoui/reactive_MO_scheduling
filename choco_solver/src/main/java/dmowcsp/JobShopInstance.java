package dmowcsp;

import java.util.List;

/**
 * Instance job-shop au format OR-Library.
 *
 * jobs.get(j) = liste des opérations du job j, chaque opération = int[]{machine, durée}.
 */
public class JobShopInstance {
    public final String name;
    public final int nJobs;
    public final int nMachines;
    public final List<int[][]> jobs;   // jobs.get(j)[k] = {machine, duration}

    public JobShopInstance(String name, int nJobs, int nMachines, List<int[][]> jobs) {
        this.name = name;
        this.nJobs = nJobs;
        this.nMachines = nMachines;
        this.jobs = jobs;
    }

    /** Borne supérieure naïve du makespan = somme de toutes les durées. */
    public int upperBoundMakespan() {
        int total = 0;
        for (int[][] job : jobs)
            for (int[] op : job)
                total += op[1];
        return total;
    }
}
