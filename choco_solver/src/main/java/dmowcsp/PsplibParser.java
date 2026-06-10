package dmowcsp;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Parser du format PSPLIB .sm (RCPSP). Miroir de parse_psplib() (main_real.py).
 *
 * Sections lues : dimensions, PRECEDENCE RELATIONS, REQUESTS/DURATIONS,
 * RESOURCEAVAILABILITIES. Jobs convertis en 0-based.
 */
public final class PsplibParser {

    private PsplibParser() {}

    public static RCPSPInstance parse(String path) throws IOException {
        String fname = Paths.get(path).getFileName().toString();
        String name = fname.contains(".") ? fname.substring(0, fname.lastIndexOf('.')) : fname;
        List<String> lines = Files.readAllLines(Paths.get(path));

        int nTotal = 0, horizon = 0, nRes = 4;
        for (String l : lines) {
            String t = l.trim();
            if (t.startsWith("jobs (incl")) nTotal = lastInt(t);
            else if (t.startsWith("horizon")) horizon = lastInt(t);
            else if (t.startsWith("- renewable")) nRes = lastInt(t);
        }

        int[][] succs = new int[Math.max(nTotal, 1)][];
        int[] durs = new int[Math.max(nTotal, 1)];
        int[][] reqs = new int[Math.max(nTotal, 1)][];
        int[] capacities = null;

        // Localiser les sections par leurs en-têtes
        int iPrec = indexOf(lines, "PRECEDENCE RELATIONS:");
        int iReq = indexOf(lines, "REQUESTS/DURATIONS:");
        int iCap = indexOf(lines, "RESOURCEAVAILABILITIES:");

        // ── Précédences ─────────────────────────────────────────────────────
        for (int i = iPrec + 1; i < iReq && i < lines.size(); i++) {
            String[] p = lines.get(i).trim().split("\\s+");
            if (p.length < 3 || !p[0].matches("\\d+")) continue;
            int jobId = Integer.parseInt(p[0]) - 1;
            int nSucc = Integer.parseInt(p[2]);
            int[] sc = new int[nSucc];
            for (int k = 0; k < nSucc && (3 + k) < p.length; k++) {
                sc[k] = Integer.parseInt(p[3 + k]) - 1;
            }
            if (jobId >= 0 && jobId < succs.length) succs[jobId] = sc;
        }

        // ── Durées + demandes ───────────────────────────────────────────────
        for (int i = iReq + 1; i < iCap && i < lines.size(); i++) {
            String[] p = lines.get(i).trim().split("\\s+");
            if (p.length < 3 || !p[0].matches("\\d+")) continue;
            int jobId = Integer.parseInt(p[0]) - 1;
            int dur = Integer.parseInt(p[2]);
            int[] r = new int[nRes];
            for (int k = 0; k < nRes && (3 + k) < p.length; k++) {
                r[k] = Integer.parseInt(p[3 + k]);
            }
            if (jobId >= 0 && jobId < durs.length) { durs[jobId] = dur; reqs[jobId] = r; }
        }

        // ── Capacités ───────────────────────────────────────────────────────
        for (int i = iCap + 1; i < lines.size(); i++) {
            String[] p = lines.get(i).trim().split("\\s+");
            if (p.length >= nRes && allInts(p)) {
                capacities = new int[nRes];
                for (int k = 0; k < nRes; k++) capacities[k] = Integer.parseInt(p[k]);
                break;
            }
        }
        if (capacities == null) {
            capacities = new int[nRes];
            java.util.Arrays.fill(capacities, 8);
        }

        List<RCPSPInstance.Job> jobs = new ArrayList<>();
        for (int j = 0; j < nTotal; j++) {
            int[] r = reqs[j] != null ? reqs[j] : new int[nRes];
            int[] sc = succs[j] != null ? succs[j] : new int[0];
            jobs.add(new RCPSPInstance.Job(durs[j], r, sc));
        }

        return new RCPSPInstance(name, nTotal, nRes, horizon, jobs, capacities);
    }

    private static int indexOf(List<String> lines, String header) {
        for (int i = 0; i < lines.size(); i++) {
            if (lines.get(i).trim().startsWith(header)) return i;
        }
        return -1;
    }

    private static int lastInt(String line) {
        String[] p = line.trim().split("\\s+");
        for (int i = p.length - 1; i >= 0; i--) {
            if (p[i].matches("\\d+")) return Integer.parseInt(p[i]);
        }
        return 0;
    }

    private static boolean allInts(String[] p) {
        for (String s : p) if (!s.matches("\\d+")) return false;
        return true;
    }
}
