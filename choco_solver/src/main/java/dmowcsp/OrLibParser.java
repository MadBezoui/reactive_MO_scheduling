package dmowcsp;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Parser du format OR-Library job-shop.
 *
 * Format attendu :
 *   n_jobs n_machines
 *   machine_0 dur_0 machine_1 dur_1 ...   (une ligne par job)
 *
 * Miroir exact de parse_orlib() côté Python (main_real.py).
 */
public final class OrLibParser {

    private OrLibParser() {}

    public static JobShopInstance parse(String path) throws IOException {
        String fname = Paths.get(path).getFileName().toString();
        String name = fname.contains(".") ? fname.substring(0, fname.lastIndexOf('.')) : fname;

        List<String> rawLines = Files.readAllLines(Paths.get(path));
        List<String> lines = new ArrayList<>();
        for (String l : rawLines) {
            String t = l.trim();
            if (!t.isEmpty() && !t.startsWith("#")) lines.add(t);
        }
        if (lines.isEmpty())
            throw new IOException("Instance vide : " + path);

        // Certaines instances OR-Library débutent par une ligne de description
        // (ex. « Lawrence 30x10 instance… », « Storer, Wu… »). On saute jusqu'à
        // la première ligne formée d'exactement 2 entiers = « n_jobs n_machines ».
        // Miroir exact de parse_orlib() côté Python.
        int startIdx = 0;
        for (int i = 0; i < lines.size(); i++) {
            String[] p = lines.get(i).split("\\s+");
            if (p.length == 2 && p[0].matches("\\d+") && p[1].matches("\\d+")) {
                startIdx = i;
                break;
            }
        }

        String[] dims = lines.get(startIdx).split("\\s+");
        int nJobs = Integer.parseInt(dims[0]);
        int nMachines = Integer.parseInt(dims[1]);

        List<int[][]> jobs = new ArrayList<>();
        for (int i = startIdx + 1; i <= startIdx + nJobs && i < lines.size(); i++) {
            String[] tok = lines.get(i).split("\\s+");
            int nOps = tok.length / 2;
            int[][] ops = new int[nOps][2];
            for (int k = 0; k < nOps; k++) {
                ops[k][0] = Integer.parseInt(tok[2 * k]);     // machine
                ops[k][1] = Integer.parseInt(tok[2 * k + 1]); // durée
            }
            jobs.add(ops);
        }

        return new JobShopInstance(name, nJobs, nMachines, jobs);
    }
}
