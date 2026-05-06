package ml.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class CSVLoader {

    public static Dataset load(String path, boolean hasHeader, int labelIndex) throws Exception {
        List<double[]> features = new ArrayList<>();
        List<String> labelsRaw = new ArrayList<>();

        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;

        if (hasHeader) br.readLine(); // skip header

        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;

            // detect separator
            String sep = line.contains(";") ? ";" : ",";

            String[] parts = line.split(sep);

            double[] row = new double[parts.length - 1];
            int idx = 0;

            for (int i = 0; i < parts.length; i++) {
                if (i == labelIndex) {
                    labelsRaw.add(parts[i]);
                } else {
                    row[idx++] = Double.parseDouble(parts[i]);
                }
            }

            features.add(row);
        }

        // map labels to numeric
        Map<String, Double> map = new HashMap<>();
        double next = 0;

        double[] y = new double[labelsRaw.size()];
        for (int i = 0; i < labelsRaw.size(); i++) {
            String s = labelsRaw.get(i);
            if (!map.containsKey(s)) map.put(s, next++);
            y[i] = map.get(s);
        }

        double[][] X = features.toArray(new double[0][]);

        return new Dataset(X, y);
    }
}
