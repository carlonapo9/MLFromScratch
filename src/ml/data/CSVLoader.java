package ml.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class CSVLoader {

    // Manual loader (kept for explicit control if needed)
    public static Dataset load(String path, boolean hasHeader, int labelIndex) throws Exception {
        List<double[]> features = new ArrayList<>();
        List<String> labelsRaw = new ArrayList<>();

        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;

        if (hasHeader) br.readLine(); // skip header

        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;

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

        br.close();

        double[] y = encodeLabels(labelsRaw);
        double[][] X = features.toArray(new double[0][]);

        return new Dataset(X, y);
    }

    // Auto: detect header, separator, label column (last), numeric vs string labels
    public static Dataset loadAuto(String path) throws Exception {
        List<String[]> rows = new ArrayList<>();

        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;
        String sep = null;

        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;

            if (sep == null) {
                sep = line.contains(";") ? ";" : ",";
            }

            rows.add(line.split(sep));
        }

        br.close();

        if (rows.isEmpty()) {
            throw new RuntimeException("Empty CSV: " + path);
        }

        String[] first = rows.get(0);

        // Detect header
        boolean hasHeader = !allNumeric(first);

        int start = hasHeader ? 1 : 0;
        int cols = first.length;

        // Detect label column:
        // If a column contains ANY non-numeric values → it's the label column.
        int labelIndex = -1;

        for (int col = 0; col < cols; col++) {
            boolean numeric = true;
            for (int i = start; i < rows.size(); i++) {
                try {
                    Double.parseDouble(rows.get(i)[col]);
                } catch (NumberFormatException e) {
                    numeric = false;
                    break;
                }
            }
            if (!numeric) {
                labelIndex = col;
                break;
            }
        }

        // If all columns are numeric → assume last column is label
        if (labelIndex == -1) {
            labelIndex = cols - 1;
        }

        List<double[]> features = new ArrayList<>();
        List<String> labelsRaw = new ArrayList<>();

        for (int i = start; i < rows.size(); i++) {
            String[] parts = rows.get(i);

            double[] row = new double[cols - 1];
            int idx = 0;

            for (int j = 0; j < cols; j++) {
                if (j == labelIndex) {
                    labelsRaw.add(parts[j]);
                } else {
                    row[idx++] = Double.parseDouble(parts[j]);
                }
            }

            features.add(row);
        }

        double[] y = encodeLabels(labelsRaw);
        double[][] X = features.toArray(new double[0][]);

        return new Dataset(X, y);
    }


    private static boolean allNumeric(String[] tokens) {
        for (String t : tokens) {
            try {
                Double.parseDouble(t);
            } catch (NumberFormatException e) {
                return false;
            }
        }
        return true;
    }

    private static double[] encodeLabels(List<String> labelsRaw) {
        Map<String, Double> map = new HashMap<>();
        double next = 0.0;

        double[] y = new double[labelsRaw.size()];
        for (int i = 0; i < labelsRaw.size(); i++) {
            String s = labelsRaw.get(i);
            if (!map.containsKey(s)) {
                map.put(s, next++);
            }
            y[i] = map.get(s);
        }
        return y;
    }
}
