package ml.algorithms;

import java.util.HashMap;
import java.util.Map;

public class KNNClassifier extends Algorithm {

    private double[][] trainX;
    private double[] trainY;
    private int k = 3;

    public KNNClassifier(int k) {
        this.k = k;
    }

    @Override
    public void fit(double[][] X, double[] y) {
        this.trainX = X;
        this.trainY = y;
    }

    @Override
    public double predict(double[] x) {
        // compute distances
        double[] distances = new double[trainX.length];
        for (int i = 0; i < trainX.length; i++) {
            distances[i] = euclidean(trainX[i], x);
        }

        // find indices sorted by distance
        int[] idx = argsort(distances);

        // majority vote
        Map<Double, Integer> counts = new HashMap<>();

        for (int i = 0; i < k; i++) {
            double label = trainY[idx[i]];
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }

        double bestLabel = -1;
        int bestCount = -1;

        for (Map.Entry<Double, Integer> e : counts.entrySet()) {
            if (e.getValue() > bestCount) {
                bestCount = e.getValue();
                bestLabel = e.getKey();
            }
        }

        return bestLabel;
    }

    private double euclidean(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    private int[] argsort(double[] arr) {
        Integer[] idx = new Integer[arr.length];
        for (int i = 0; i < arr.length; i++) idx[i] = i;

        java.util.Arrays.sort(idx, (i, j) -> Double.compare(arr[i], arr[j]));

        int[] result = new int[arr.length];
        for (int i = 0; i < arr.length; i++) result[i] = idx[i];
        return result;
    }
}
