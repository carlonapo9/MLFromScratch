package ml.utils;

public class Scaler {

    private double[] min;
    private double[] max;

    private double[] mean;
    private double[] std;

    // MinMaxScaler
    public void fitMinMax(double[][] X) {
        int nFeatures = X[0].length;
        min = new double[nFeatures];
        max = new double[nFeatures];

        for (int j = 0; j < nFeatures; j++) {
            min[j] = Double.POSITIVE_INFINITY;
            max[j] = Double.NEGATIVE_INFINITY;

            for (double[] row : X) {
                min[j] = Math.min(min[j], row[j]);
                max[j] = Math.max(max[j], row[j]);
            }
        }
    }

    public double[][] transformMinMax(double[][] X) {
        double[][] out = new double[X.length][X[0].length];

        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[0].length; j++) {
                out[i][j] = (X[i][j] - min[j]) / (max[j] - min[j] + 1e-9);
            }
        }
        return out;
    }

    // StandardScaler
    public void fitStandard(double[][] X) {
        int nFeatures = X[0].length;
        mean = new double[nFeatures];
        std = new double[nFeatures];

        for (int j = 0; j < nFeatures; j++) {
            double sum = 0;
            for (double[] row : X) sum += row[j];
            mean[j] = sum / X.length;

            double var = 0;
            for (double[] row : X) var += Math.pow(row[j] - mean[j], 2);
            std[j] = Math.sqrt(var / X.length);
        }
    }

    public double[][] transformStandard(double[][] X) {
        double[][] out = new double[X.length][X[0].length];

        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[0].length; j++) {
                out[i][j] = (X[i][j] - mean[j]) / (std[j] + 1e-9);
            }
        }
        return out;
    }
}
