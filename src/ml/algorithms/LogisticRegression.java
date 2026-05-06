package ml.algorithms;

public class LogisticRegression extends Algorithm {

    private double[] weights;
    private int epochs;
    private double lr;

    public LogisticRegression(int epochs, double lr) {
        this.epochs = epochs;
        this.lr = lr;
    }

    @Override
    public void fit(double[][] X, double[] y) {
        int nSamples = X.length;
        int nFeatures = X[0].length;

        weights = new double[nFeatures];

        for (int epoch = 0; epoch < epochs; epoch++) {

            double[] grad = new double[nFeatures];

            for (int i = 0; i < nSamples; i++) {
                double z = 0;
                for (int j = 0; j < nFeatures; j++) {
                    z += weights[j] * X[i][j];
                }

                double pred = 1.0 / (1.0 + Math.exp(-z));
                double error = pred - y[i];

                for (int j = 0; j < nFeatures; j++) {
                    grad[j] += error * X[i][j];
                }
            }

            for (int j = 0; j < nFeatures; j++) {
                weights[j] -= lr * grad[j] / nSamples;
            }
        }
    }

    // ⭐ Probability output for OvR
    public double predictProbability(double[] x) {
        double z = 0;
        for (int i = 0; i < weights.length; i++) {
            z += weights[i] * x[i];
        }
        return 1.0 / (1.0 + Math.exp(-z));
    }

    @Override
    public double predict(double[] x) {
        return predictProbability(x) >= 0.5 ? 1.0 : 0.0;
    }
}
