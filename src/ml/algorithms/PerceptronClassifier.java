package ml.algorithms;

import java.util.Arrays;

public class PerceptronClassifier extends Algorithm {

    private double[] weights;
    private double bias;
    private double learningRate = 0.1;
    private int epochs = 1000;

    public PerceptronClassifier(int epochs, double learningRate) {
        this.epochs = epochs;
        this.learningRate = learningRate;
    }

    @Override
    public void fit(double[][] X, double[] y) {
        int nSamples = X.length;
        int nFeatures = X[0].length;

        weights = new double[nFeatures];
        bias = 0;

        Arrays.fill(weights, 0);

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < nSamples; i++) {

                double linear = dot(X[i], weights) + bias;
                double prediction = linear >= 0 ? 1 : 0;

                double error = y[i] - prediction;

                // update rule
                for (int j = 0; j < nFeatures; j++) {
                    weights[j] += learningRate * error * X[i][j];
                }
                bias += learningRate * error;
            }
        }
    }

    @Override
    public double predict(double[] x) {
        double linear = dot(x, weights) + bias;
        return linear >= 0 ? 1 : 0;
    }

    private double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }
}
