package ml.algorithms;

import java.util.Random;

public class MLPClassifier extends Algorithm {

    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;
    private final int epochs;
    private final double lr;

    private double[][] W1;
    private double[] b1;
    private double[][] W2;
    private double[] b2;

    private final Random rand = new Random(42);

    public MLPClassifier(int inputSize, int hiddenSize, int outputSize, int epochs, double lr) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.epochs = epochs;
        this.lr = lr;

        W1 = new double[inputSize][hiddenSize];
        b1 = new double[hiddenSize];
        W2 = new double[hiddenSize][outputSize];
        b2 = new double[outputSize];

        init(W1);
        init(W2);
    }

    private void init(double[][] W) {
        for (int i = 0; i < W.length; i++)
            for (int j = 0; j < W[0].length; j++)
                W[i][j] = rand.nextGaussian() * 0.1;
    }

    @Override
    public void fit(double[][] X, double[] y) {
        int n = X.length;

        for (int epoch = 0; epoch < epochs; epoch++) {

            for (int i = 0; i < n; i++) {

                double[] x = X[i];
                int label = (int) y[i];

                // Forward pass
                double[] z1 = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    for (int k = 0; k < inputSize; k++)
                        z1[j] += x[k] * W1[k][j];
                    z1[j] += b1[j];
                    z1[j] = Math.max(0, z1[j]); // ReLU
                }

                double[] z2 = new double[outputSize];
                for (int j = 0; j < outputSize; j++) {
                    for (int k = 0; k < hiddenSize; k++)
                        z2[j] += z1[k] * W2[k][j];
                    z2[j] += b2[j];
                }

                double[] probs = softmax(z2);

                // Backprop
                double[] dZ2 = new double[outputSize];
                for (int j = 0; j < outputSize; j++)
                    dZ2[j] = probs[j] - (j == label ? 1 : 0);

                double[] dZ1 = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    double grad = 0;
                    for (int k = 0; k < outputSize; k++)
                        grad += dZ2[k] * W2[j][k];
                    dZ1[j] = z1[j] > 0 ? grad : 0;
                }

                // Update W2, b2
                for (int j = 0; j < outputSize; j++) {
                    for (int k = 0; k < hiddenSize; k++)
                        W2[k][j] -= lr * dZ2[j] * z1[k];
                    b2[j] -= lr * dZ2[j];
                }

                // Update W1, b1
                for (int j = 0; j < hiddenSize; j++) {
                    for (int k = 0; k < inputSize; k++)
                        W1[k][j] -= lr * dZ1[j] * x[k];
                    b1[j] -= lr * dZ1[j];
                }
            }
        }
    }

    private double[] softmax(double[] z) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : z) max = Math.max(max, v);

        double sum = 0;
        double[] out = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            out[i] = Math.exp(z[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < z.length; i++)
            out[i] /= sum;

        return out;
    }

    @Override
    public double predict(double[] x) {
        double[] z1 = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            for (int k = 0; k < inputSize; k++)
                z1[j] += x[k] * W1[k][j];
            z1[j] += b1[j];
            z1[j] = Math.max(0, z1[j]);
        }

        double[] z2 = new double[outputSize];
        for (int j = 0; j < outputSize; j++) {
            for (int k = 0; k < hiddenSize; k++)
                z2[j] += z1[k] * W2[k][j];
            z2[j] += b2[j];
        }

        double[] probs = softmax(z2);

        int best = 0;
        for (int i = 1; i < probs.length; i++)
            if (probs[i] > probs[best]) best = i;

        return best;
    }
}
