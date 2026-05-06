package ml.algorithms;

import java.util.ArrayList;
import java.util.List;

public class SVMClassifierRBF extends Algorithm {

    private final int epochs;
    private final double lr;
    private final double C;
    private final double gamma;

    private final List<double[]> supportVectors = new ArrayList<>();
    private final List<Double> supportLabels = new ArrayList<>();
    private final List<Double> alphas = new ArrayList<>();

    private double b = 0;

    public SVMClassifierRBF(int epochs, double lr, double C, double gamma) {
        this.epochs = epochs;
        this.lr = lr;
        this.C = C;
        this.gamma = gamma;
    }

    @Override
    public void fit(double[][] X, double[] y) {
        int n = X.length;

        supportVectors.clear();
        supportLabels.clear();
        alphas.clear();
        b = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {

            double lr_t = lr / (1.0 + epoch * 0.001);

            for (int i = 0; i < n; i++) {

                double margin = y[i] * decisionFunction(X[i]);

                if (margin < 1) {
                    supportVectors.add(X[i]);
                    supportLabels.add(y[i]);
                    alphas.add(lr_t * C);

                    b += lr_t * C * y[i];
                }
            }
        }
    }

    private double rbf(double[] x1, double[] x2) {
        double sum = 0;
        for (int i = 0; i < x1.length; i++) {
            double diff = x1[i] - x2[i];
            sum += diff * diff;
        }
        return Math.exp(-gamma * sum);
    }

    public double decisionFunction(double[] x) {
        double sum = 0;

        for (int i = 0; i < supportVectors.size(); i++) {
            sum += alphas.get(i) * supportLabels.get(i) * rbf(x, supportVectors.get(i));
        }

        return sum + b;
    }

    @Override
    public double predict(double[] x) {
        return decisionFunction(x);
    }
}
