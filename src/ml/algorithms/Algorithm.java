package ml.algorithms;

public abstract class Algorithm {

    // Train the model on data
    public abstract void fit(double[][] X, double[] y);

    // Predict a single value
    public abstract double predict(double[] x);

    // Predict multiple values
    public double[] predict(double[][] X) {
        double[] results = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            results[i] = predict(X[i]);
        }
        return results;
    }
}
