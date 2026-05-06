package ml.data;

import java.util.Random;

public class TrainTestSplit {

    public static SplitResult split(double[][] X, double[] y, double testRatio) {
        int n = X.length;
        int testSize = (int) (n * testRatio);
        int trainSize = n - testSize;

        double[][] Xtrain = new double[trainSize][];
        double[] ytrain = new double[trainSize];

        double[][] Xtest = new double[testSize][];
        double[] ytest = new double[testSize];

        int[] indices = shuffledIndices(n);

        for (int i = 0; i < trainSize; i++) {
            Xtrain[i] = X[indices[i]];
            ytrain[i] = y[indices[i]];
        }

        for (int i = 0; i < testSize; i++) {
            Xtest[i] = X[indices[trainSize + i]];
            ytest[i] = y[indices[trainSize + i]];
        }

        return new SplitResult(Xtrain, ytrain, Xtest, ytest);
    }

    private static int[] shuffledIndices(int n) {
        int[] arr = new int[n];
        for (int i = 0; i < n; i++) arr[i] = i;

        Random rand = new Random();
        for (int i = n - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
        return arr;
    }

    public static class SplitResult {
        public double[][] Xtrain;
        public double[] ytrain;
        public double[][] Xtest;
        public double[] ytest;

        public SplitResult(double[][] Xtrain, double[] ytrain,
                           double[][] Xtest, double[] ytest) {
            this.Xtrain = Xtrain;
            this.ytrain = ytrain;
            this.Xtest = Xtest;
            this.ytest = ytest;
        }
    }
}
