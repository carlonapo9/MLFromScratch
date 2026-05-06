package ml.utils;

public class Metrics {

    public static double accuracy(double[] yTrue, double[] yPred) {
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == yPred[i]) correct++;
        }
        return (double) correct / yTrue.length;
    }

    public static double precision(double[] yTrue, double[] yPred) {
        int tp = 0;
        int fp = 0;

        for (int i = 0; i < yTrue.length; i++) {
            if (yPred[i] == 1) {
                if (yTrue[i] == 1) tp++;
                else fp++;
            }
        }

        return tp + fp == 0 ? 0 : (double) tp / (tp + fp);
    }

    public static double recall(double[] yTrue, double[] yPred) {
        int tp = 0;
        int fn = 0;

        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == 1) {
                if (yPred[i] == 1) tp++;
                else fn++;
            }
        }

        return tp + fn == 0 ? 0 : (double) tp / (tp + fn);
    }

    public static double f1(double precision, double recall) {
        return (precision + recall == 0) ? 0 : 2 * (precision * recall) / (precision + recall);
    }
}
