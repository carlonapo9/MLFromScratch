package ml.utils;

import java.util.Random;

public class Shuffle {

    public static void shuffle(double[][] X, double[] y) {
        Random rand = new Random();
        for (int i = X.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);

            double[] tempX = X[i];
            X[i] = X[j];
            X[j] = tempX;

            double tempY = y[i];
            y[i] = y[j];
            y[j] = tempY;
        }
    }
}
