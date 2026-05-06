package ml.algorithms;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

public class OneVsRestClassifier extends Algorithm {

    private final Supplier<Algorithm> factory;
    private final int numClasses;
    private final List<Algorithm> models = new ArrayList<>();

    public OneVsRestClassifier(Supplier<Algorithm> factory, int numClasses) {
        this.factory = factory;
        this.numClasses = numClasses;
    }

    @Override
    public void fit(double[][] X, double[] y) {

        models.clear();

        for (int c = 0; c < numClasses; c++) {

            // ⭐ OvR must use +1 for target class, -1 for others
            double[] binaryY = new double[y.length];
            for (int i = 0; i < y.length; i++) {
                binaryY[i] = (y[i] == c) ? 1.0 : -1.0;
            }

            Algorithm model = factory.get();
            model.fit(X, binaryY);

            models.add(model);
        }
    }

    @Override
    public double predict(double[] x) {

        double bestScore = -Double.MAX_VALUE;
        int bestClass = -1;

        for (int c = 0; c < numClasses; c++) {

            Algorithm model = models.get(c);

            double score;

            // ⭐ Logistic Regression → probability
            if (model instanceof LogisticRegression lr) {
                score = lr.predictProbability(x);

                // ⭐ SVM → margin (decision function)
            } else if (model instanceof SVMClassifierRBF svm) {
                score = svm.decisionFunction(x);

                // fallback for other models
            } else {
                score = model.predict(x);
            }

            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }

        return bestClass;
    }
}
