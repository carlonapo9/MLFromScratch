import ml.algorithms.*;
import ml.data.CSVLoader;
import ml.data.Dataset;
import ml.data.TrainTestSplit;
import ml.utils.Metrics;
import ml.utils.Scaler;
import ml.utils.Shuffle;

import java.util.Arrays;

public class Main {

    record DatasetConfig(String name, String path) {}

    public static void main(String[] args) throws Exception {

        DatasetConfig[] datasets = new DatasetConfig[] {
                new DatasetConfig("Iris", "datasets/iris.csv"),
                new DatasetConfig("Breast Cancer (WDBC)", "datasets/wdbc.csv"),
                new DatasetConfig("Wine Quality (Red)", "datasets/winequality-red.csv"),
                new DatasetConfig("Wine Quality (White)", "datasets/winequality-white.csv")
        };

        int runs = 5;

        for (DatasetConfig cfg : datasets) {

            System.out.println("\n==============================");
            System.out.println(" DATASET: " + cfg.name());
            System.out.println("==============================");

            Dataset data = CSVLoader.loadAuto(cfg.path());
            double[][] X = data.X;
            double[] y = data.y;

            int nFeatures = X[0].length;
            int nClasses = numClasses(y);

            Algorithm[] models = new Algorithm[] {
                    new OneVsRestClassifier(() -> new KNNClassifier(5), nClasses),
                    new OneVsRestClassifier(() -> new LogisticRegression(2000, 0.1), nClasses),
                    new OneVsRestClassifier(() -> new SVMClassifierRBF(20, 0.01, 1.0, 0.5), nClasses),
                    new MLPClassifier(nFeatures, 8, nClasses, 500, 0.01)
            };

            String[] modelNames = {
                    "KNN (k=5)",
                    "Logistic Regression",
                    "RBF SVM",
                    "MLP (" + nFeatures + "→8→" + nClasses + ")"
            };

            double[] sumAcc = new double[models.length];

            for (int run = 1; run <= runs; run++) {

                System.out.println("\n=== RUN " + run + " ===");

                Shuffle.shuffle(X, y);
                TrainTestSplit.SplitResult split = TrainTestSplit.split(X, y, 0.2);

                Scaler scaler = new Scaler();
                scaler.fitStandard(split.Xtrain);

                double[][] Xtrain = scaler.transformStandard(split.Xtrain);
                double[][] Xtest = scaler.transformStandard(split.Xtest);

                for (int i = 0; i < models.length; i++) {

                    Algorithm model = models[i];
                    model.fit(Xtrain, split.ytrain);

                    double[] preds = model.predict(Xtest);
                    double acc = Metrics.accuracy(split.ytest, preds);

                    sumAcc[i] += acc;

                    System.out.println(modelNames[i] + " → accuracy: " + acc);
                }
            }

            System.out.println("\n=== FINAL AVERAGE ACCURACIES FOR " + cfg.name() + " ===");
            for (int i = 0; i < models.length; i++) {
                double avg = sumAcc[i] / runs;
                System.out.println(modelNames[i] + " → avg accuracy: " + avg);
            }
        }
    }

    private static int numClasses(double[] y) {
        return (int) Arrays.stream(y).distinct().count();
    }
}
