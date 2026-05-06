import ml.algorithms.*;
import ml.data.CSVLoader;
import ml.data.Dataset;
import ml.data.TrainTestSplit;
import ml.utils.Metrics;
import ml.utils.Scaler;
import ml.utils.Shuffle;

public class Main {

    record DatasetConfig(String name, String path, boolean hasHeader, int labelIndex) {}

    public static void main(String[] args) throws Exception {

        DatasetConfig[] datasets = new DatasetConfig[] {
                new DatasetConfig("Iris", "datasets/iris.csv", true, 4),
                new DatasetConfig("Wine", "datasets/wine.csv", true, 0),
                new DatasetConfig("Breast Cancer", "datasets/breast_cancer.csv", true, 1)
        };

        Algorithm[] models = new Algorithm[] {
                new OneVsRestClassifier(() -> new KNNClassifier(5), 3),
                new OneVsRestClassifier(() -> new LogisticRegression(2000, 0.1), 3),
                new OneVsRestClassifier(() -> new SVMClassifierRBF(20, 0.01, 1.0, 0.5), 3),
                new MLPClassifier(4, 8, 3, 500, 0.01)
        };

        String[] modelNames = {
                "KNN (k=5)",
                "Logistic Regression",
                "RBF SVM",
                "MLP (4→8→3)"
        };

        int runs = 10;

        for (DatasetConfig cfg : datasets) {

            System.out.println("\n==============================");
            System.out.println(" DATASET: " + cfg.name());
            System.out.println("==============================");

            Dataset data = CSVLoader.load(cfg.path(), cfg.hasHeader(), cfg.labelIndex());
            double[][] X = data.X;
            double[] y = data.y;

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
}
