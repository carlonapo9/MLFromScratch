package ml.data;

public class Dataset {
    public final double[][] X;
    public final double[] y;

    public Dataset(double[][] X, double[] y) {
        this.X = X;
        this.y = y;
    }
}
