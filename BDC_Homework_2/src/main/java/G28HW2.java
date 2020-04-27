import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.log4j.Logger;
import org.apache.log4j.Level;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;
import shapeless.Tuple;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

public abstract class G28HW2 {

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Auxiliary methods
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }

    public static void main(String[] args) throws IOException {
        String filename = args[0];
        ArrayList<Vector> inputPoints = new ArrayList<>();
        inputPoints = readVectorsSeq(filename);
    }

    /**
     * receives in input a set of points S and returns the max distance between two points in S.
     * @param S input set
     * @return
     *
     * EXACT ALGORITHM
     * Max distance =  max distance returned by the method
     * Running time =  running time of the method.
     */
    public abstract double exactMPD(ArrayList<Vector> S);

    /**
     * receives in input a set of points S and an interger k < |S|, selects k points at random from S (let S' denote
     * the set of these k points) and returns the maximum distance d(x,y), over all x in S' and y in S. Define a
     * constant SEED in your main program (e.g., assigning it one of your university IDs as a value), and use that
     * value as a seed for the random generator. For Java users: SEED must be a long and you can use method setSeed
     * from your random generator to initialize the seed.
     * @param S input set
     * @param k number of clusters
     * @return
     *
     * 2-APPROXIMATION ALGORITHM
     * k =  value of k.
     * Max distance =  max distance returned by the method
     * Running time =  running time of the method.
     */
    public abstract double twoApproxMPD(ArrayList<Vector> S, int k);

    /**
     *  receives in input a set of points S and an integer k < |S|, and returns a set C of k centers selected from
     *  S using the Farthest-First Traversal algorithm. It is important that kCenterMPD(S,k) run in O(|S|*k) time
     * @param S input set
     * @param k number of cluster
     * @return
     *
     * k-CENTER-BASED ALGORITHM
     * k =  value of k.
     * Max distance =  max distance returned by exactMPD(centers)
     * Running time =  combined running time of the two methods.
     */
    public abstract ArrayList<Vector> kCenterMPD(ArrayList<Vector> S, int k);
}
