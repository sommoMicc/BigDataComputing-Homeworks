import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Random;

public class G28HW2 {

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
        
        int k = Integer.parseInt(args[1]);

        // exactMPD(inputPoints);
        double twoApproxDistance = G28HW2.twoApproxMPD(inputPoints, k);
        System.out.println("TwoApproxDistance: " + twoApproxDistance);
        // kCenterMPD(inputPoints, k);
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
    public static double exactMPD(ArrayList<Vector> S) {
        return 0.0;
    }

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
    public static double twoApproxMPD(ArrayList<Vector> S, int k) {
        Random random = new Random();
        ArrayList<Vector> S_prime = new ArrayList<Vector>();

        for(int i=0; i<k; i++)
            S_prime.add(S.get(random.nextInt(S.size())));

        double max_distance = -1;

        for(int i=0; i<S_prime.size(); i++) {
            Vector x = S_prime.get(i);
            for(int j=0; j<S_prime.size(); j++) {
                Vector y = S_prime.get(j);

                double current_distance = G28HW2.euclideanDistance(x,y);
                if(current_distance > max_distance) {
                    max_distance = current_distance;
                }
            }
        }
        return max_distance;
    }

    public static double euclideanDistance(Vector x, Vector y) {
        return Math.sqrt(Vectors.sqdist(x,y));
    }

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
    public static ArrayList<Vector> kCenterMPD(ArrayList<Vector> S, int k) {
        return new ArrayList<>();
    }
}
