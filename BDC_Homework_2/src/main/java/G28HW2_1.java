
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

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
        // Reading points from a file whose name is provided as args[0]
        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        ArrayList<Vector> inputPoints = new ArrayList<>();
        inputPoints = readVectorsSeq(filename);

        System.out.println("\n1-EXACT ALGORITHM\n");
        long startTime1 = System.currentTimeMillis();
        double exactMaxDistance1 = exactMPD(inputPoints);
        System.out.println("Max distance = " + exactMaxDistance1 +
                "\nRunning time = " + (System.currentTimeMillis() - startTime1) + "\n");

        System.out.println("\n2-APPROXIMATION ALGORITHM\n");

        System.out.println("\n3-KCENTER-MPD\n");
        long startTime3 = System.currentTimeMillis();
        ArrayList<Vector> centers = kCenterMPD(inputPoints, k);
        double  exactMaxDistance2 = exactMPD(centers);
        System.out.println("k = " + k +
                           "\nMax distance = " + exactMaxDistance2 +
                           "\nRunning time = " + (System.currentTimeMillis() - startTime3) + "\n");

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
    // Exact algorithm
    public static double exactMPD(ArrayList<Vector> s) {
        double maxDistance = 0;
        for (Vector v1 : s) {
            for (Vector v2 : s) {
                double currentDistance = Math.sqrt(Vectors.sqdist(v1, v2));
                if (currentDistance > maxDistance) {
                    maxDistance = currentDistance;
                }
            }
        }
        return maxDistance;
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
    //public abstract double twoApproxMPD(ArrayList<Vector> S, int k);

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
    public static ArrayList<Vector> kCenterMPD(ArrayList<Vector> P, int k) throws IllegalArgumentException
    {
        int PSize = P.size();
        // A control on the input. Simple control to verify that k ha not a number grater than PSize
        if (k > PSize)
            throw new IllegalArgumentException("K cannot be greater than the number of P!");

        // Choose a random point as first center
        final boolean chosen [] = new boolean[PSize];
        final int firstPointIndex = (int)(Math.random() * PSize);

        // Set to true that we chosen the point
        chosen[firstPointIndex] = true;

        // Add the point to the centers
        ArrayList<Vector> centers = new ArrayList<>();
        centers.add(P.get(firstPointIndex));

        // Let's create the matrix for the distances
        double [] dist = new double[PSize];

        // Let's memorize the minDistances
        double minDist[] = new double[PSize];


        // Now we compute the distances from all the points to the center
        int index = 0;
        for (Vector v : P)
        {
            dist [index] = Vectors.sqdist(v, centers.get(0));
            minDist[index] = dist [index++];
        }

        // Let's find all the other centers
        for (int h = 1; h < k; h++)
        {
            double max = -1;
            index = -1;

            // First, compute the distances from the new center, then find the max min
            for (int j = 0; j < PSize; j++)
            {
                if (chosen[j] == false)
                {
                    // Compute all the new distances
                    dist[j] = Vectors.sqdist(P.get(j), centers.get(h-1));
                    if (dist[j] < minDist[j])
                        minDist[j] = dist[j];
                }
            }

            // We choose the maximum minimum value
            for (int j = 0; j < PSize; j++)
            {
                if (chosen[j] == false) {
                    if (minDist[j] > max) {
                        max = minDist[j];
                        index = j;
                    }
                }
            }

            // Add the new center
            centers.add(P.get(index));
            chosen[index] = true;
        }
        return centers;
    }
}
