import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Arrays;


public class G28HW3 {

    private static Random random = new Random();

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Auxiliary methods
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static double euclideanDistance(Vector x, Vector y) {
        return Math.sqrt(Vectors.sqdist(x, y));
    }

    /**
     * receives in input a set of points S and an integer k < |S|, and returns a set C of k centers selected from
     * S using the Farthest-First Traversal algorithm. It is important that kCenterMPD(S,k) run in O(|S|*k) time
     *
     * @param S input set
     * @param k number of cluster
     * @return k-CENTER-BASED ALGORITHM
     * k =  value of k.
     * Max distance =  max distance returned by exactMPD(centers)
     * Running time =  combined running time of the two methods.
     */

    public static ArrayList<Vector> kCenterMPD(ArrayList<Vector> S, int k) throws IllegalArgumentException {
        int PSize = S.size();
        // A control on the input. Simple control to verify that k ha not a number grater than PSize
        if (k > PSize)
            throw new IllegalArgumentException("K cannot be greater than the number of P!");

        // Choose a random point as first center
        final boolean chosen[] = new boolean[PSize];
        final int firstPointIndex = G28HW3.random.nextInt(PSize);

        // Set to true that we chosen the point
        chosen[firstPointIndex] = true;

        // Add the point to the centers
        ArrayList<Vector> centers = new ArrayList<>();
        centers.add(S.get(firstPointIndex));

        // Let's create the matrix for the distances
        double[] dist = new double[PSize];

        // Let's memorize the minDistances
        double minDist[] = new double[PSize];


        // Now we compute the distances from all the points to the center
        int index = 0;
        for (Vector v : S) {
            dist[index] = G28HW3.euclideanDistance(v, centers.get(0));
            minDist[index] = dist[index++];
        }

        // Let's find all the other centers
        for (int h = 1; h < k; h++) {
            double max = -1;
            index = -1;

            // First, compute the distances from the new center, then find the max min
            for (int j = 0; j < PSize; j++) {
                if (!chosen[j]) {
                    // Compute all the new distances
                    dist[j] = G28HW3.euclideanDistance(S.get(j), centers.get(h - 1));
                    if (dist[j] < minDist[j])
                        minDist[j] = dist[j];
                }
            }

            // We choose the maximum minimum value
            for (int j = 0; j < PSize; j++) {
                if (!chosen[j]) {
                    if (minDist[j] > max) {
                        max = minDist[j];
                        index = j;
                    }
                }
            }

            // Add the new center
            centers.add(S.get(index));
            chosen[index] = true;
        }
        return centers;
    }


    public static void main(String[] args) throws IOException {
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: number_partitions, <path to file>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: file_name K L");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        SparkConf conf = new SparkConf(true).setAppName("Homework3");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        String inputPath = args[0];
        int k = Integer.parseInt(args[1]);
        int L = Integer.parseInt(args[2]);


        long start = System.currentTimeMillis();
        // Read input file and subdivide it into K random partitions
        JavaRDD<Vector> inputPoints = sc.textFile(inputPath).map(G28HW3::strToVector).repartition(L).cache();
        long numdocs = inputPoints.count(); // perform an action to materialize rdd and count the time in the right way
        long end = System.currentTimeMillis();

        System.out.println("Number of points = " + numdocs);
        System.out.println("k = " + k);
        System.out.println("L = " + L);
        System.out.println("Initialization time = " + (end - start));


        ArrayList<Vector> solution = runMapReduce(inputPoints, k, L);

        double avgDist = measure(solution);
        System.out.println("Average distance = " + avgDist);

    }

    /**
     * Implements the map reduce algorithm for diversity maximization
     *
     * @param PointsRDD an rdd of points where each point is a Vector
     * @param k param of the diversity maximisation
     * @param L number of partitions to use
     * @return out points computed using runSequential method which are a solution to diversity maximization problem
     */

    public static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsRDD, int k, int L) {
        long start1 = System.currentTimeMillis();
        JavaRDD<Vector> outRound1 = pointsRDD.mapPartitions((cc) -> { // map to partition randomly the pointsRDD

            ArrayList<Vector> temp = new ArrayList<>();
            while (cc.hasNext()) {
                Vector el=cc.next();
                temp.add(el);
            }
            ArrayList<Vector> outFFT = kCenterMPD(temp, k);  // extracts k points from each partition using Farthest-First Traversal algorithm

            return outFFT.iterator();
        }).cache();  // cache to avoid lazy evaluation and to materialize the rdd in memory
        long temp = outRound1.count(); //to avoid lazy evaluation performs an action
        long end1 = System.currentTimeMillis();
        System.out.println("Runtime of round 1 = " + (end1 - start1));

        long start2 = System.currentTimeMillis();
        ArrayList<Vector> coreset = new ArrayList<>(outRound1.collect()); // collect the centers computed by the partitions
        ArrayList<Vector> out = runSequential(coreset, k);  // run sequential algorithm (2-approximation)
        long end2 = System.currentTimeMillis();
        System.out.println("Runtime of round 2 = " + (end2 - start2));

        return out;
    }

    /**
     * Compute the average distance between the solution points
     *
     * @param pointsSet an array list of points returned by runMapReduce
     * @return result: the average distance among the solution points
     */

    public static double measure(ArrayList<Vector> pointsSet) {
        int k = pointsSet.size();
        double result = 0;

        for (int i = 0; i < k; i++) {

            for (int j = i+1; j < k; j++) {
// compute the distance between a point and every other points and in the next iteration do not compute distances already computed
                    result = result + euclideanDistance(pointsSet.get(i), pointsSet.get(j));
                }
            }

        result = result / (k*(k-1)/2);  //divide by the number of distances computed

        return result;

    }



    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // METHOD runSequential
    // Sequential 2-approximation based on matching
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {

        final int n = points.size();
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);
        for (int iter=0; iter<k/2; iter++) {
            // Find the maximum distance pair among the candidates
            double maxDist = 0;
            int maxI = 0;
            int maxJ = 0;
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    for (int j = i+1; j < n; j++) {
                        if (candidates[j]) {
                            // Use squared euclidean distance to avoid an sqrt computation!
                            double d = Vectors.sqdist(points.get(i), points.get(j));
                            if (d > maxDist) {
                                maxDist = d;
                                maxI = i;
                                maxJ = j;
                            }
                        }
                    }
                }
            }
            // Add the points maximizing the distance to the solution
            result.add(points.get(maxI));
            result.add(points.get(maxJ));
            // Remove them from the set of candidates
            candidates[maxI] = false;
            candidates[maxJ] = false;
        }
        // Add an arbitrary point to the solution, if k is odd.
        if (k % 2 != 0) {
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    result.add(points.get(i));
                    break;
                }
            }
        }
        if (result.size() != k) {
            throw new IllegalStateException("Result of the wrong size");
        }
        return result;

    } // END runSequential

}
