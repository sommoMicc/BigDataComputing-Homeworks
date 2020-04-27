import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class G28HW1 {

    public static void main(String[] args) throws IOException {
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: number_partitions, <path to file>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        SparkConf conf = new SparkConf(true).setAppName("Homework1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int K = Integer.parseInt(args[0]);

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> docs = sc.textFile(args[1]).repartition(K);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING GLOBAL VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        JavaPairRDD<String, Long> count;
        
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CLASS COUNT WITH DETERMINISTIC PARTITION
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        count = docs
                .flatMapToPair((line) -> {    // <-- MAP PHASE (R1)  generate key value pairs for every line
                    String[] tokens = line.split(" ");
                    ArrayList<Tuple2<Long, String>> pairs = new ArrayList<>();

                    // Read all the classes in the input file
                    // Each line of the file, so each instance of the reducer, represents a single object/class
                    pairs.add(new Tuple2<Long,String>(Long.parseLong(tokens[0]) % K, tokens[1]));  // generate key value pair

                    return pairs.iterator();
                })
                .groupByKey()    // <-- REDUCE PHASE (R1) return a list of (k, list of Strings)
                .flatMapToPair((cls) -> {  // cls is the list of key-value pairs <K, Classes> of the same partition
                    Iterable<String> classes = cls._2(); // List of classes of the current partition
                    HashMap<String, Long> counts = new HashMap<>();
                    for (String e : classes) {
                        counts.put(e, 1L + counts.getOrDefault(e, 0L));
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<String, Long>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();  // Returns a list of intermediate key-value pairs <Class, count_in_partition>,
                                              // where "count_in_partition" is the number of objects having the class
                                              // "Class" that are in the same partition
                })
                .reduceByKey(Long::sum);    // <-- REDUCE PHASE (R2). Sums all the values that have the same key (class)

        System.out.println("VERSION WITH DETERMINISTIC PARTITIONS");
        System.out.print("OUTPUT PAIRS:");
        for(Tuple2<String,Long> tuple:count.sortByKey().collect()) {
            // Outputs the solution
            System.out.print(" ("+tuple._1()+","+tuple._2()+")");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CLASS COUNT WITH SPARK PARTITIONS
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        JavaPairRDD<String, Tuple2<Long, Long>> sp_count;
        sp_count = docs
                .flatMapToPair((line) -> {    // <-- MAP PHASE (R1)

                    String[] tokens = line.split(" ");
                    ArrayList<Tuple2<Long, String>> pairs = new ArrayList<>();

                    // Max partition size
                    long J = (long) Math.floor(Math.sqrt(K));

                    pairs.add(new Tuple2<Long,String>(Long.parseLong(tokens[0]) % J, tokens[1]));  // generate key value pair

                    return pairs.iterator();
                })
                .mapPartitionsToPair((cc) -> {    // <-- REDUCE PHASE (R1)
                    long N_max = 0L; // Number of operations in each worker
                    HashMap<String, Long> counts = new HashMap<>();
                    while (cc.hasNext()) {
                        N_max += 1L;
                        Tuple2<Long, String> tuple = cc.next();
                        // Count the occurrencies of a specific class in the partition
                        counts.put(tuple._2(), 1L + counts.getOrDefault(tuple._2(), 0L));
                    }
                    ArrayList<Tuple2<String, Tuple2<Long, Long>>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        // Generate the key-value pair, where the key is the class and the value is a tuple <freq, n_max>,
                        // where "freq" is the frequency of the class in the partition and "n_max" is the number of
                        // operation done by the current worker.
                        pairs.add(new Tuple2<String, Tuple2<Long,Long>>(e.getKey(), new Tuple2<Long, Long>(e.getValue(), N_max)));
                    }
                    return pairs.iterator();
                })
                .groupByKey()   // <-- REDUCE PHASE (R2)
                .mapValues((it)->{
                    long sum = 0L;
                    long max = -1L;

                    for (Tuple2<Long, Long> el :it){
                        // Sum all the occurrencies of the same class
                        sum = el._1()+ sum;
                        if (el._2() > max){
                            // Detect the maximum number of operations
                            max = el._2();
                        }
                    }

                    return new Tuple2<>(sum,max);
                });

        long N_max = -1;
        for(Tuple2<String,Tuple2<Long, Long>> tuple:sp_count.sortByKey().collect()) {
            // Get the absolute maximum number of pairs that in round 1 are processed by a single reducer
            if (tuple._2()._2() > N_max){
                N_max=tuple._2()._2();
            }
        }
        // Most frequent class computation
        Tuple2<String,Tuple2<Long,Long>> mf_class = sp_count.reduce((c1,c2)->(c1._2()._1()>c2._2()._1()?c1:c2));

        System.out.println("\n\nVERSION WITH SPARK PARTITIONS");
        System.out.println("Most frequent class = ("+mf_class._1()+","+ mf_class._2()._1()+")");
        System.out.println("Max partition size = "+N_max);

    }

}
