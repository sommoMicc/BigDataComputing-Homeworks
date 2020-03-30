import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

public class TemplateHW1 {

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

    long numdocs, numwords;
    numdocs = docs.count();
    System.out.println("Number of documents = " + numdocs);
    JavaPairRDD<String, Long> count;

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // STANDARD WORD COUNT with reduceByKey
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    count = docs
            .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
              String[] tokens = document.split(" ");
              HashMap<String, Long> counts = new HashMap<>();
              ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
              for (String token : tokens) {
                counts.put(token, 1L + counts.getOrDefault(token, 0L));
              }
              for (Map.Entry<String, Long> e : counts.entrySet()) {
                pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
              }
              return pairs.iterator();
            })
            .reduceByKey((x, y) -> x+y);    // <-- REDUCE PHASE (R1)
    numwords = count.count();
    System.out.println("Number of distinct words in the documents = " + numwords);

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // IMPROVED WORD COUNT with groupByKey
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    Random randomGenerator = new Random();
    count = docs
            .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
                String[] tokens = document.split(" ");
                HashMap<String, Long> counts = new HashMap<>();
                ArrayList<Tuple2<Integer, Tuple2<String, Long>>> pairs = new ArrayList<>();
                for (String token : tokens) {
                    counts.put(token, 1L + counts.getOrDefault(token, 0L));
                }
                for (Map.Entry<String, Long> e : counts.entrySet()) {
                    pairs.add(new Tuple2<>(randomGenerator.nextInt(K), new Tuple2<>(e.getKey(), e.getValue())));
                }
                return pairs.iterator();
            })
            .groupByKey()    // <-- REDUCE PHASE (R1)
            .flatMapToPair((triplet) -> {
                HashMap<String, Long> counts = new HashMap<>();
                for (Tuple2<String, Long> c : triplet._2()) {
                    counts.put(c._1(), c._2() + counts.getOrDefault(c._1(), 0L));
                }
                ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                for (Map.Entry<String, Long> e : counts.entrySet()) {
                    pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                }
                return pairs.iterator();
            })
            .groupByKey()    // <-- REDUCE PHASE (R2)
            .mapValues((it) -> {
                long sum = 0;
                for (long c : it) {
                    sum += c;
                }
                return sum;
            });
    numwords = count.count();
    System.out.println("Number of distinct words in the documents = " + numwords);

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // IMPROVED WORD COUNT with mapPartitions
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    count = docs
	    .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
                 String[] tokens = document.split(" ");
                 HashMap<String, Long> counts = new HashMap<>();
                 ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                 for (String token : tokens) {
                     counts.put(token, 1L + counts.getOrDefault(token, 0L));
                 }
                 for (Map.Entry<String, Long> e : counts.entrySet()) {
                     pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                 }
                 return pairs.iterator();
            })
            .mapPartitionsToPair((wc) -> {    // <-- REDUCE PHASE (R1)
                 HashMap<String, Long> counts = new HashMap<>();
                 while (wc.hasNext()){
                      Tuple2<String, Long> tuple = wc.next();
                      counts.put(tuple._1(), tuple._2() + counts.getOrDefault(tuple._1(), 0L));
                 }
                 ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                 for (Map.Entry<String, Long> e : counts.entrySet()) {
                     pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                 }
                 return pairs.iterator();
            })
            .groupByKey()     // <-- REDUCE PHASE (R2)
            .mapValues((it) -> {
                 long sum = 0;
                 for (long c : it) {
                     sum += c;
                 }
                 return sum;
            });
    numwords = count.count();
    System.out.println("Number of distinct words in the documents = " + numwords);

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // COMPUTE AVERAGE WORD LENGTH
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    int avgwordlength = count
            .map((tuple) -> tuple._1().length())
            .reduce((x, y) -> x+y);
    System.out.println("Average word length = " + avgwordlength/numwords);
  }

}
