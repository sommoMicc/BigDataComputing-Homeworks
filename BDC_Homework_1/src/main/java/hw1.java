import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.log4j.Logger;
import org.apache.log4j.Level;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;
import shapeless.Tuple;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

public class hw1 {

    public static void main(String[] args) throws IOException {
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: number_partitions, <path to file>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        System.setProperty("hadoop.home.dir", "C:\\winutils");
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
        
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // IMPROVED WORD COUNT with groupByKey
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        Random randomGenerator = new Random();
        count = docs
                .flatMapToPair((line) -> {    // <-- MAP PHASE (R1)  generate key value pairs for every line

                    String[] tokens = line.split(" ");
                    HashMap<Long, String> counts = new HashMap<>();
                    ArrayList<Tuple2<Long, String>> pairs = new ArrayList<>();
                    for (int i=0; i<2; i++){
                        counts.put(Long.parseLong(tokens[0]), tokens[1]);  // inserisce la chiave i e valore parola di ogni riga
                    }

                    for (Map.Entry<Long, String> e : counts.entrySet()) {
                        pairs.add(new Tuple2<Long,String>(e.getKey()%K, e.getValue()));  // generate key value pair
                    }
                    return pairs.iterator();
                })
                .groupByKey()    // <-- REDUCE PHASE (R1) return a list of (k, list of Strings)
                .flatMapToPair((parole) -> {  // parole è la lista di tutte le parole con la stessa chiave
                    Iterable<String> words = parole._2();
                    HashMap<String, Long> counts = new HashMap<>();
                    for (String e : words) {
                        counts.put(e, 1L + counts.getOrDefault(e, 0L));
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<String, Long>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();  // la reduce ha ritornato una lista formata da chiave->parola  e valore->suo conteggio
                })
                    // <-- REDUCE PHASE (R2)
                .reduceByKey((x,y) ->   // it è la lista di tutti i ounter parziali associata alla chiave graggruppata da grupByKey()
                    x+y
                );
        numwords = count.count();
        System.out.println("Number of distinct words in the documents = " + numwords);

        for(Tuple2<String,Long> tuple:count.sortByKey().collect()) {
            System.out.println("("+tuple._1()+","+tuple._2()+")");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // IMPROVED WORD COUNT with mapPartitions
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        JavaPairRDD<String, Tuple2<Long, Long>> count1;

        count1 = docs
                .flatMapToPair((line) -> {    // <-- MAP PHASE (R1)

                    String[] tokens = line.split(" ");
                    HashMap<Long, String> counts = new HashMap<>();
                    ArrayList<Tuple2<Long, String>> pairs = new ArrayList<>();
                    for (int i=0; i<2; i++){
                        counts.put(Long.parseLong(tokens[0]), tokens[1]);  // inserisce la chiave i e valore parola di ogni riga
                    }
                    Long J = (long)Math.floor(Math.sqrt(K));

                    for (Map.Entry<Long, String> e : counts.entrySet()) {
                        pairs.add(new Tuple2<Long,String>(e.getKey()%J, e.getValue()));  // generate key value pair
                    }
                    return pairs.iterator();
                })
                .mapPartitionsToPair((wc) -> {    // <-- REDUCE PHASE (R1)
                    Long N_max = 0L;
                    HashMap<String, Long> counts = new HashMap<>();
                    while (wc.hasNext()) {
                        N_max += 1L;
                        Tuple2<Long, String> tuple = wc.next();
                        counts.put(tuple._2(), 1L + counts.getOrDefault(tuple._2(), 0L));
                    }
                    ArrayList<Tuple2<String, Tuple2<Long, Long>>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<String, Tuple2<Long,Long>>(e.getKey(), new Tuple2<Long, Long>(e.getValue(), N_max)));
                    }
                    return pairs.iterator();
                })
                .groupByKey()
                .mapValues((it)->{
                    //HashMap<Long, String> counts = new HashMap<>();
                    long sum = 0L;
                    long max = -1L;
                    for (Tuple2<Long, Long> el :it){
                        sum = el._1()+ sum;
                        if (el._2() > max){
                            max = el._2();
                        }
                    }

                    return new Tuple2<>(sum,max);
                });
        numwords = count.count();
        System.out.println("Number of distinct words in the documents = " + numwords);

        long N_max = -1;

        for(Tuple2<String,Tuple2<Long, Long>> tuple:count1.sortByKey().collect()) {
            if (tuple._2()._2() > N_max){
                N_max=tuple._2()._2();
            }
        }
        /*Tuple2<String,Tuple2<Long,Long>> massimo = count1.max((Tuple2<String,Tuple2<Long,Long>> a, Tuple2<String,Tuple2<Long,Long>> b)->
            a._2()._1().compareTo(b._2()._2()));*/

        Tuple2<String,Tuple2<Long,Long>> massimo2 = count1.reduce((c1,c2)->(c1._2()._1()>c2._2()._1()?c1:c2));


        System.out.println("N_max= "+N_max);
        System.out.println("Most frequent frequent class = "+massimo2._1()+" "+ massimo2._2()._1());

       /* count = docs
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
                    while (wc.hasNext()) {
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
*/
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // COMPUTE AVERAGE WORD LENGTH
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

       /* int avgwordlength = count
                .map((tuple) -> tuple._1().length())
                .reduce((x, y) -> x + y);
        System.out.println("Average word length = " + avgwordlength / numwords);*/
    }

}
