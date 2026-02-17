import json
import os
import textwrap

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


def get_config_path():
    return Variable.get("gmdh_config_path",
                        default_var="gmdh_secrets.json")


CONFIG_PATH = get_config_path()

with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

SET_PROFILES_SCRIPT = CONFIG['paths']['set_profiles']
AWS_PROFILE = CONFIG['aws']['profile']
BUCKET_NAME = CONFIG['aws']['s3_bucket_name']
ACCOUNT_ID = CONFIG['aws']['account_id']
FULL_S3_PATH = f"s3://{BUCKET_NAME}-{ACCOUNT_ID}/temp/E2E_TEST_GMDH_V1/"
SCALA_TMP_TRAIN = "/tmp/GmdhTrain.scala"
SCALA_TMP_SIM = "/tmp/GmdhSim.scala"


def write_scripts_func():
    with open(SCALA_TMP_TRAIN, "w") as f: f.write(TRAIN_CODE)
    with open(SCALA_TMP_SIM, "w") as f: f.write(SIM_CODE)

    BUCKET_NAME = CONFIG['aws']['s3_bucket_name']
    ACCOUNT_ID = CONFIG['aws']['account_id']
    S3_PATH = f"s3://{BUCKET_NAME}-{ACCOUNT_ID}/temp/E2E_TEST_GMDH_V1/"

    launcher_content = textwrap.dedent(f"""
        #!/bin/bash
        set +x
        sh {SET_PROFILES_SCRIPT} > /dev/null 2>&1
        export AWS_PROFILE={AWS_PROFILE}

        if [ "$1" == "cleanup" ]; then
            echo "🧹 Cleaning up S3 location..."
            aws s3 rm "{S3_PATH}" --recursive --profile {AWS_PROFILE} > /dev/null 2>&1
            echo "✅ S3 Cleanup finished."
        else
            spark-shell --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.profile.ProfileCredentialsProvider \
                        --conf spark.hadoop.fs.s3a.aws.profile={AWS_PROFILE} \
                        -i "$1"
        fi
    """).strip()

    with open("/tmp/spark_launcher.sh", "w") as f:
        f.write(launcher_content)

    os.chmod("/tmp/spark_launcher.sh", 0o755)


# --- TRAIN CODE ---
TRAIN_CODE = textwrap.dedent(r"""
    import org.apache.hadoop.fs.{FileSystem, Path}
    import org.apache.spark.sql.{DataFrame, SparkSession}
    import org.apache.spark.sql.functions._
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.regression.LinearRegression
    import org.json4s.jackson.JsonMethods._
    import org.json4s.jackson.Serialization
    import org.json4s.jackson.Serialization.writePretty
    import org.json4s.{NoTypeHints, DefaultFormats}
    import java.io.PrintWriter
    import java.net.URI

    object GmdhTrainer {
      implicit val formats = Serialization.formats(NoTypeHints)
      def main(args: Array[String]): Unit = {
        val configPath = if (args.nonEmpty) args(0) else "REPLACE_WITH_CONFIG_PATH"
        val configStr = scala.io.Source.fromFile(configPath).mkString
        val conf = parse(configStr)
        
        val m2 = (conf \ "user_context" \ "m2_repository").extract[String]
        val root = (conf \ "user_context" \ "project_root").extract[String]
        val jars = s"$m2/${(conf \ "paths" \ "jackson_databind").extract[String]}:$m2/${(conf \ "paths" \ "jackson_core").extract[String]}"
        val accId = (conf \ "aws" \ "account_id").extract[String]
        val env = (conf \ "aws" \ "env").extract[String]
        val bucketBase = (conf \ "aws" \ "s3_bucket_name").extract[String]

        val spark = SparkSession.builder.appName("GmdhTrain").master("local[*]")
          .config("spark.driver.extraClassPath", jars)

          .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.profile.ProfileCredentialsProvider")
          .config("spark.hadoop.fs.s3a.aws.profile", env)
          .getOrCreate()
          
        // Force refresh hadoop config
        spark.sparkContext.hadoopConfiguration.set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.profile.ProfileCredentialsProvider")
        spark.sparkContext.hadoopConfiguration.set("fs.s3a.aws.profile", env)

        try {
          val s3Bucket = s"$bucketBase-$accId"
          val s3Path = s"s3a://$s3Bucket/temp/E2E_TEST_GMDH_V1/model.json"

          val csvSource = s"$root/${(conf \ "paths" \ "data_source").extract[String]}"
          val df = spark.read.option("header","true").option("inferSchema","true").csv(csvSource)
          
          val ds = df.select(
            (col("amazon_api_latency")/2000.0).as("x1"), 
            col("cs_auth_status").cast("double").as("x2"), 
            (col("system_cpu_load")/100.0).as("x3"), 
            col("architecture_efficiency").as("label")
          ).na.fill(0.0)
          
          val Array(tA, vB) = ds.randomSplit(Array(0.7, 0.3))
          val colNames = Array("x1", "x2", "x3")

          val nodes = for { 
            i <- 0 until colNames.length
            j <- (i + 1) until colNames.length 
          } yield {
            val cI: String = colNames(i)
            val cJ: String = colNames(j)
            val dft = tA.withColumn("interaction", col(cI)*col(cJ))
            val ass = new VectorAssembler().setInputCols(Array(cI, cJ, "interaction")).setOutputCol("f")
            val model = new LinearRegression().setLabelCol("label").setFeaturesCol("f").fit(ass.transform(dft))
            val vBTransformed = vB.withColumn("interaction", col(cI)*col(cJ))
            (s"node_${cI}_${cJ}", model.evaluate(ass.transform(vBTransformed)).rootMeanSquaredError, model.intercept, model.coefficients.toArray, cI, cJ)
          }
          
          val winners = nodes.sortBy(_._2).take(2)
          val w1 = winners(0)
          val w2 = winners(1)

          def calc(df: DataFrame, w: (String, Double, Double, Array[Double], String, String), out: String): DataFrame = {
            df.withColumn(out, lit(w._3) + col(w._5)*w._4(0) + col(w._6)*w._4(1) + col(w._5)*col(w._6)*w._4(2))
          }

          val l2t = calc(calc(tA, w1, "z1"), w2, "z2").withColumn("interaction", col("z1")*col("z2"))
          val finalModel = new LinearRegression().setLabelCol("label").setFeaturesCol("f").fit(
            new VectorAssembler().setInputCols(Array("z1","z2","interaction")).setOutputCol("f").transform(l2t)
          )

          val modelExport = Map("layers" -> List(
            Map("node_z1" -> Map("intercept" -> w1._3, "coeffs" -> w1._4), "node_z2" -> Map("intercept" -> w2._3, "coeffs" -> w2._4)),
            Map("master_node" -> Map("intercept" -> finalModel.intercept, "coeffs" -> finalModel.coefficients.toArray))
          ))

          val fs = FileSystem.get(new URI(s3Path), spark.sparkContext.hadoopConfiguration)
          val pw = new PrintWriter(fs.create(new Path(s3Path)))
          pw.write(writePretty(modelExport)); pw.close()
          println(s"Model saved to S3: ${s3Path.split("/").last}")
        } finally { spark.stop() }
      }
    }
    GmdhTrainer.main(Array("REPLACE_WITH_CONFIG_PATH"))
""").strip().replace("REPLACE_WITH_CONFIG_PATH", CONFIG_PATH)

# --- SIMULATION CODE ---
SIM_CODE = textwrap.dedent(r"""
    import org.apache.hadoop.fs.{FileSystem, Path}
    import org.apache.spark.sql.SparkSession
    import org.json4s.jackson.JsonMethods._
    import java.net.URI

    object GmdhSimulator {
      implicit val formats = org.json4s.DefaultFormats
      def main(args: Array[String]): Unit = {
        val configPath = if (args.nonEmpty) args(0) else "REPLACE_WITH_CONFIG_PATH"
        val configStr = scala.io.Source.fromFile(configPath).mkString
        val conf = parse(configStr)
        val accId = (conf \ "aws" \ "account_id").extract[String]
        val env = (conf \ "aws" \ "env").extract[String]
        val bucketBase = (conf \ "aws" \ "s3_bucket_name").extract[String]

        val spark = SparkSession.builder.appName("GmdhSim").master("local[*]")
          .config("spark.hadoop.fs.s3a.aws.profile", env)
          .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.profile.ProfileCredentialsProvider")
          .getOrCreate()
          
        spark.sparkContext.hadoopConfiguration.set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.profile.ProfileCredentialsProvider")
        spark.sparkContext.hadoopConfiguration.set("fs.s3a.aws.profile", env)

        try {
          val s3Bucket = s"$bucketBase-$accId"
          val s3Path = s"s3a://$s3Bucket/temp/E2E_TEST_GMDH_V1/model.json"
          
          val fs = FileSystem.get(new URI(s3Path), spark.sparkContext.hadoopConfiguration)
          val jsonStr = scala.io.Source.fromInputStream(fs.open(new Path(s3Path))).mkString
          val json = parse(jsonStr)
          
          val layers = (json \ "layers").extract[List[Map[String, Map[String, Any]]]]
          val l1 = layers.head
          val master = layers(1)("master_node")

          def getV(d: Map[String, Any], k: String): Double = d(k).toString.toDouble
          def getC(d: Map[String, Any]): List[Double] = d("coeffs").asInstanceOf[List[Double]]

          println("-" * 85 + "\n| STEP | AMZ LAT | CS | CPU | EFFICIENCY | STATUS |\n" + "-" * 85)
          val rnd = new scala.util.Random()
          (1 to 15).foreach { i =>
            val (lat, cs, cpu) = (300 + rnd.nextInt(1500), if(i==7) 0.0 else 1.0, 20 + rnd.nextInt(70))
            val (x1, x2, x3) = (lat/2000.0, cs, cpu/100.0)
            val c1 = getC(l1("node_z1")); val z1 = getV(l1("node_z1"), "intercept") + x1*c1(0) + x3*c1(1) + x1*x3*c1(2)
            val c2 = getC(l1("node_z2")); val z2 = getV(l1("node_z2"), "intercept") + x1*c2(0) + x2*c2(1) + x1*x2*c2(2)
            val cm = getC(master); val eff = getV(master, "intercept") + z1*cm(0) + z2*cm(1) + z1*z2*cm(2)
            val st = if(eff > 0.75) "✅ OK" else if(eff > 0.45) "⚠️ WARN" else "🚨 CRIT"
            println(f"| $i%4d | $lat%5d ms | $cs%2.1f | $cpu%2d%% | ${eff*100}%9.2f%% | $st%-6s |")
          }
        } finally { spark.stop() }
      }
    }
    GmdhSimulator.main(Array("REPLACE_WITH_CONFIG_PATH"))
""").strip().replace("REPLACE_WITH_CONFIG_PATH", CONFIG_PATH)

with DAG(
        'gmdh_predictive_engine_it',
        default_args={'owner': 'airflow-spark-dev', 'start_date': days_ago(1)},
        schedule_interval=None,
        catchup=False
) as dag:
    t1 = PythonOperator(
        task_id='prepare_scripts',
        python_callable=write_scripts_func
    )

    t2 = BashOperator(
        task_id='train_model',
        bash_command=f"/tmp/spark_launcher.sh {SCALA_TMP_TRAIN}"
    )

    t3 = BashOperator(
        task_id='live_simulation_monitoring',
        bash_command=f"/tmp/spark_launcher.sh {SCALA_TMP_SIM}"
    )

    t4 = BashOperator(
        task_id='cleanup',
        bash_command=textwrap.dedent(f"""
        /tmp/spark_launcher.sh cleanup
        rm -f {SCALA_TMP_TRAIN} {SCALA_TMP_SIM} /tmp/spark_launcher.sh
    """),
        trigger_rule='all_done'
    )

    t1 >> t2 >> t3 >> t4
