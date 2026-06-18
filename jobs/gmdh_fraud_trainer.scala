import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.writePretty
import org.json4s.NoTypeHints
import java.io.PrintWriter
import java.io.File

/**
 * GMDH Fraud Trainer (Model A)
 *
 * 2-layer self-organizing polynomial model for fraud detection.
 * Inputs: semantic_risk (from Bedrock), velocity_1h, proxy_score, amount_deviation
 * Output: fraud probability (0-1)
 *
 * Architecture mirrors Model B (System Efficiency) but trained on fraud features.
 */
object FraudTrainer {
  implicit val formats = Serialization.formats(NoTypeHints)

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("GMDH_Fraud_Trainer")
      .master("local[*]")
      .getOrCreate()

    try {
      val csvPath = if (args.nonEmpty) args(0) else "data/fraud_transactions.csv"
      val outputPath = if (args.length > 1) args(1) else "data/fraud_model_coeffs.json"

      val df = spark.read.option("header", "true").option("inferSchema", "true").csv(csvPath)

      val ds = df.select(
        col("semantic_risk").as("x1"),
        (col("velocity_1h") / 50.0).as("x2"),
        col("proxy_score").as("x3"),
        col("amount_deviation").as("x4"),
        col("is_fraud").as("label")
      ).na.fill(0.0)

      val Array(train, valid) = ds.randomSplit(Array(0.7, 0.3), seed = 42)
      val colNames = Array("x1", "x2", "x3", "x4")

      // Layer 1: All C(4,2) = 6 pairwise nodes
      val nodes = for {
        i <- 0 until colNames.length
        j <- (i + 1) until colNames.length
      } yield {
        val cI = colNames(i)
        val cJ = colNames(j)
        val dft = train.withColumn("interaction", col(cI) * col(cJ))
        val ass = new VectorAssembler().setInputCols(Array(cI, cJ, "interaction")).setOutputCol("f")
        val model = new LinearRegression().setLabelCol("label").setFeaturesCol("f").fit(ass.transform(dft))
        val vt = valid.withColumn("interaction", col(cI) * col(cJ))
        val rmse = model.evaluate(ass.transform(vt)).rootMeanSquaredError
        (s"node_${cI}_${cJ}", rmse, model.intercept, model.coefficients.toArray, cI, cJ)
      }

      // Selection: top 2 by lowest RMSE
      val winners = nodes.sortBy(_._2).take(2)
      val w1 = winners(0)
      val w2 = winners(1)

      println(s"Layer 1 winners: ${w1._1} (RMSE=${w1._2}), ${w2._1} (RMSE=${w2._2})")

      // Layer 2: Master node combining z1, z2
      def calc(df: DataFrame, w: (String, Double, Double, Array[Double], String, String), out: String): DataFrame = {
        df.withColumn(out, lit(w._3) + col(w._5) * w._4(0) + col(w._6) * w._4(1) + col(w._5) * col(w._6) * w._4(2))
      }

      val l2 = calc(calc(train, w1, "z1"), w2, "z2").withColumn("interaction", col("z1") * col("z2"))
      val masterModel = new LinearRegression().setLabelCol("label").setFeaturesCol("f").fit(
        new VectorAssembler().setInputCols(Array("z1", "z2", "interaction")).setOutputCol("f").transform(l2)
      )

      // Export as flat JSON for fast inference
      val export = Map(
        "beta0" -> masterModel.intercept,
        "betas" -> masterModel.coefficients.toArray.toList,
        "layer1" -> List(
          Map("node" -> w1._1, "intercept" -> w1._3, "coeffs" -> w1._4.toList, "inputs" -> List(w1._5, w1._6)),
          Map("node" -> w2._1, "intercept" -> w2._3, "coeffs" -> w2._4.toList, "inputs" -> List(w2._5, w2._6))
        ),
        "timestamp" -> System.currentTimeMillis()
      )

      val pw = new PrintWriter(new File(outputPath))
      pw.write(writePretty(export))
      pw.close()

      println(s"✅ Fraud Model A saved to: $outputPath")

    } finally {
      spark.stop()
    }
  }
}

FraudTrainer.main(Array("data/fraud_transactions.csv", "data/fraud_model_coeffs.json"))
