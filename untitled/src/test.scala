import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

object test {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .getOrCreate()
    import spark.implicits._
    val dat=spark.read.options(Map(("delimiter", "|"), ("header", "true"))).csv("/Users/jingjiang/Documents/work/testdata/day02/doc_class.dat")
    val data=dat.select("typenameid","myapp_word_all")

    val labelIndexer = new StringIndexer()
      .setInputCol("typenameid")
      .setOutputCol("indexedLabel")
      .fit(data)

    val tokenizer = new Tokenizer()
      .setInputCol("myapp_word_all")
      .setOutputCol("words")
    //tokenizer.transform(training.select("myapp_word_all")).show(false)
    val hashingTF= new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("indexedFeatures")
    //training.show(false)

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer,hashingTF,labelIndexer, dt, labelConverter))

    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.show(100,false)
    predictions.select("predictedLabel", "indexedLabel", "indexedFeatures").show(100,false)




    // println(trainingData.describe())
    // println(testData.describe())
    //val numClasses = data.select("typenameid").distinct().count()





  }
}
