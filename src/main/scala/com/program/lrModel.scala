package com.program

import com.program.CommonObject
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.classification._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD

object lrModel {

  def main(args: Array[String]): Unit = {
    var spark = CommonObject.getHead("lrModel")
    // 1.获取原数据
    var data_test = spark.read.table("human.aug_test")
    //    data_test.show(10);

    var data_train_source = spark.read.table("human.aug_train")
    //    data_train_source.show(10);
    //2. 提取需要的字段
    var df = data_train_source.select("target", "city", "city_development_index", "gender",
      "relevent_experience", "enrolled_university", "education_level", "major_discipline"
      , "experience", "company_size", "company_type", "last_new_job", "training_hours")

    //    df.show(10)
    //3.提取特征工程
    var labeledPointRDD = df.rdd.map(row => {
      var label = row.getAs[Double]("target")
      var city = row.getAs[Double]("city")
      var city_development_index = row.getAs[Double]("city_development_index")
      var gender = row.getAs[Double]("gender")
      var relevent_experience = row.getAs[Double]("relevent_experience")
      var enrolled_university = row.getAs[Double]("enrolled_university")
      var education_level = row.getAs[Double]("education_level")
      var major_discipline = row.getAs[Double]("major_discipline")
      var experience = row.getAs[Double]("experience")
      var company_size = row.getAs[Double]("company_size")
      var company_type = row.getAs[Double]("company_type")
      var last_new_job = row.getAs[Double]("last_new_job")
      var training_hours = row.getAs[Double]("training_hours")

      LabeledPoint(label, Vectors.dense(city, city_development_index, gender
        , relevent_experience, enrolled_university, education_level, major_discipline,
        experience, company_size, company_type, last_new_job, training_hours))
    })
    //4.持久化
    labeledPointRDD.cache()
    // 5.分割训练集，测试集
    var Array(trainRDD, testRDD) = labeledPointRDD.randomSplit(Array(0.8, 0.2))

    //6.model svm
    var accuracy = 0.0
    var roc = 0.0
    var i = 1
    for (i <- 1 to 5) {
      // model lr
      var lr = new LogisticRegressionWithLBFGS().setNumClasses(2)

      var lrModel: LogisticRegressionModel = lr.run(trainRDD)
      var lrPredictAndActualRDD: RDD[(Double, Double)] = testRDD.map {
        case LabeledPoint(label, features) => (lrModel.predict(features), label)
      }
      var lrMetrics = new BinaryClassificationMetrics(lrPredictAndActualRDD)
      var lrRoc = lrMetrics.areaUnderROC()
      roc = roc+lrRoc

      val LrnbTotalCorrect = testRDD.map { point =>
        if (lrModel.predict(point.features) == point.label) 1 else 0
      }.sum
      val LrnumData = testRDD.count()
      val LrAuc = LrnbTotalCorrect / LrnumData
      accuracy = accuracy + LrAuc
    }
    println("逻辑回归10次平均 ROC = " + roc / 5)
    println("逻辑回归10次平均准确度 = " + accuracy / 5)
  }


}
