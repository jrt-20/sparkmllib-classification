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
import org.apache.zookeeper.server.ServerMetrics.getMetrics


object Process {
  def main(args: Array[String]): Unit = {

    var spark = CommonObject.getHead("RLModel")
    // 1.获取原数据
    var data_test = spark.read.table("human.aug_test")
//    data_test.show(10);

    var data_train_source = spark.read.table("human.aug_train")
//    data_train_source.show(10);
    //2. 提取需要的字段
    var df = data_train_source.select("target","city","city_development_index","gender",
      "relevent_experience","enrolled_university","education_level","major_discipline"
    ,"experience","company_size","company_type","last_new_job","training_hours")

//    df.show(10)
    //3.提取特征工程
      var labeledPointRDD = df.rdd.map(row=>{
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

        LabeledPoint(label,Vectors.dense(city,city_development_index,gender
        ,relevent_experience,enrolled_university,education_level,major_discipline,
          experience,company_size,company_type,last_new_job,training_hours))
      })
    //4.持久化
    labeledPointRDD.cache()
    // 5.分割训练集，测试集
    var Array(trainRDD,testRDD) = labeledPointRDD.randomSplit(Array(0.8,0.2))

    //6.model svm
    var svmModel:SVMModel = SVMWithSGD.train(trainRDD,100)

    var svmPredictAndActualRDD:RDD[(Double,Double)] = testRDD.map{
      case LabeledPoint(label,features)=>(svmModel.predict(features),label)
    }
    // 评价roc
    var svmMetrics = new BinaryClassificationMetrics(svmPredictAndActualRDD)
    var svmRoc = svmMetrics.areaUnderROC()

    val nbTotalCorrect = testRDD.map { point =>
      if (svmModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val numData = testRDD.count()
    val svmpre = nbTotalCorrect/numData

    println("支持向量机 准确度 = "+svmpre)
    println("支持向量机 ROC = "+svmRoc)//接近0.5预测效果好，接近1预测效果不好

    println("")
    // model lr
    var lr = new LogisticRegressionWithLBFGS().setNumClasses(2)

    var lrModel:LogisticRegressionModel = lr.run(trainRDD)
    var lrPredictAndActualRDD:RDD[(Double,Double)] = testRDD.map{
      case LabeledPoint(label,features)=>(lrModel.predict(features),label)
    }
    var lrMetrics = new BinaryClassificationMetrics(lrPredictAndActualRDD)
    var lrRoc = lrMetrics.areaUnderROC()

    val LrnbTotalCorrect = testRDD.map { point =>
      if (lrModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val LrnumData = testRDD.count()
    val LrAuc = LrnbTotalCorrect/LrnumData
    println("逻辑回归 准确度 = "+LrAuc)
    println("逻辑回归 ROC = "+lrRoc)
    println("")

    //model 决策树
    var dtModel:DecisionTreeModel = DecisionTree.trainClassifier(trainRDD,2
    ,Map[Int,Int](),"gini",6,2)

    var dtPredictAndActualRDD:RDD[(Double,Double)] = testRDD.map{
      case LabeledPoint(label,features) =>(dtModel.predict(features),label)
    }
    val dtMetrics = new BinaryClassificationMetrics(dtPredictAndActualRDD)
    val dtRoc = dtMetrics.areaUnderROC()

    val DtnbTotalCorrect = testRDD.map { point =>
      if (dtModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val DtnumData = testRDD.count()
    val Dtauc = DtnbTotalCorrect/DtnumData
    println("决策树 准确度 = "+Dtauc)
    println("决策树 ROC = "+dtRoc)
    println("")

    //model 朴素贝叶斯
    var nbModel = NaiveBayes.train(trainRDD,1.0)
    var nbPredictAndActualRDD:RDD[(Double,Double)] = testRDD.map{
      case LabeledPoint(label,features)=>(nbModel.predict(features),label)
    }
    var nbMetrics = new BinaryClassificationMetrics(nbPredictAndActualRDD)
    val nbRoc = nbMetrics.areaUnderROC()

    val NbnbTotalCorrect = testRDD.map { point =>
      if (nbModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val NbnumData = testRDD.count()
    var nbauc = NbnbTotalCorrect/NbnumData
    println("朴素贝叶斯 准确度 = "+nbauc)
    println("朴素贝叶斯 ROC = "+nbRoc)

    println("")
    spark.stop()
  }
}
