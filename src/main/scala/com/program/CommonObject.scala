package com.program

import org.apache.spark.sql.SparkSession

object CommonObject {
  def getHead(appName:String):SparkSession = {
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("appName")
      .enableHiveSupport()
      .getOrCreate()

    spark.sparkContext.setLogLevel("Error")
    return spark
  }
}
