package com.program

object DataExploer {

  def main(args: Array[String]): Unit = {
    var spark = CommonObject.getHead("DataExploer")
    // 1. 获取原数据
    var data_train_source = spark.read.table("human.aug_train")
    //    data_train_source.show(10);
    //2. 提取需要的字段
    var df = data_train_source.select("target","city","city_development_index","gender",
      "relevent_experience","enrolled_university","education_level","major_discipline"
      ,"experience","company_size","company_type","last_new_job","training_hours")

    df.show(10)
  }
}
