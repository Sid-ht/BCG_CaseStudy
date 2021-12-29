from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType, DecimalType, DateType
from pyspark.sql.functions import col, count,expr, max, dense_rank, desc
from pyspark.sql.window import Window
spark = SparkSession.builder.appName("CaseAnalysis").master("local[*]").getOrCreate()

schema = StructType([(StructField("CRASH_ID", IntegerType())),
                     (StructField("UNIT_NBR", IntegerType())),
                     (StructField("PRSN_NBR", IntegerType())),
                     (StructField("CHARGE", StringType())),
                     (StructField("CITATION_NBR", StringType()))])

charges_use_df = spark.read.format("csv").schema(schema).option("header", "true").load("dbfs:/FileStore/shared_uploads/siddharthsinha.28@gmail.com/Charges_use.csv")

charges_use_df.show(10,truncate=False)

person_schema = StructType([(StructField("CRASH_ID", IntegerType())),
(StructField("UNIT_NBR", IntegerType())),
(StructField("PRSN_NBR", IntegerType())),
(StructField("PRSN_TYPE_ID", StringType())),
(StructField("PRSN_OCCPNT_POS_ID", StringType())),
(StructField("PRSN_INJRY_SEV_ID", StringType())),
(StructField("PRSN_AGE", IntegerType())),
(StructField("PRSN_ETHNICITY_ID", StringType())),
(StructField("PRSN_GNDR_ID", StringType())),
(StructField("PRSN_EJCT_ID", StringType())),
(StructField("PRSN_REST_ID", StringType())),
(StructField("PRSN_AIRBAG_ID", StringType())),
(StructField("PRSN_HELMET_ID", StringType())),
(StructField("PRSN_SOL_FL", StringType())),
(StructField("PRSN_ALC_SPEC_TYPE_ID", StringType())),
(StructField("PRSN_ALC_RSLT_ID", StringType())),
(StructField("PRSN_BAC_TEST_RSLT",DecimalType())),
(StructField("PRSN_DRG_SPEC_TYPE_ID", StringType())),
(StructField("PRSN_DRG_RSLT_ID", StringType())),
(StructField("DRVR_DRG_CAT_1_ID", StringType())),
(StructField("PRSN_DEATH_TIME", TimestampType())),
(StructField("INCAP_INJRY_CNT", IntegerType())),
(StructField("NONINCAP_INJRY_CNT", IntegerType())),
(StructField("POSS_INJRY_CNT", IntegerType())),
(StructField("NON_INJRY_CNT", IntegerType())),
(StructField("UNKN_INJRY_CNT", IntegerType())),
(StructField("TOT_INJRY_CNT", IntegerType())),
(StructField("DEATH_CNT", IntegerType())),
(StructField("DRVR_LIC_TYPE_ID", StringType())),
(StructField("DRVR_LIC_STATE_ID", StringType())),
(StructField("DRVR_LIC_CLS_ID", StringType())),
(StructField("DRVR_ZIP", IntegerType()))])

person_df = spark.read.format("csv").schema(person_schema).option("header", "true").load("dbfs:/FileStore/shared_uploads/siddharthsinha.28@gmail.com/Primary_Person_use.csv")

#person_df.show(10,truncate=False)

male_person_df = person_df.filter(col("PRSN_GNDR_ID") =='MALE')
male_person_df.count()  #96782

unit_case_df = spark.read.format("csv").option("header", "true").option("inferSchema","true").load("dbfs:/FileStore/shared_uploads/siddharthsinha.28@gmail.com/Units_use.csv")

#unit_case_df.show(10,truncate=False)

#unit_case_df.select("VEH_BODY_STYL_ID").distinct().show(truncate=False)

unit_case_df.filter(unit_case_df.VEH_BODY_STYL_ID.like('%MOTORCYCLE%')).count()

female_person_df = person_df.filter(col("PRSN_GNDR_ID") =='FEMALE')
female_person_df.groupBy("DRVR_LIC_STATE_ID").agg((count("CRASH_ID")).alias("count")).sort("count",  ascending=False ).show(1,truncate=False)

+-----------------+-----+
|DRVR_LIC_STATE_ID|count|
+-----------------+-----+
|Texas            |53319|
+-----------------+-----+

most_vehicle_crashes = unit_case_df.groupBy("VEH_MAKE_ID").agg(count("DEATH_CNT").alias("death_count")).orderBy(col("death_count"))
#most_vehicle_crashes.show()

windowSpec = Window.orderBy(desc("death_count"))
most_vehicle_crashes.withColumn("rank", dense_rank().over(windowSpec)).filter((col("rank")>=5) &(col("rank")<=15)).drop("rank").show(truncate=False)

+------------+-----------+
|VEH_MAKE_ID |death_count|
+------------+-----------+
|NISSAN      |10964      |
|HONDA       |10460      |
|NA          |6234       |
|GMC         |5044       |
|JEEP        |4170       |
|HYUNDAI     |3859       |
|KIA         |3186       |
|CHRYSLER    |3144       |
|FREIGHTLINER|2861       |
|MAZDA       |2585       |
|PONTIAC     |2144       |
+------------+-----------+


join_expr = unit_case_df.CRASH_ID == person_df.CRASH_ID

join_df = unit_case_df.join(person_df, join_expr, "inner").select(unit_case_df.CRASH_ID,unit_case_df.VEH_BODY_STYL_ID,person_df.PRSN_ETHNICITY_ID)

top_body_styles = join_df.groupBy(join_df.VEH_BODY_STYL_ID, join_df.PRSN_ETHNICITY_ID).count().orderBy(col("count").desc())

window = Window.partitionBy("VEH_BODY_STYL_ID").orderBy(col("count").desc())

top_group = top_body_styles.withColumn("rank", dense_rank().over(window)).filter("rank= 1").drop("rank", "count")

top_group.show()

+--------------------+-----------------+
|    VEH_BODY_STYL_ID|PRSN_ETHNICITY_ID|
+--------------------+-----------------+
|           AMBULANCE|            WHITE|
|                 BUS|         HISPANIC|
|      FARM EQUIPMENT|            WHITE|
|          FIRE TRUCK|            WHITE|
|          MOTORCYCLE|            WHITE|
|                  NA|            WHITE|
|NEV-NEIGHBORHOOD ...|            WHITE|
|        NOT REPORTED|         HISPANIC|
|        NOT REPORTED|            WHITE|
|OTHER  (EXPLAIN I...|            WHITE|
|PASSENGER CAR, 2-...|            WHITE|
|PASSENGER CAR, 4-...|            WHITE|
|              PICKUP|            WHITE|
|    POLICE CAR/TRUCK|            WHITE|
|   POLICE MOTORCYCLE|         HISPANIC|
|SPORT UTILITY VEH...|            WHITE|
|               TRUCK|            WHITE|
|       TRUCK TRACTOR|            WHITE|
|             UNKNOWN|            WHITE|
|                 VAN|            WHITE|
+--------------------+-----------------+
only showing top 20 rows






