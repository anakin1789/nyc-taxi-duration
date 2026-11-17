from pyspark.sql import SparkSession
from pyspark.sql import functions as F


#start spark 

spark = (
    SparkSession.builder
    .appName("NYC-Taxi-Silver")
    .master("local[*]")
    .getOrCreate()
)


spark.conf.set("spark.sql.session.timeZone", "UTC")


#Read the bronze dataset
df = spark.read.csv(
    "data/bronze/training/train.csv",
    header=True,
    inferSchema=True
)

#Identify invalid trip durations
df_invalid = df.filter(
    F.col("trip_duration") <= 0
)

print("Invalid trip:", df_invalid.count())

#calculate duration using the timestamps
df_check = df.withColumn(
    "calculated_duration",
    F.unix_timestamp("dropoff_datetime") 
    - F.unix_timestamp("pickup_datetime")
)

#find trips where duration dont match 

df_mismatch = df_check.filter(
F.col("calculated_duration") != F.col("trip_duration")
)



print("Inconsistent trip durations:", df_mismatch.count())




df_mismatch.withColumn("difference_seconds",
 F.col("calculated_duration") - F.col("trip_duration")
).groupBy("difference_seconds")\
    .count()\
    .orderBy(F.desc("count"))\
    .show(10)    


df_mismatch.select(
    "pickup_datetime",
    "dropoff_datetime",
    "trip_duration",
    "calculated_duration"
).show(10, truncate=False)



df_mismatch.groupBy(
    F.to_date("dropoff_datetime").alias("dropoff_date")
).count().show()





print(
    "Spark timezone:",
    spark.conf.get("spark.sql.session.timeZone")
)


df_invalid_coordinates = df.filter(
    (F.col("pickup_latitude") == 0) |
    (F.col("pickup_longitude") == 0) |
    (F.col("dropoff_latitude") == 0) |
    (F.col("dropoff_longitude") == 0)
)

print(
    "Trips with zero coordinates:",
    df_invalid_coordinates.count()
)





spark.stop()