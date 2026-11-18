from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pathlib import Path


#start spark 


spark = (
    SparkSession.builder
    .appName("NYC-Taxi-Silver")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.parquet.enableDictionary", "false")
    .getOrCreate()
)


spark.conf.set(
    "spark.sql.session.timeZone",
    "America/New_York"
)

print(
    "Spark timezone:",
    spark.conf.get("spark.sql.session.timeZone")
)




# Find all 12 Parquet files
files = sorted(
    Path("data/bronze").glob("yellow_tripdata_*.parquet")
)

# Read the first month
df = spark.read.parquet(str(files[0]))

# Combine the remaining months
for file in files[1:]:
    df_month = spark.read.parquet(str(file))

    df = df.unionByName(
        df_month,
        allowMissingColumns=True
    )

print("Months combined:", len(files))


#calculate the trip duration in second 
df = df.withColumn(
    "duration_seconds",
    F.expr(
        "timestampdiff(SECOND, tpep_pickup_datetime, tpep_dropoff_datetime)"
    )
)


df.select(
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "duration_seconds",
).show(5, truncate=False)


df_valid_dates = df.filter(
    (F.col("tpep_pickup_datetime") >= "2025-10-01") &
    (F.col("tpep_pickup_datetime") < "2026-10-01")
)

print("Trips after date filtering:", df_valid_dates.count())



#detect zero or negative trip durations
df_invalid_durations = df_valid_dates.filter(
  
        F.col("duration_seconds") <= 0
    )
print(
    "trips with invalid durations ",
    df_invalid_durations.count()
)



df_valid_dates.select(
    F.when(F.col("duration_seconds").isNull(), "NULL")
     .when(F.col("duration_seconds") < 0, "Negative")
     .when(F.col("duration_seconds") == 0, "Zero")
     .otherwise("Positive")
     .alias("duration_status")
).groupBy("duration_status").count().show()


#keep only trips with positive durations
df_cleaned = df_valid_dates.filter(
    F.col("duration_seconds") > 0
)

print("Trips after duration cleaning:", df_cleaned.count())



# Save the cleaned DataFrame to the Silver layer
df_cleaned.write \
    .mode("overwrite") \
    .parquet("data/silver/trips_stage1")

print("Silver Stage 1 saved successfully!")



spark.stop()