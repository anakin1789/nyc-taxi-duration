from pyspark.sql import functions as F

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("NYC-Taxi-Analysis")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)



df_zones = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("data/reference/taxi_zone_lookup.csv")
)

df_zones.printSchema()
df_zones.show(5, truncate=False)




spark.conf.set(
    "spark.sql.session.timeZone",
    "America/New_York"
)

print(
    "Spark timezone:",
    spark.conf.get("spark.sql.session.timeZone")
)


# Read the saved Silver dataset
df = spark.read.parquet("data/silver/trips_stage2")



df_extreme = df.filter(
    F.col("duration_seconds") > 24 * 3600
)

df_extreme.select(
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "duration_seconds",
    "trip_distance",
    "PULocationID",
    "DOLocationID"
).show(10, truncate=False)


# Calculate trip duration percentiles

percentiles = df.approxQuantile(
    "duration_seconds",
    [0.99, 0.995, 0.999],
    0.0001
)

print("99th percentile:", percentiles[0])
print("99.5th percentile:", percentiles[1])
print("99.9th percentile:", percentiles[2])




print(
    "Spark timezone:",
    spark.conf.get("spark.sql.session.timeZone")
)

print(
    "Pickup timestamp type:",
    df.schema["tpep_pickup_datetime"].dataType
)


df.select(
    F.max("duration_seconds").alias("max_duration"),
    F.sum(
        F.when(F.col("duration_seconds") > 3 * 3600, 1)
         .otherwise(0)
    ).alias("trips_over_3_hours"),
    F.sum(
        F.when(F.col("duration_seconds") > 24 * 3600, 1)
         .otherwise(0)
    ).alias("trips_over_24_hours")
).show()




# Columns to investigate
columns = [
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime"
]

# Count missing values in each column
df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in columns
]).show()



df.groupBy(
    F.date_format(
        "tpep_pickup_datetime", "yyyy-MM"
    ).alias("month")
).agg(
    F.count("*").alias("total_trips"),

    F.sum(
        F.col("passenger_count").isNull().cast("int")
    ).alias("missing_passengers")
).orderBy("month").show(20, truncate=False)



df.select(
    F.sum(
        (F.col("trip_distance") < 0).cast("int")
    ).alias("negative_distance"),

    F.sum(
        (F.col("trip_distance") == 0).cast("int")
    ).alias("zero_distance"),

    F.sum(
        (F.col("trip_distance") > 100).cast("int")
    ).alias("over_100_miles")
).show()



# Select trips with zero recorded distance
df_zero = df.filter(
    F.col("trip_distance") == 0
)

# Analyze their duration
df_zero.select(
    F.count("*").alias("total_zero_distance"),

    F.sum(
        (F.col("duration_seconds") > 600).cast("int")
    ).alias("over_10_minutes"),

    F.sum(
        (F.col("duration_seconds") > 1800).cast("int")
    ).alias("over_30_minutes")
).show()




df_zero.select(
    F.sum(
        (F.col("PULocationID") == F.col("DOLocationID")).cast("int")
    ).alias("same_zone"),

    F.sum(
        (F.col("PULocationID") != F.col("DOLocationID")).cast("int")
    ).alias("different_zones")
).show()



df_suspicious = df.filter(
    (F.col("trip_distance") == 0) &
    (F.col("PULocationID") != F.col("DOLocationID"))
)

df_suspicious.select(
    "trip_distance",
    "duration_seconds",
    "fare_amount",
    "PULocationID",
    "DOLocationID"
).show(10, truncate=False)





# Read Silver Stage 3
df = spark.read.parquet("data/silver/trips_stage3")




# Read the entire Silver Stage 3 dataset
df = spark.read.parquet("data/silver/trips_stage3")
"""
# Generate a fingerprint for every row
df_hashes = df.select(
    F.xxhash64(
        F.struct(*df.columns)
    ).alias("row_hash")
)

# Find duplicate fingerprints across all 12 months
duplicates = (
    df_hashes.groupBy("row_hash")
    .count()
    .filter(F.col("count") > 1)
)

print("Duplicate groups in entire dataset:", duplicates.count())
"""


# Batch 3: June to September 2026
df_batch = df.filter(
    (F.col("tpep_pickup_datetime") >= "2026-06-01") &
    (F.col("tpep_pickup_datetime") < "2026-10-01")
)





df_speed = df.withColumn(
    "average_speed_mph",
    F.col("trip_distance") * 3600 /
    F.col("duration_seconds")
)

print(
    "Trips exceeding 100 mph:",
    df_speed.filter(
        F.col("average_speed_mph") > 100
    ).count()
)




df_fast = df_speed.filter(
    F.col("average_speed_mph") > 100
)



df_fast.select(
    F.count("*").alias("total_fast_trips"),

    F.sum(
        (F.col("duration_seconds") < 60).cast("int")
    ).alias("under_1_minute"),

    F.sum(
        (
            (F.col("duration_seconds") >= 60) &
            (F.col("duration_seconds") < 300)
        ).cast("int")
    ).alias("between_1_and_5_minutes"),

    F.sum(
        (F.col("duration_seconds") >= 300).cast("int")
    ).alias("over_5_minutes")
).show()



df_fast.filter(
    F.col("duration_seconds") >= 300
).select(
    "trip_distance",
    "duration_seconds",
    "average_speed_mph",
    "PULocationID",
    "DOLocationID"
).orderBy(
    F.desc("average_speed_mph")
).show(10, truncate=False)





df.select(
    F.sum(
        (F.col("trip_distance") > 1000).cast("int")
    ).alias("trips_over_1000_miles"),

    F.sum(
        (
            (F.col("trip_distance") > 100) &
            (F.col("trip_distance") <= 1000)
        ).cast("int")
    ).alias("trips_between_100_and_1000_miles")
).show()



# Verify the saved data-quality flags
df = spark.read.parquet("data/silver/trips_stage3")

df.select(
    "suspicious_distance",
    "extreme_distance",
    "unrealistic_speed"
).show(10, truncate=False)

df.select(
    F.sum(F.col("extreme_distance").cast("int")).alias("extreme_distance"),
    F.sum(F.col("unrealistic_speed").cast("int")).alias("unrealistic_speed")
).show()


# Count zero-distance trips with different pickup and dropoff zones
df.select(
    F.sum(
        F.col("suspicious_distance").cast("int")
    ).alias("suspicious_distance_count")
).show()




# Read original November 2025 records
df_nov = spark.read.parquet(
    "data/bronze/yellow_tripdata_2025-11.parquet"
)

# Calculate recorded trip duration
df_nov = df_nov.withColumn(
    "duration_check",
    F.expr(
        "timestampdiff(SECOND, tpep_pickup_datetime, tpep_dropoff_datetime)"
    )
)

# Investigate negative durations on DST transition day
df_dst = df_nov.filter(
    (F.to_date("tpep_pickup_datetime") == "2025-11-02") &
    (F.col("duration_check") < 0)
)

print("Negative durations on DST day:", df_dst.count())





# Inspect negative durations on DST transition day
df_dst.select(
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "duration_check"
).show(10, truncate=False)

# Count negative durations within and beyond one hour
df_dst.select(
    F.sum(
        (F.col("duration_check") >= -3600).cast("int")
    ).alias("within_one_hour"),

    F.sum(
        (F.col("duration_check") < -3600).cast("int")
    ).alias("more_than_one_hour")
).show()




spark.stop()
