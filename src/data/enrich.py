
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("NYC-Taxi-Enrichment")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.parquet.enableDictionary", "false")
    .config("spark.sql.session.timeZone", "America/New_York")
    .getOrCreate()
)

# Read Silver Stage 2
df = spark.read.parquet("data/silver/trips_stage2")

# Read taxi zones
df_zones = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("data/reference/taxi_zone_lookup.csv")
)

# Prepare pickup zone information
pickup_zones = df_zones.select(
    F.col("LocationID").alias("pickup_id"),
    F.col("Borough").alias("pickup_borough"),
    F.col("Zone").alias("pickup_zone")
)

# Join taxi trips with pickup zones
df_enriched = df.join(
    F.broadcast(pickup_zones),
    df["PULocationID"] == pickup_zones["pickup_id"],
    "left"
).drop("pickup_id")

df_enriched.select(
    "PULocationID",
    "pickup_borough",
    "pickup_zone"
).show(10, truncate=False)


# Prepare destination zone information
dropoff_zones = df_zones.select(
    F.col("LocationID").alias("dropoff_id"),
    F.col("Borough").alias("dropoff_borough"),
    F.col("Zone").alias("dropoff_zone")
)

# Join destination locations
df_enriched = df_enriched.join(
    F.broadcast(dropoff_zones),
    df_enriched["DOLocationID"] == dropoff_zones["dropoff_id"],
    "left"
).drop("dropoff_id")


# Flag extremely large distances
df_enriched = df_enriched.withColumn(
    "extreme_distance",
    F.col("trip_distance") > 1000
)

# Flag unrealistic average speeds
df_enriched = df_enriched.withColumn(
    "unrealistic_speed",
    (
        F.col("trip_distance") * 3600 /
        F.col("duration_seconds")
    ) > 100
)


# Display the new columns
print("\n=== EXAMPLES OF SUSPICIOUS TRIPS ===")

df_enriched.filter(
    F.col("extreme_distance") |
    F.col("unrealistic_speed")
).select(
    "trip_distance",
    "duration_seconds",
    "extreme_distance",
    "unrealistic_speed"
).show(10, truncate=False)


# Count records with each quality flag
print("\n=== DATA QUALITY FLAG COUNTS ===")

df_enriched.select(
    F.sum(
        F.col("extreme_distance").cast("int")
    ).alias("extreme_distance_count"),

    F.sum(
        F.col("unrealistic_speed").cast("int")
    ).alias("unrealistic_speed_count")
).show()


# Check the results
print("\n=== PICKUP AND DESTINATION ZONES ===")

df_enriched.select(
    "PULocationID",
    "pickup_zone",
    "DOLocationID",
    "dropoff_zone"
).show(10, truncate=False)


# Validate joins
print("\n=== JOIN VALIDATION ===")

df_enriched.select(
    F.count("*").alias("total_trips"),

    F.sum(
        F.col("pickup_zone").isNull().cast("int")
    ).alias("unmatched_pickup"),

    F.sum(
        F.col("dropoff_zone").isNull().cast("int")
    ).alias("unmatched_dropoff")
).show()


# Save enriched Silver dataset
df_enriched.write \
    .mode("overwrite") \
    .parquet("data/silver/trips_stage3")

print("Silver Stage 3 saved successfully!")

spark.stop()
