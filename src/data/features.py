
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Start Spark
spark = (
    SparkSession.builder
    .appName("NYC-Taxi-Gold")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.session.timeZone", "America/New_York")
    .getOrCreate()
)

# Read the Silver Stage 3 dataset
df = spark.read.parquet(
    "data/silver/trips_stage3"
)

# Extract temporal features from pickup time
df_gold = (
    df.withColumn(
        "pickup_hour",
        F.hour("tpep_pickup_datetime")
    )
    .withColumn(
        "pickup_day_of_week",
        F.dayofweek("tpep_pickup_datetime")
    )
)

# Weekend indicator: Saturday or Sunday
df_gold = df_gold.withColumn(
    "is_weekend",
    F.when(
        F.col("pickup_day_of_week").isin(1, 7), 1
    ).otherwise(0)
)

# Extract pickup month
df_gold = df_gold.withColumn(
    "pickup_month",
    F.month("tpep_pickup_datetime")
)

# Rush-hour indicator: weekdays, 7–9 AM or 4–7 PM
df_gold = df_gold.withColumn(
    "is_rush_hour",
    F.when(
        (F.col("is_weekend") == 0) &
        (
            F.col("pickup_hour").between(7, 9) |
            F.col("pickup_hour").between(16, 19)
        ),
        1
    ).otherwise(0)
)

# Select features and target for our Gold dataset
df_gold = df_gold.select(
    "tpep_pickup_datetime",
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "pickup_month",
    "is_rush_hour",
    "PULocationID",
    "DOLocationID",
    "duration_seconds"
)

# Verify the selected columns
df_gold.printSchema()
df_gold.show(10, truncate=False)



# Save the Gold dataset
df_gold.write \
    .mode("overwrite") \
    .parquet("data/gold/trips_features")

print("Gold dataset saved successfully!")


spark.stop()
