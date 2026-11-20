
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Start Spark
spark = (
    SparkSession.builder
    .appName("NYC-Taxi-Split")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.session.timeZone", "America/New_York")
    .getOrCreate()
)

# Read Gold dataset
df = spark.read.parquet("data/gold/trips_features")

# Training: October 2025 to May 2026
df_train = df.filter(
    (F.col("tpep_pickup_datetime") >= "2025-10-01") &
    (F.col("tpep_pickup_datetime") < "2026-06-01")
)

# Validation: June to July 2026
df_val = df.filter(
    (F.col("tpep_pickup_datetime") >= "2026-06-01") &
    (F.col("tpep_pickup_datetime") < "2026-08-01")
)

# Test: August to September 2026
df_test = df.filter(
    (F.col("tpep_pickup_datetime") >= "2026-08-01") &
    (F.col("tpep_pickup_datetime") < "2026-10-01")
)

# Verify the split
train_count = df_train.count()
val_count = df_val.count()
test_count = df_test.count()

print("Training:", train_count)
print("Validation:", val_count)
print("Test:", test_count)
print("Total:", train_count + val_count + test_count)

spark.stop()
