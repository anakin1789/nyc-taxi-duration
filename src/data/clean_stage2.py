
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("NYC-Taxi-Silver-Stage2")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.parquet.enableDictionary", "false")
    .config("spark.sql.session.timeZone", "America/New_York")
    .getOrCreate()
)

# Read existing Silver Stage 1
df = spark.read.parquet("data/silver/trips_stage1")

# Exclude trips longer than 24 hours
df_stage2 = df.filter(
    F.col("duration_seconds") <= 24 * 3600
)


print("Trips after outlier filtering:", df_stage2.count())

# Add the suspicious distance flag
df_stage2 = df_stage2.withColumn(
    "suspicious_distance",
    (F.col("trip_distance") == 0) &
    (F.col("PULocationID") != F.col("DOLocationID"))
)

# Save Silver Stage 2 with the new column
df_stage2.write.mode("overwrite").parquet(
    "data/silver/trips_stage2"
)

spark.stop()
