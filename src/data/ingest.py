
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. Create a Spark session
spark = SparkSession.builder \
    .appName("NYC-Taxi-Bronze") \
    .master("local[*]") \
    .getOrCreate()

# 2. Read the raw dataset
df = spark.read.csv(
    "data/bronze/training/train.csv",
    header=True,
    inferSchema=True 
)

# 3. Display the first 5 records
df.show(5)
df.printSchema()
print("total records:", df.count())

#we need to find out whether any columns contain null values

df.select([
    F.count(F.when(F.col(c).isNull(), 1)).alias(c) 
    for c in df.columns
]).show()


df.select(
    F.min("trip_duration").alias("min_duration"),
    F.max("trip_duration").alias("max_duration"),
    F.mean("trip_duration").alias("mean_duration"),
).show()

df.select(
    F.count(F.when(F.col("trip_duration") < 60, 1)).alias("under_1_minute"),
    F.count(F.when(F.col("trip_duration") > 10800, 1)).alias("over_3_hour"),
).show()

percentiles = df.approxQuantile(
    "trip_duration",
    [0.50,0.95,0.99,0.999],
    0.001
)

print("P50:", percentiles[0])
print("P95:", percentiles[1])
print("P99:", percentiles[2])
print("P99.9:", percentiles[3])
print("duration percentiles:", percentiles)

p99= df.approxQuantile("trip_duration", [0.999], 0.001)
print("P99.9:", p99[0])

spark.stop()
