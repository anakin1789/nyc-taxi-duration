from pyspark.sql import SparkSession
from pathlib import Path
from pyspark.sql import functions as F

#1. create the spark session

spark = (SparkSession.builder
.appName("NYC-Taxi-Bronze")
.master("local[*]")
.getOrCreate()
)

#Read october 2025 parquet file from the landing zone and write it to the bronze zone

df = spark.read.parquet(
    "data/bronze/yellow_tripdata_2025-10.parquet"
    )


#3. inspect the dataset

df.printSchema()
df.show(5 , truncate=False)

#4. count all trips 
print("Total records:", df.count())

#read november 2025

df_nov = spark.read.parquet(
    "data/bronze/yellow_tripdata_2025-11.parquet"
)

#compare both shcemas 
print("Same schema:", df.schema == df_nov.schema)

files = sorted(
    Path("data/bronze").glob("yellow_tripdata_*.parquet")
)
print("Number of files :", len(files))

for file in files:
    df_month = spark.read.parquet(str(file))

    print(
        file.name,
        "Same schema:",
         df_month.schema == df.schema,
    )


df_june = spark.read.parquet(
    "data/bronze/yellow_tripdata_2026-06.parquet"
)

print("June 2026 schema:")
df_june.printSchema()

for month in ["07", "08", "09"]:
    df_month = spark.read.parquet(
        f"data/bronze/yellow_tripdata_2026-{month}.parquet"
    )

    print(
        f"2026-{month}:",
        df_month.schema == df_june.schema
    )

files = sorted(
    Path("data/bronze").glob("yellow_tripdata_*.parquet")
)

# Start with an empty reference
df_combined = None

for i, file in enumerate(files, start=1):
    df_month = spark.read.parquet(str(file))

    if df_combined is None:
        df_combined = df_month
    else:
        df_combined = df_combined.unionByName(
            df_month,
            allowMissingColumns=True
        )

    print(f"Added {i}/{len(files)}: {file.name}")

print("Months combined:", len(files))
print("Total trips:", df_combined.count())



files = sorted(
    Path("data/bronze").glob("yellow_tripdata_*.parquet")
)

print("Files detected:", len(files))

for file in files:
    print(file.name)
    


df_combined.select(
    F.min("tpep_pickup_datetime").alias("first_pickup"),
    F.max("tpep_pickup_datetime").alias("last_pickup")
).show(truncate=False)



df_outside = df_combined.filter(
    (F.col("tpep_pickup_datetime") < "2025-10-01") |
    (F.col("tpep_pickup_datetime") >= "2026-10-01")
)

print("Trips outside study period:", df_outside.count())


spark.stop()