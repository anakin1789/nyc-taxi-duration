
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

spark = (
    SparkSession.builder
    .appName("NYC-Taxi-ML-Preprocessing")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.session.timeZone", "America/New_York")
    .getOrCreate()
)

# Read Gold
df = spark.read.parquet("data/gold/trips_features")

# Select training period only
df_train = df.filter(
    (F.col("tpep_pickup_datetime") >= "2025-10-01") &
    (F.col("tpep_pickup_datetime") < "2026-06-01")
)

# Small sample for testing the preprocessing pipeline
train_sample = df_train.sample(
    withReplacement=False,
    fraction=0.003,
    seed=42
)

# Categorical columns
categorical_cols = [
    "pickup_day_of_week",
    "pickup_month",
    "PULocationID",
    "DOLocationID"
]

# Convert categories into indices
indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=col + "_index",
        handleInvalid="keep"
    )
    for col in categorical_cols
]

# Convert category indices into vectors
encoder = OneHotEncoder(
    inputCols=[col + "_index" for col in categorical_cols],
    outputCols=[col + "_encoded" for col in categorical_cols],
    dropLast=True
)

# Build the preprocessing pipeline
pipeline = Pipeline(stages=indexers + [encoder])

# Learn the encoding using training data
pipeline_model = pipeline.fit(train_sample)

# Transform the training sample
encoded_train = pipeline_model.transform(train_sample)



# Combine numerical and encoded categorical features
assembler = VectorAssembler(
    inputCols=[
        "pickup_hour",
        "is_rush_hour",
        "pickup_day_of_week_encoded",
        "pickup_month_encoded",
        "PULocationID_encoded",
        "DOLocationID_encoded"
    ],
    outputCol="features"
)

# Create the final feature vector
ml_train = assembler.transform(encoded_train)

# Display features and target
ml_train.select(
    "features",
    "duration_seconds"
).show(5, truncate=False)


# Verify results
encoded_train.select(
    "pickup_day_of_week",
    "pickup_day_of_week_encoded",
    "PULocationID",
    "PULocationID_encoded"
).show(10, truncate=False)







# Create our first Linear Regression model
lr = LinearRegression(
    featuresCol="features",
    labelCol="duration_seconds",
    maxIter=50,
    regParam=0.0,
    solver="l-bfgs"
)

# Train the model
import time

start = time.perf_counter()

# Train Linear Regression
lr_model = lr.fit(ml_train)

elapsed = time.perf_counter() - start

print(f"Training completed in {elapsed:.2f} seconds")

print("\n=== LINEAR REGRESSION TRAINING ===")

# Get training metrics
training_summary = lr_model.summary

print("Training MAE:", training_summary.meanAbsoluteError)
print("Training RMSE:", training_summary.rootMeanSquaredError)
print("Training R2:", training_summary.r2)

# Compare real and predicted durations
lr_model.transform(ml_train).select(
    "duration_seconds",
    "prediction"
).show(10, truncate=False)


spark.stop()
