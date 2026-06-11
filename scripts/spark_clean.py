from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import FloatType

spark = SparkSession.builder \
    .appName("FlightPipeline-Dataproc") \
    .config("spark.jars.packages",
            "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.0") \
    .getOrCreate()

spark.conf.set("viewsEnabled", "true")
spark.conf.set("materializationDataset", "staging")

PROJECT = "dataeng2-481419"
BUCKET  = "onky-flight-pipeline-raw"

print("✅ Spark Session started on Dataproc!")

# ── Read Bronze Historical ──────────────────────────────────────
hist_df = spark.read.format("bigquery") \
    .option("table", f"{PROJECT}.raw.raw_flights_historical") \
    .load()
print(f"📥 Historical Bronze: {hist_df.count():,} rows")

# ── Clean Historical (Bronze → Silver) ─────────────────────────
hist_clean = hist_df \
    .withColumnRenamed("FL_DATE",              "flight_date") \
    .withColumnRenamed("AIRLINE_CODE",         "carrier_code") \
    .withColumnRenamed("AIRLINE",              "carrier_name") \
    .withColumnRenamed("FL_NUMBER",            "flight_number") \
    .withColumnRenamed("ORIGIN",               "origin_airport") \
    .withColumnRenamed("ORIGIN_CITY",          "origin_city") \
    .withColumnRenamed("DEST",                 "dest_airport") \
    .withColumnRenamed("DEST_CITY",            "dest_city") \
    .withColumnRenamed("DEP_DELAY",            "dep_delay_min") \
    .withColumnRenamed("ARR_DELAY",            "arr_delay_min") \
    .withColumnRenamed("CANCELLED",            "is_cancelled") \
    .withColumnRenamed("DISTANCE",             "distance_miles") \
    .withColumnRenamed("AIR_TIME",             "air_time_min") \
    .withColumnRenamed("DELAY_DUE_CARRIER",    "carrier_delay_min") \
    .withColumnRenamed("DELAY_DUE_WEATHER",    "weather_delay_min") \
    .withColumnRenamed("DELAY_DUE_NAS",        "nas_delay_min") \
    .withColumnRenamed("DELAY_DUE_SECURITY",   "security_delay_min") \
    .withColumnRenamed("CANCELLATION_CODE",    "cancellation_code") \
    .withColumn("dep_delay_min",   col("dep_delay_min").cast(FloatType())) \
    .withColumn("arr_delay_min",   col("arr_delay_min").cast(FloatType())) \
    .withColumn("distance_miles",  col("distance_miles").cast(FloatType())) \
    .withColumn("air_time_min",    col("air_time_min").cast(FloatType())) \
    .withColumn("carrier_delay_min",  col("carrier_delay_min").cast(FloatType())) \
    .withColumn("weather_delay_min",  col("weather_delay_min").cast(FloatType())) \
    .withColumn("nas_delay_min",      col("nas_delay_min").cast(FloatType())) \
    .withColumn("carrier_code",    upper(trim(col("carrier_code")))) \
    .withColumn("origin_airport",  upper(trim(col("origin_airport")))) \
    .withColumn("dest_airport",    upper(trim(col("dest_airport")))) \
    .dropna(subset=["carrier_code", "origin_airport", "dest_airport"]) \
    .dropDuplicates() \
    .withColumn("delay_category",
        when(col("is_cancelled") == 1,      lit("CANCELLED"))
        .when(col("arr_delay_min") <= 0,    lit("ON_TIME"))
        .when(col("arr_delay_min") <= 15,   lit("MINOR_DELAY"))
        .when(col("arr_delay_min") <= 60,   lit("MAJOR_DELAY"))
        .otherwise(                         lit("SEVERE_DELAY"))
    ) \
    .withColumn("delay_reason",
        when(col("carrier_delay_min") > 0,  lit("CARRIER"))
        .when(col("weather_delay_min") > 0, lit("WEATHER"))
        .when(col("nas_delay_min") > 0,     lit("NAS"))
        .when(col("security_delay_min") > 0,lit("SECURITY"))
        .otherwise(                         lit("OTHER"))
    ) \
    .withColumn("processed_at", current_timestamp())

print(f"✅ Historical Silver rows: {hist_clean.count():,}")

# ── Write Historical Silver ─────────────────────────────────────
hist_clean.write.format("bigquery") \
    .option("table", f"{PROJECT}.staging.stg_flights_historical") \
    .option("temporaryGcsBucket", BUCKET) \
    .option("writeDisposition", "WRITE_TRUNCATE") \
    .mode("overwrite") \
    .save()
print("✅ Historical Silver written to BigQuery!")

# ── Read Bronze Live ────────────────────────────────────────────
live_df = spark.read.format("bigquery") \
    .option("table", f"{PROJECT}.raw.raw_flights_live") \
    .load()
print(f"📥 Live Bronze: {live_df.count():,} rows")

# ── Clean Live ──────────────────────────────────────────────────
live_clean = live_df \
    .dropna(subset=["callsign"]) \
    .withColumn("callsign",      upper(trim(col("callsign")))) \
    .withColumn("carrier_code",
        regexp_extract(col("callsign"), r'^([A-Z]{2,3})\d', 1)
    ) \
    .withColumn("speed_kmh",     round(col("velocity_ms") * 3.6, 2)) \
    .withColumn("altitude_ft",   round(col("altitude_m") * 3.28084, 0)) \
    .withColumn("flight_phase",
        when(col("altitude_m") < 1000,  lit("TAKEOFF_LANDING"))
        .when(col("altitude_m") < 6000, lit("CLIMBING_DESCENDING"))
        .otherwise(                      lit("CRUISING"))
    ) \
    .withColumn("speed_category",
        when(col("speed_kmh") < 300,  lit("SLOW"))
        .when(col("speed_kmh") < 700, lit("NORMAL"))
        .otherwise(                    lit("FAST"))
    ) \
    .filter(col("carrier_code") != "") \
    .withColumn("processed_at", current_timestamp())

print(f"✅ Live Silver rows: {live_clean.count():,}")

# ── Write Live Silver ───────────────────────────────────────────
live_clean.write.format("bigquery") \
    .option("table", f"{PROJECT}.staging.stg_flights_live") \
    .option("temporaryGcsBucket", BUCKET) \
    .option("writeDisposition", "WRITE_TRUNCATE") \
    .mode("overwrite") \
    .save()
print("✅ Live Silver written to BigQuery!")
print("\n🎉 Dataproc PySpark pipeline COMPLETE!")

spark.stop()
