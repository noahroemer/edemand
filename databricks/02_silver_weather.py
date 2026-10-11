from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

spark = DatabricksSession.builder.serverless(True).getOrCreate()

# All-time continental US records. A reading outside this range is a data error.
TEMP_MIN_C, TEMP_MAX_C = -56.7, 56.7

bronze = spark.table("workspace.bronze.weather_raw")

# Grid per (region, source): each weather source gets its own complete set of hours.
bounds = bronze.groupBy("region", "source").agg(
    F.min("period_utc").alias("start"),
    F.max("period_utc").alias("end"),
)
grid = bounds.select(
    "region",
    "source",
    F.explode(F.sequence("start", "end", F.expr("INTERVAL 1 HOUR"))).alias("period_utc"),
)

# Left join keeps every grid hour. Problems are flagged, never filled or dropped.
silver = (
    grid.join(bronze, on=["region", "source", "period_utc"], how="left")
    .withColumn("has_gap", F.col("temp_c").isNull())
    .withColumn(
        "temp_out_of_range",
        F.col("temp_c").isNotNull() & ~F.col("temp_c").between(TEMP_MIN_C, TEMP_MAX_C),
    )
    .withColumn(
        "humidity_out_of_range",
        F.col("humidity_pct").isNotNull() & ~F.col("humidity_pct").between(0, 100),
    )
    .select(
        "region",
        "source",
        "period_utc",
        "temp_c",
        "humidity_pct",
        "has_gap",
        "temp_out_of_range",
        "humidity_out_of_range",
        "fetched_at",
    )
)

silver.groupBy("region", "source").agg(
    F.count("*").alias("rows"),
    F.sum(F.col("has_gap").cast("int")).alias("gaps"),
    F.sum(F.col("temp_out_of_range").cast("int")).alias("bad_temp"),
    F.sum(F.col("humidity_out_of_range").cast("int")).alias("bad_humidity"),
    F.min("temp_c").alias("min_temp"),
    F.max("temp_c").alias("max_temp"),
).orderBy("region", "source").show()

silver.write.mode("overwrite").saveAsTable("workspace.silver.weather")
