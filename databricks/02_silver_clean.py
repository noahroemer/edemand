from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

spark = DatabricksSession.builder.serverless(True).getOrCreate()

bronze = spark.table("workspace.bronze.demand_raw")

# First and last hour each region has data for.
bounds = bronze.groupBy("region").agg(
    F.min("period_utc").alias("start"),
    F.max("period_utc").alias("end"),
)

# One row per region per hour between start and end.
grid = bounds.select(
    "region",
    F.explode(F.sequence("start", "end", F.expr("INTERVAL 1 HOUR"))).alias("period_utc"),
)


# Attach bronze values to the grid. "left" keeps every grid hour,
# even ones bronze has no row for (demand_mw comes back null).
silver = (
    grid.join(bronze, on=["region", "period_utc"], how="left")
    .withColumn("has_gap", F.col("demand_mw").isNull())
    .select("region", "period_utc", "demand_mw", "has_gap", "fetched_at")
)

silver.groupBy("region").agg(
    F.count("*").alias("rows"),
    F.sum(F.col("has_gap").cast("int")).alias("gaps"),
).orderBy("region").show()

silver.write.mode("overwrite").saveAsTable("workspace.silver.demand")

spark.sql(
    "SELECT COUNT(*) AS n, SUM(CAST(has_gap AS INT)) AS gaps FROM workspace.silver.demand"
).show()
