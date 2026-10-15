"""EDA: where does seasonal-naive actually fail?

Not part of the numbered pipeline - this is exploration, so it is named outside the
01_/02_/03_ sequence to keep `04_` free for train_fleet (CLAUDE.md section 8).

Seasonal-naive is the benchmark reference line from section 6: the forecast for a given
hour is simply the demand observed at the same hour one week earlier. So its error is
exactly the week-over-week change in demand. That single fact explains everything this
script measures - the model is blind to weather, calendar effects, and anything else that
differs between last week and this week.

Reads workspace.gold.features. Two small aggregations, CISO only, no writes.
"""

from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

spark = DatabricksSession.builder.serverless(True).getOrCreate()

# Same pin as 03_gold_features.py. Without it, Spark would interpret these timestamps
# through whatever session timezone the server defaults to, and the local calendar columns
# we already computed would no longer line up with the UTC instants.
spark.conf.set("spark.sql.session.timeZone", "UTC")

REGION = "CISO"

features = spark.table("workspace.gold.features").where(F.col("region") == REGION)

# --- Eligibility --------------------------------------------------------------------------
# The same filter the backtest will use, so these numbers are comparable to it later.
#   has_gap        - EIA published nothing for this hour; there is no truth to score against
#   lag_168h null  - the hour one week earlier is missing, so the benchmark has no forecast
#   demand_mw <= 0 - guards the percentage error below against division by ~zero. Bronze had
#                    genuine zeros for NYIS and a 39 MW hour for FPL, so this is not
#                    hypothetical even if silver's range checks caught those.
eligible = features.where(
    ~F.col("has_gap") & F.col("lag_168h").isNotNull() & (F.col("demand_mw") > 0)
)

# The benchmark, in one line. q50 = lag_168h, so the error is actual minus last week.
scored = eligible.withColumn("error", F.col("demand_mw") - F.col("lag_168h")).withColumn(
    "abs_pct_error", F.abs(F.col("error")) / F.col("demand_mw") * 100.0
)

# Reused by both aggregations below.
METRICS = [
    F.count("*").alias("n"),
    F.round(F.avg("demand_mw")).alias("mean_demand_mw"),
    # MAE is in MW, so it grows with the level of demand - that is the units problem.
    F.round(F.avg(F.abs(F.col("error")))).alias("mae_mw"),
    # MAPE divides the level out, which is what makes July vs October a fair comparison.
    F.round(F.avg("abs_pct_error"), 2).alias("mape_pct"),
    # Mean *signed* error: near zero means the benchmark is unbiased at this slice; strongly
    # positive means demand is systematically higher than a week ago (a warming trend).
    F.round(F.avg("error")).alias("bias_mw"),
]

print(f"\n=== {REGION}: seasonal-naive error by LOCAL HOUR (prediction (a)) ===")
print("Watch mae_mw and mape_pct. Does the worst hour sit mid-afternoon or in the evening?")
(scored.groupBy("hour_local").agg(*METRICS).orderBy("hour_local").show(24, truncate=False))

print(f"\n=== {REGION}: seasonal-naive error by LOCAL MONTH (prediction (b)) ===")
print("Compare month 7 (July) with month 10 (October) in BOTH columns.")
print("mae_mw larger in July partly just reflects higher demand; mape_pct is the fair test.")
(scored.groupBy("month").agg(*METRICS).orderBy("month").show(12, truncate=False))
