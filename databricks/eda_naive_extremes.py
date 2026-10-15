"""EDA: is September broadly hard, or is one heat wave carrying the average?

Follow-up to eda_naive_error.py, which showed CISO seasonal-naive at 11.99% MAPE in
September against a 4-8% norm. With only four Septembers in the data, a single extreme week
could be most of that number. The distinction matters for how it gets described:

  "September is hard"                    -> a seasonal effect, every year
  "naive fails during heat waves"        -> a tail effect, concentrated in a few days

Those call for different fixes and different README sentences, so it is worth 60 seconds of
compute to find out which it is.

The tool for this is comparing the MEAN to the MEDIAN. The mean is pulled by extremes; the
median ignores them. If mean >> median, a few days dominate.

Reads workspace.gold.features. Three aggregations, CISO only, no writes.
"""

from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

spark = DatabricksSession.builder.serverless(True).getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")

REGION = "CISO"

features = spark.table("workspace.gold.features").where(F.col("region") == REGION)

# Same eligibility rule as eda_naive_error.py, so the numbers are comparable to it.
eligible = features.where(
    ~F.col("has_gap") & F.col("lag_168h").isNotNull() & (F.col("demand_mw") > 0)
)

scored = (
    eligible.withColumn("error", F.col("demand_mw") - F.col("lag_168h"))
    .withColumn("abs_error", F.abs(F.col("error")))
    .withColumn("abs_pct_error", F.col("abs_error") / F.col("demand_mw") * 100.0)
    # local_date is already a local calendar date, so the year derived from it matches the
    # local month used in the previous script. Deriving it from period_utc instead would
    # misassign the first hours of January for western regions.
    .withColumn("year", F.year("local_date"))
)

# --- 1. Is every September bad, or just one? ----------------------------------------------
print(f"\n=== {REGION}: September, year by year ===")
print("If all four years sit near 12%, it is seasonal. If one spikes, it is an event.")
(
    scored.where(F.col("month") == 9)
    .groupBy("year")
    .agg(
        F.count("*").alias("n"),
        F.round(F.avg("demand_mw")).alias("mean_demand_mw"),
        F.round(F.avg("abs_error")).alias("mae_mw"),
        F.round(F.avg("abs_pct_error"), 2).alias("mape_pct"),
        F.round(F.avg("error")).alias("bias_mw"),
        F.round(F.max("temp_c"), 1).alias("max_temp_c"),
    )
    .orderBy("year")
    .show(10, truncate=False)
)

# --- 2. Mean vs median: how much is the tail doing? ---------------------------------------
print(f"\n=== {REGION}: mean vs median absolute error, by month ===")
print("ratio = mae / median. Near 1.3 is an ordinary right-skewed spread.")
print("Much higher means a handful of hours are carrying that month's average.")
(
    scored.groupBy("month")
    .agg(
        F.round(F.avg("abs_error")).alias("mae_mw"),
        # median is immune to the extreme values that drag the mean upward
        F.round(F.median("abs_error")).alias("median_abs_err_mw"),
        F.round(F.percentile_approx("abs_error", 0.95)).alias("p95_abs_err_mw"),
        F.round(F.max("abs_error")).alias("max_abs_err_mw"),
    )
    .withColumn(
        "ratio_mean_over_median",
        F.round(F.col("mae_mw") / F.col("median_abs_err_mw"), 2),
    )
    .orderBy("month")
    .show(12, truncate=False)
)

# --- 3. The individual worst days, with weather attached ----------------------------------
# Including temperature turns "these days were bad" into evidence about WHY. cdh is cooling
# degree-hours (base 18C), so a high daily mean cdh is a hot day by construction.
print(f"\n=== {REGION}: 20 worst individual days (by mean absolute error) ===")
print("Check the dates: clustered in a few heat waves, or scattered across the calendar?")
(
    scored.groupBy("local_date", "year", "month")
    .agg(
        F.round(F.avg("abs_error")).alias("day_mae_mw"),
        F.round(F.avg("abs_pct_error"), 2).alias("day_mape_pct"),
        F.round(F.avg("error")).alias("day_bias_mw"),
        F.round(F.max("demand_mw")).alias("peak_demand_mw"),
        F.round(F.max("temp_c"), 1).alias("max_temp_c"),
        F.round(F.avg("cdh"), 1).alias("mean_cdh"),
        F.max("is_major_holiday").alias("holiday"),
    )
    .orderBy(F.col("day_mae_mw").desc())
    .show(20, truncate=False)
)
