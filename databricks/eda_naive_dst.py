"""EDA: does the DST transition break seasonal-naive, and by how much?

The claim being tested. `lag_168h` steps back exactly 168 hours of *UTC* time. That equals
one week of *local clock* time only while the UTC offset is unchanged. Across a DST
transition it does not, so for the seven days following a transition the benchmark compares
a target hour against a different local hour.

For CISO (PST = UTC-8, PDT = UTC-7), with the target at local hour H:

    after spring forward   the source lands on local hour H-1   (shift = +1)
    after fall back        the source lands on local hour H+1   (shift = -1)

That gives a falsifiable prediction with signs, not just "error goes up". During the evening
ramp demand climbs steeply, so:

    shift = +1  ->  source hour is earlier, demand lower  ->  forecast too LOW  ->  bias > 0
    shift = -1  ->  source hour is later,  demand higher  ->  forecast too HIGH ->  bias < 0

and the effect should concentrate in the steep-ramp hours (16-20) and nearly vanish
overnight, where neighbouring hours barely differ.

Rather than hardcoding DST dates, this measures the misalignment directly: self-join each
row to the row 168 hours earlier and compare `hour_local`. That detects the shift from the
data itself, and works for any region's timezone without a lookup table.

Reads workspace.gold.features. Three aggregations, no writes.
"""

from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

spark = DatabricksSession.builder.serverless(True).getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")

features = spark.table("workspace.gold.features")

eligible = features.where(
    ~F.col("has_gap") & F.col("lag_168h").isNotNull() & (F.col("demand_mw") > 0)
)

# The source side of the comparison, shifted forward 168h so it joins onto its own target.
# Adding the interval to the SOURCE (rather than subtracting from the target) means the join
# key is already the target's period_utc.
source = eligible.select(
    "region",
    F.expr("period_utc + INTERVAL 168 HOURS").alias("period_utc"),
    F.col("hour_local").alias("src_hour_local"),
    F.col("demand_mw").alias("src_demand_mw"),
)

joined = (
    eligible.join(source, on=["region", "period_utc"], how="inner")
    .withColumn("error", F.col("demand_mw") - F.col("lag_168h"))
    .withColumn("abs_error", F.abs(F.col("error")))
    .withColumn("abs_pct_error", F.col("abs_error") / F.col("demand_mw") * 100.0)
    # Normalise the raw hour difference into {-1, 0, +1}, mapping wrap-around cases
    # (hour 23 vs hour 0) onto small signed numbers instead of +-23.
    #
    # Must be pmod, NOT %. Spark's % follows the sign of the dividend, like Java: in Spark
    # (-11) % 24 = -11, where Python gives 13. With %, the hour-0 rows of a spring-forward
    # week came out as shift = -23 instead of +1, splitting them into a bogus fourth group
    # and leaving spring_bias_mw null at hour 0.
    .withColumn(
        "shift",
        F.pmod(F.col("hour_local") - F.col("src_hour_local") + F.lit(12), F.lit(24)) - F.lit(12),
    )
)

METRICS = [
    F.count("*").alias("n"),
    F.countDistinct("local_date").alias("distinct_days"),
    F.round(F.avg("abs_error")).alias("mae_mw"),
    F.round(F.avg("abs_pct_error"), 2).alias("mape_pct"),
    F.round(F.avg("error")).alias("bias_mw"),
]

print("\n=== CISO: error grouped by local-hour misalignment ===")
print("shift 0 = aligned (normal weeks). shift +1 = after spring forward. -1 = after fall back.")
print("Also checks gold's lag_168h against the self-joined value - mismatches should be 0.")
(
    joined.where(F.col("region") == "CISO")
    .groupBy("shift")
    .agg(
        *METRICS,
        F.min("local_date").alias("first_date"),
        F.max("local_date").alias("last_date"),
        # Validates that lag_168h really is the demand 168h earlier. If this is non-zero,
        # the window function in 03_gold_features.py is not doing what we think.
        F.sum(
            F.when(F.abs(F.col("lag_168h") - F.col("src_demand_mw")) > 0.01, 1).otherwise(0)
        ).alias("lag_mismatches"),
    )
    .orderBy("shift")
    .show(10, truncate=False)
)

print("\n=== CISO: the misaligned weeks only, broken out by local hour ===")
print("Expect the damage concentrated in the evening ramp and small overnight.")
(
    joined.where((F.col("region") == "CISO") & (F.col("shift") != 0))
    .groupBy("hour_local")
    .agg(
        F.count("*").alias("n"),
        F.round(F.avg("abs_error")).alias("mae_mw"),
        F.round(F.avg("abs_pct_error"), 2).alias("mape_pct"),
        # Split the signed error by direction so the two transitions do not cancel out.
        F.round(F.avg(F.when(F.col("shift") == 1, F.col("error")))).alias("spring_bias_mw"),
        F.round(F.avg(F.when(F.col("shift") == -1, F.col("error")))).alias("fall_bias_mw"),
    )
    .orderBy("hour_local")
    .show(24, truncate=False)
)

print("\n=== All 10 regions: aligned vs misaligned ===")
print("If this is a real timezone effect it should appear everywhere, not just CISO.")
(
    joined.withColumn("aligned", F.col("shift") == 0)
    .groupBy("region")
    .agg(
        F.round(F.avg(F.when(F.col("aligned"), F.col("abs_pct_error"))), 2).alias("aligned_mape"),
        F.round(F.avg(F.when(~F.col("aligned"), F.col("abs_pct_error"))), 2).alias(
            "misaligned_mape"
        ),
        F.sum(F.when(~F.col("aligned"), 1).otherwise(0)).alias("misaligned_rows"),
    )
    .withColumn(
        "penalty_x",
        F.round(F.col("misaligned_mape") / F.col("aligned_mape"), 2),
    )
    .orderBy(F.col("penalty_x").desc())
    .show(12, truncate=False)
)
