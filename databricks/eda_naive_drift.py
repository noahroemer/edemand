"""Why does the single-split naive interval under-cover in 7 of 10 regions?

eda_naive_intervals.py found stratified coverage at 69.7-82.9% against a nominal 80%, with
seven regions 6-10 points low. Hour stratification does not help, because it redistributes
width across hours without changing the total. So the cause is a level effect, not an
allocation effect.

Hypothesis: NON-STATIONARITY. The offsets are fixed MW amounts estimated on 2023-24 and
applied unchanged to 2025-26. If demand grew over that period, errors grew with it, and a
fixed-MW interval is too narrow everywhere.

Three predictions, each falsifiable:

  1. Regions that grew more should under-cover more.  -> block 1, compare growth_pct
     against the coverage from the previous script.
  2. The DIRECTION of the breaches separates two different growth mechanisms.

     Note first that the naive point forecast self-updates: it is lag_168h, drawn from the
     test period itself, so a rising level does not bias it. The error is a DIFFERENCE,
     and its centre stays near zero regardless of level (measured bias was ~+52 MW against
     widths of several thousand).

     So level growth does not SHIFT the error distribution, it WIDENS it: 14% more demand
     means week-over-week swings roughly 14% larger in MW, while the fixed training
     offsets stay put.

       symmetric excess (below ~= above)  -> variance scaling with level.
                                             Fix: multiplicative offsets, quantiles of
                                             e/y instead of e.
       excess skewed ABOVE q90            -> a systematic upward week-over-week trend,
                                             a shift rather than a widening.
                                             Fix: a drift term in the benchmark.
  3. Coverage should DECAY with distance from the training window, since drift compounds.
     -> block 2. A flat line falsifies the whole hypothesis.

Block 3 isolates the hours 8-11 dip separately: that one is local to the morning solar
ramp and may have its own cause.

Reads workspace.gold.features. No writes.
"""

from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

spark = DatabricksSession.builder.serverless(True).getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")

TRAIN_END = "2025-01-01"
TEST_END = "2026-10-01"
QUANTILES = [0.1, 0.5, 0.9]

features = spark.table("workspace.gold.features")

eligible = features.where(
    ~F.col("has_gap") & F.col("lag_168h").isNotNull() & (F.col("demand_mw") > 0)
).withColumn("error", F.col("demand_mw") - F.col("lag_168h"))

train = eligible.where(F.col("period_utc") < TRAIN_END)
test = eligible.where((F.col("period_utc") >= TRAIN_END) & (F.col("period_utc") < TEST_END))

offsets = (
    train.groupBy("region", "hour_local")
    .agg(F.percentile_approx("error", QUANTILES).alias("q"))
    .select(
        "region",
        "hour_local",
        F.col("q")[0].alias("e10"),
        F.col("q")[1].alias("e50"),
        F.col("q")[2].alias("e90"),
    )
)

scored = (
    test.join(F.broadcast(offsets), on=["region", "hour_local"], how="left")
    .withColumn("q10", F.col("lag_168h") + F.col("e10"))
    .withColumn("q90", F.col("lag_168h") + F.col("e90"))
    # Split the miss into its two directions. Coverage alone hides which tail is failing,
    # and the direction is what distinguishes growth from plain extra volatility.
    .withColumn("below", (F.col("demand_mw") < F.col("q10")).cast("int"))
    .withColumn("above", (F.col("demand_mw") > F.col("q90")).cast("int"))
    .withColumn("covered", (1 - F.col("below") - F.col("above")))
)

# Train-period level per region, to pair against test-period level.
train_level = train.groupBy("region").agg(
    F.avg("demand_mw").alias("train_mean_mw"),
    F.avg(F.abs("error")).alias("train_mae_mw"),
)

print("\n=== 1. Growth and breach direction, by region ===")
print("Nominal: 10% below q10, 10% above q90. Excess concentrated ABOVE supports growth.")
print("growth_pct = test mean demand vs train mean demand.")
(
    scored.groupBy("region")
    .agg(
        F.avg("demand_mw").alias("test_mean_mw"),
        F.avg(F.abs("error")).alias("test_mae_mw"),
        F.round(F.avg("covered") * 100, 1).alias("cover_pct"),
        F.round(F.avg("below") * 100, 1).alias("below_q10_pct"),
        F.round(F.avg("above") * 100, 1).alias("above_q90_pct"),
    )
    .join(F.broadcast(train_level), on="region")
    .withColumn(
        "growth_pct",
        F.round((F.col("test_mean_mw") / F.col("train_mean_mw") - 1) * 100, 1),
    )
    # If errors scale with level, error growth should roughly track demand growth.
    .withColumn(
        "mae_growth_pct",
        F.round((F.col("test_mae_mw") / F.col("train_mae_mw") - 1) * 100, 1),
    )
    .select(
        "region",
        "cover_pct",
        "below_q10_pct",
        "above_q90_pct",
        "growth_pct",
        "mae_growth_pct",
    )
    .orderBy("cover_pct")
    .show(12, truncate=False)
)

print("\n=== 2. Does coverage decay with distance from the training window? ===")
print("Drift compounds, so a drifting series should get worse month by month.")
print("A flat line falsifies the drift hypothesis. LDWP excluded - its data is bad (section 4).")
(
    scored.where(F.col("region") != "LDWP")
    .withColumn("test_month", F.date_trunc("month", F.col("period_utc")))
    .groupBy("test_month")
    .agg(
        F.round(F.avg("covered") * 100, 1).alias("cover_pct"),
        F.round(F.avg("below") * 100, 1).alias("below_pct"),
        F.round(F.avg("above") * 100, 1).alias("above_pct"),
    )
    .orderBy("test_month")
    .show(24, truncate=False)
)

print("\n=== 3. The CISO hours 8-11 dip: which tail is failing? ===")
print("If the morning solar ramp steepened, misses should skew one way, not both.")
(
    scored.where(F.col("region") == "CISO")
    .groupBy("hour_local")
    .agg(
        F.round(F.avg("covered") * 100, 1).alias("cover_pct"),
        F.round(F.avg("below") * 100, 1).alias("below_q10_pct"),
        F.round(F.avg("above") * 100, 1).alias("above_q90_pct"),
    )
    .orderBy("hour_local")
    .show(24, truncate=False)
)
