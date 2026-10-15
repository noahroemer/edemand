"""Build seasonal-naive prediction intervals and measure whether they are calibrated.

This is the construction step from concept 2, on a SINGLE train/test split. It is not the
21-fold backtest yet - the point is to see the mechanism work, and to see what hour
stratification actually buys, before generalising the machinery.

The method (empirical residual quantiles):

    1. On the TRAINING window, compute every naive error  e = actual - lag_168h
    2. Group by (region, hour_local) and take the empirical 10th/50th/90th percentiles
    3. Add those offsets to the test-window point forecast

        q10(t) = lag_168h(t) + e10(region, hour_local)
        q90(t) = lag_168h(t) + e90(region, hour_local)

Training quantiles are estimated ONLY on data before the test window starts, which is what
makes this leak-free. Using test-window errors to set the interval would guarantee good
coverage and prove nothing.

Two variants are built side by side so the stratification question is answered by
measurement rather than assertion:

    stratified    offsets per (region, hour_local)   - 240 groups
    pooled        offsets per (region)               - 10 groups, one width all day

Reads workspace.gold.features. No writes.
"""

from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

spark = DatabricksSession.builder.serverless(True).getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")

# Same boundary as fold 1 of the planned backtest, so these numbers preview it.
TRAIN_END = "2025-01-01"
# Stop before the partial month: the data currently ends mid-October 2026.
TEST_END = "2026-10-01"

NOMINAL = 0.80  # the 80% interval from CLAUDE.md section 10

features = spark.table("workspace.gold.features")

eligible = features.where(
    ~F.col("has_gap") & F.col("lag_168h").isNotNull() & (F.col("demand_mw") > 0)
).withColumn("error", F.col("demand_mw") - F.col("lag_168h"))

train = eligible.where(F.col("period_utc") < TRAIN_END)
test = eligible.where((F.col("period_utc") >= TRAIN_END) & (F.col("period_utc") < TEST_END))

# --- Fit: the only thing seasonal-naive "learns" is its error distribution --------------
# The point forecast has zero fitted parameters. These offsets are the entire model.
#
# percentile_approx takes an array of probabilities and returns an array, so one pass gets
# all three. Spark has no exact percentile at scale; the default accuracy (10000) puts the
# error well below the MW resolution we care about.
QUANTILES = [0.1, 0.5, 0.9]

stratified_offsets = (
    train.groupBy("region", "hour_local")
    .agg(
        F.percentile_approx("error", QUANTILES).alias("q"),
        F.count("*").alias("n_train"),
    )
    .select(
        "region",
        "hour_local",
        "n_train",
        F.col("q")[0].alias("s_e10"),
        F.col("q")[1].alias("s_e50"),
        F.col("q")[2].alias("s_e90"),
    )
)

pooled_offsets = (
    train.groupBy("region")
    .agg(F.percentile_approx("error", QUANTILES).alias("q"))
    .select(
        "region",
        F.col("q")[0].alias("p_e10"),
        F.col("q")[1].alias("p_e50"),
        F.col("q")[2].alias("p_e90"),
    )
)

# --- Predict: add the offsets to the point forecast -------------------------------------
# broadcast because both offset tables are tiny (240 and 10 rows) - this avoids a shuffle
# of the 300k-row test set.
scored = (
    test.join(F.broadcast(stratified_offsets), on=["region", "hour_local"], how="left")
    .join(F.broadcast(pooled_offsets), on="region", how="left")
    .withColumn("s_q10", F.col("lag_168h") + F.col("s_e10"))
    .withColumn("s_q50", F.col("lag_168h") + F.col("s_e50"))
    .withColumn("s_q90", F.col("lag_168h") + F.col("s_e90"))
    .withColumn("p_q10", F.col("lag_168h") + F.col("p_e10"))
    .withColumn("p_q90", F.col("lag_168h") + F.col("p_e90"))
    # Coverage is a simple indicator: did the actual land inside the interval?
    .withColumn(
        "s_covered",
        ((F.col("demand_mw") >= F.col("s_q10")) & (F.col("demand_mw") <= F.col("s_q90"))).cast(
            "int"
        ),
    )
    .withColumn(
        "p_covered",
        ((F.col("demand_mw") >= F.col("p_q10")) & (F.col("demand_mw") <= F.col("p_q90"))).cast(
            "int"
        ),
    )
    .withColumn("s_width", F.col("s_q90") - F.col("s_q10"))
    .withColumn("p_width", F.col("p_q90") - F.col("p_q10"))
)

print(f"\n=== Coverage by region (nominal {NOMINAL:.0%}) ===")
print("Read 78-82% as calibrated. Much above = too wide. Much below = overconfident.")
print("rel_width expresses sharpness as a % of mean demand, so regions are comparable.")
(
    scored.groupBy("region")
    .agg(
        F.count("*").alias("n_test"),
        F.round(F.avg("s_covered") * 100, 1).alias("strat_cover_pct"),
        F.round(F.avg("p_covered") * 100, 1).alias("pooled_cover_pct"),
        F.round(F.avg("s_width")).alias("strat_width_mw"),
        F.round(F.avg("s_width") / F.avg("demand_mw") * 100, 1).alias("strat_rel_width_pct"),
    )
    .orderBy("region")
    .show(12, truncate=False)
)

print("\n=== CISO by local hour: what stratification actually buys ===")
print("Pooled uses ONE width for all 24 hours. Watch it over-cover overnight")
print("and under-cover at the evening peak - failing in both directions at once.")
(
    scored.where(F.col("region") == "CISO")
    .groupBy("hour_local")
    .agg(
        F.round(F.avg("s_covered") * 100, 1).alias("strat_cover_pct"),
        F.round(F.avg("p_covered") * 100, 1).alias("pooled_cover_pct"),
        F.round(F.avg("s_width")).alias("strat_width_mw"),
        F.round(F.avg("p_width")).alias("pooled_width_mw"),
    )
    .orderBy("hour_local")
    .show(24, truncate=False)
)

print("\n=== Calibration error: mean |coverage - nominal| across the 24 hours, CISO ===")
print("One number per method. Lower is better calibrated across the day.")
by_hour = (
    scored.where(F.col("region") == "CISO")
    .groupBy("hour_local")
    .agg(
        F.avg("s_covered").alias("s_cov"),
        F.avg("p_covered").alias("p_cov"),
    )
)
(
    by_hour.agg(
        F.round(F.avg(F.abs(F.col("s_cov") - F.lit(NOMINAL))) * 100, 2).alias(
            "stratified_mean_abs_gap_pct"
        ),
        F.round(F.avg(F.abs(F.col("p_cov") - F.lit(NOMINAL))) * 100, 2).alias(
            "pooled_mean_abs_gap_pct"
        ),
    ).show(truncate=False)
)
