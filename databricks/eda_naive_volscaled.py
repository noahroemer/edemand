"""Option C: volatility-scaled residuals, instead of calendar stratification.

eda_naive_seasonal.py killed option A. The decisive evidence was August: coverage 53.2% in
2025 and 77.5% in 2026 under the same baseline, and 67.5% / 88.1% under month
stratification. Two Augusts, opposite failures. A calendar label cannot carry information
that differs between years with the same label - the driver is the weather in that
particular month, not the month itself.

So condition on something observable and current instead. Decompose the error into a scale
that moves with recent conditions and a shape that does not:

    e = sigma(t) * z

    sigma(t)  recent realised volatility - mean |error| over a trailing window, per region.
              Rises in a volatile August, falls in a calm one, and tracks growth and drift
              without being told about any of them.
    z         the standardised residual. Its quantiles are estimated per (region, hour)
              over the FULL training history, so each group keeps ~700 observations
              instead of the 52 that month stratification left us with.

    interval  q_tau(t) = lag_168h(t) + sigma(t) * z_tau(region, hour)

That is the trade option A forced - adaptivity against sample size - avoided rather than
made. It also handles seasonality, growth and drift at once, without needing to know which
is operating.

TWO THINGS THAT MATTER FOR CORRECTNESS:

1. Leakage. sigma(t) is computed from errors in a window ending 48 HOURS BEFORE the target,
   matching the rule in CLAUDE.md section 9. A window running up to t would use demand not
   yet published at the 13:00 UTC issue time, and would leak in the direction that flatters
   coverage.

2. sigma is computed over the whole series, train and test alike, and that is correct, not
   a leak: at serving time you genuinely do know last month's errors. What must come from
   training alone is the DISTRIBUTION z_tau. Scale is observed; shape is learned.

Block 1 checks the premise before any coverage number: if sigma works, z should have
roughly the same spread in train and test. If z still widens in test, the scaling has not
captured what changed and the coverage results will not be fixable either.

Reads workspace.gold.features. No writes.
"""

from databricks.connect import DatabricksSession
from pyspark.sql import Window
from pyspark.sql import functions as F

spark = DatabricksSession.builder.serverless(True).getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")

TRAIN_END = "2025-01-01"
TEST_END = "2026-10-01"
NOMINAL = 0.80
QUANTILES = [0.1, 0.5, 0.9]

HOUR = 3600
EMBARGO = 48 * HOUR
"""Gap between the end of the volatility window and the target hour. Section 9's rule."""

WINDOWS_DAYS = {"c7": 7, "c28": 28, "c56": 56}
"""Trailing window lengths to compare. Short adapts fast but is noisy; long is smooth but
lags a regime change. Which wins is empirical."""

features = spark.table("workspace.gold.features")

base = (
    features.where(~F.col("has_gap") & F.col("lag_168h").isNotNull() & (F.col("demand_mw") > 0))
    .withColumn("error", F.col("demand_mw") - F.col("lag_168h"))
    .withColumn("abs_error", F.abs(F.col("error")))
    # rangeBetween needs a numeric ordering column, so work in epoch seconds.
    .withColumn("ts", F.unix_timestamp("period_utc"))
)

# --- sigma(t): trailing mean |error|, embargoed by 48h -----------------------------------
for name, days in WINDOWS_DAYS.items():
    win = (
        Window.partitionBy("region")
        .orderBy("ts")
        .rangeBetween(-(days * 24 * HOUR + EMBARGO), -EMBARGO)
    )
    base = base.withColumn(f"{name}_sigma", F.avg("abs_error").over(win))
    # Guard against a null or zero sigma, which happens at the very start of the series
    # before the window has filled. Those rows cannot be scored by this method.
    base = base.withColumn(
        f"{name}_z",
        F.when(F.col(f"{name}_sigma") > 0, F.col("error") / F.col(f"{name}_sigma")),
    )

train = base.where(F.col("period_utc") < TRAIN_END)
test = base.where((F.col("period_utc") >= TRAIN_END) & (F.col("period_utc") < TEST_END))

print("\n=== 1. Did standardisation stabilise the distribution? ===")
print("If sigma captures what changed, z should have the same spread in train and test.")
print("raw_* is the unscaled error in MW, for contrast - it is expected to widen.")
spread = [
    F.round(F.percentile_approx("error", 0.9) - F.percentile_approx("error", 0.1)).alias(
        "raw_p10_p90_mw"
    )
]
for name in WINDOWS_DAYS:
    spread.append(
        F.round(
            F.percentile_approx(f"{name}_z", 0.9) - F.percentile_approx(f"{name}_z", 0.1), 3
        ).alias(f"{name}_z_p10_p90")
    )
(
    base.where(F.col("period_utc") < TEST_END)
    .withColumn("split", F.when(F.col("period_utc") < TRAIN_END, "1_train").otherwise("2_test"))
    .where(F.col("region") != "LDWP")
    .groupBy("split")
    .agg(*spread)
    .orderBy("split")
    .show(truncate=False)
)

# --- Fit: quantiles of z per (region, hour), on TRAINING rows only -----------------------
# sigma is region-level, pooled across hours, so z's per-hour distribution carries the
# shape of the day while sigma carries the level of volatility. Clean separation.
scored = test
for name in WINDOWS_DAYS:
    offsets = (
        train.where(F.col(f"{name}_z").isNotNull())
        .groupBy("region", "hour_local")
        .agg(
            F.percentile_approx(f"{name}_z", QUANTILES).alias("q"),
            F.count("*").alias(f"{name}_n_train"),
        )
        .select(
            "region",
            "hour_local",
            f"{name}_n_train",
            F.col("q")[0].alias(f"{name}_z10"),
            F.col("q")[2].alias(f"{name}_z90"),
        )
    )
    scored = scored.join(F.broadcast(offsets), on=["region", "hour_local"], how="left")

# Baseline for comparison: absolute MW offsets per (region, hour), the current best.
h_offsets = (
    train.groupBy("region", "hour_local")
    .agg(F.percentile_approx("error", QUANTILES).alias("q"))
    .select(
        "region",
        "hour_local",
        F.col("q")[0].alias("h_e10"),
        F.col("q")[2].alias("h_e90"),
    )
)
scored = scored.join(F.broadcast(h_offsets), on=["region", "hour_local"], how="left")

scored = (
    scored.withColumn("h_q10", F.col("lag_168h") + F.col("h_e10"))
    .withColumn("h_q90", F.col("lag_168h") + F.col("h_e90"))
    .withColumn(
        "h_cov",
        ((F.col("demand_mw") >= F.col("h_q10")) & (F.col("demand_mw") <= F.col("h_q90"))).cast(
            "int"
        ),
    )
    .withColumn("h_width", F.col("h_q90") - F.col("h_q10"))
)

for name in WINDOWS_DAYS:
    # The scaling step: a learned SHAPE multiplied by an observed SCALE.
    scored = (
        scored.withColumn(
            f"{name}_q10", F.col("lag_168h") + F.col(f"{name}_sigma") * F.col(f"{name}_z10")
        )
        .withColumn(
            f"{name}_q90", F.col("lag_168h") + F.col(f"{name}_sigma") * F.col(f"{name}_z90")
        )
        .withColumn(
            f"{name}_cov",
            (
                (F.col("demand_mw") >= F.col(f"{name}_q10"))
                & (F.col("demand_mw") <= F.col(f"{name}_q90"))
            ).cast("int"),
        )
        .withColumn(f"{name}_width", F.col(f"{name}_q90") - F.col(f"{name}_q10"))
    )

VARIANTS = ["h", *WINDOWS_DAYS]
COVERAGE = [F.round(F.avg(f"{v}_cov") * 100, 1).alias(f"{v}_cover") for v in VARIANTS]

print("\n=== 2. Coverage by test month (nominal 80%), LDWP excluded ===")
print("The two Augusts are the test: 2025-08 was 53.2 and 2026-08 was 77.5 under h.")
print("A method that adapts should pull BOTH toward 80, which no calendar method could.")
(
    scored.where(F.col("region") != "LDWP")
    .withColumn("test_month", F.date_format("period_utc", "yyyy-MM"))
    .groupBy("test_month")
    .agg(*COVERAGE)
    .orderBy("test_month")
    .show(24, truncate=False)
)

print("\n=== 3. Coverage and sharpness by region ===")
(
    scored.groupBy("region")
    .agg(
        *COVERAGE,
        F.round(F.avg("h_width")).alias("h_width_mw"),
        F.round(F.avg("c28_width")).alias("c28_width_mw"),
    )
    .orderBy("region")
    .show(12, truncate=False)
)

print("\n=== 4. Calibration summary across the 21 test months ===")
print("mean_abs_gap = mean |coverage - 80|. month_range = worst month minus best month.")
print("Reference: h scored 8.85 gap / 36.7 range; best calendar variant was 7.29 / 24.8.")
by_month = (
    scored.where(F.col("region") != "LDWP")
    .withColumn("test_month", F.date_format("period_utc", "yyyy-MM"))
    .groupBy("test_month")
    .agg(*[F.avg(f"{v}_cov").alias(v) for v in VARIANTS])
)
(
    by_month.agg(
        *[
            F.round(F.avg(F.abs(F.col(v) - F.lit(NOMINAL))) * 100, 2).alias(f"{v}_gap")
            for v in VARIANTS
        ],
        *[
            F.round((F.max(F.col(v)) - F.min(F.col(v))) * 100, 1).alias(f"{v}_range")
            for v in VARIANTS
        ],
    ).show(truncate=False)
)
