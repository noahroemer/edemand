"""Does seasonal stratification fix the calibration failure? Option A, measured.

eda_naive_drift.py established the diagnosis: the naive interval offsets are pooled across
all twelve months, but error width is strongly seasonal (CISO MAE 3,571 in September
against 1,023 in November). Feeding one all-year width into a summer month makes it too
narrow and into November too wide, which is exactly what coverage showed - 62-73% in
Jun/Jul and Dec/Jan, 85-90% in Mar-May and Oct-Nov.

It is the same pooled-vs-stratified failure already fixed in the hourly dimension, one
level up.

Three variants, same single split, so the comparison is clean:

    h    (region, hour_local)            240 groups   - current baseline
    hs   (region, hour_local, season)    960 groups   - meteorological seasons
    hm   (region, hour_local, month)   2,880 groups   - option A

The tension is sample size against resolution. At fold 1 the training window is 24 months,
so an `hm` group holds only ~60 errors and each tail quantile rests on ~6 points. Thin
tails are noisy. `hs` has ~4x the data per group but pools September with October and
November - and September is the hardest month of the year while those two are among the
easiest, so that particular pooling is suspect.

Whether the extra resolution beats the extra noise is an empirical question, which is what
this script answers. Block 3 reports the smallest training group, since that is where the
thin-tail risk actually bites.

Reads workspace.gold.features. No writes.
"""

from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

spark = DatabricksSession.builder.serverless(True).getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")

TRAIN_END = "2025-01-01"
TEST_END = "2026-10-01"
NOMINAL = 0.80
QUANTILES = [0.1, 0.5, 0.9]

features = spark.table("workspace.gold.features")

eligible = (
    features.where(~F.col("has_gap") & F.col("lag_168h").isNotNull() & (F.col("demand_mw") > 0))
    .withColumn("error", F.col("demand_mw") - F.col("lag_168h"))
    # Meteorological seasons: DJF / MAM / JJA / SON.
    .withColumn(
        "season",
        F.when(F.col("month").isin(12, 1, 2), "DJF")
        .when(F.col("month").isin(3, 4, 5), "MAM")
        .when(F.col("month").isin(6, 7, 8), "JJA")
        .otherwise("SON"),
    )
)

train = eligible.where(F.col("period_utc") < TRAIN_END)
test = eligible.where((F.col("period_utc") >= TRAIN_END) & (F.col("period_utc") < TEST_END))


def fit_offsets(keys: list[str], prefix: str) -> DataFrame:
    """Empirical error quantiles per stratum - the whole 'fit' for seasonal-naive.

    Column names are prefixed so three variants can be joined onto the same test rows and
    scored side by side in one pass.
    """
    return (
        train.groupBy(*keys)
        .agg(
            F.percentile_approx("error", QUANTILES).alias("q"),
            F.count("*").alias(f"{prefix}_n_train"),
        )
        .select(
            *keys,
            f"{prefix}_n_train",
            F.col("q")[0].alias(f"{prefix}_e10"),
            F.col("q")[2].alias(f"{prefix}_e90"),
        )
    )


VARIANTS = {
    "h": ["region", "hour_local"],
    "hs": ["region", "hour_local", "season"],
    "hm": ["region", "hour_local", "month"],
}

scored = test
for prefix, keys in VARIANTS.items():
    # broadcast: the largest offset table is 2,880 rows, so this avoids shuffling the
    # ~320k-row test set three times.
    scored = scored.join(F.broadcast(fit_offsets(keys, prefix)), on=keys, how="left")

for prefix in VARIANTS:
    scored = (
        scored.withColumn(f"{prefix}_q10", F.col("lag_168h") + F.col(f"{prefix}_e10"))
        .withColumn(f"{prefix}_q90", F.col("lag_168h") + F.col(f"{prefix}_e90"))
        .withColumn(
            f"{prefix}_cov",
            (
                (F.col("demand_mw") >= F.col(f"{prefix}_q10"))
                & (F.col("demand_mw") <= F.col(f"{prefix}_q90"))
            ).cast("int"),
        )
        .withColumn(f"{prefix}_width", F.col(f"{prefix}_q90") - F.col(f"{prefix}_q10"))
    )

COVERAGE = [F.round(F.avg(f"{p}_cov") * 100, 1).alias(f"{p}_cover") for p in VARIANTS]

print("\n=== 1. Coverage by test month (nominal 80%), LDWP excluded ===")
print("The failure was Jun/Jul and Dec/Jan too narrow, Mar-May and Oct-Nov too wide.")
print("A fix should pull both ends toward 80, not just improve the average.")
(
    scored.where(F.col("region") != "LDWP")
    .withColumn("test_month", F.date_format("period_utc", "yyyy-MM"))
    .groupBy("test_month")
    .agg(*COVERAGE)
    .orderBy("test_month")
    .show(24, truncate=False)
)

print("\n=== 2. Coverage and sharpness by region ===")
print("Watch width too: a variant that fixes coverage only by widening everywhere")
print("has bought calibration at the cost of sharpness, which CRPS will penalise.")
(
    scored.groupBy("region")
    .agg(
        *COVERAGE,
        F.round(F.avg("h_width")).alias("h_width_mw"),
        F.round(F.avg("hm_width")).alias("hm_width_mw"),
    )
    .orderBy("region")
    .show(12, truncate=False)
)

print("\n=== 3. Calibration error across test months + thin-group risk ===")
print("mean_abs_gap = mean |coverage - 80| over the 21 test months. Lower is better.")
print("min_n_train is the smallest training stratum: where noisy tail quantiles bite.")
by_month = (
    scored.where(F.col("region") != "LDWP")
    .withColumn("test_month", F.date_format("period_utc", "yyyy-MM"))
    .groupBy("test_month")
    .agg(*[F.avg(f"{p}_cov").alias(p) for p in VARIANTS])
)
(
    by_month.agg(
        *[
            F.round(F.avg(F.abs(F.col(p) - F.lit(NOMINAL))) * 100, 2).alias(f"{p}_mean_abs_gap")
            for p in VARIANTS
        ],
        *[
            F.round((F.max(F.col(p)) - F.min(F.col(p))) * 100, 1).alias(f"{p}_month_range")
            for p in VARIANTS
        ],
    ).show(truncate=False)
)
(
    scored.agg(*[F.min(f"{p}_n_train").alias(f"{p}_min_n_train") for p in VARIANTS]).show(
        truncate=False
    )
)
