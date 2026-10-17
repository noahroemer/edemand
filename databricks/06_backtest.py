"""Walk-forward backtest, starting with the seasonal-naive benchmark.

Design is in CLAUDE.md section 7 ("Backtest design"). In short:

    21 monthly folds, expanding window, test months 2025-01 .. 2026-09.
    Each fold trains on 2023-01-01 up to 48h before its test month (the lag embargo).
    Every model emits 19 quantiles per hour; CRPS = 2 x mean pinball loss over them.

Seasonal-naive intervals are volatility-scaled (CLAUDE.md section 7a):

    q_tau(t) = lag_168h(t) + sigma(t) * z_tau(region, hour_local)

    sigma(t)  mean |naive error| over the 7 days ending 48h before t. Known at issue time,
              so it is computed over the whole series, train and test alike.
    z_tau     quantiles of error / sigma, learned per fold from that fold's training rows.

Writes:
    workspace.gold.backtest_folds         fold definitions, shared by every model
    workspace.gold.backtest_predictions   one row per model x fold x region x hour
    workspace.gold.backtest_metrics       one row per model x fold x region

Each model replaces only its own rows, so models can be rerun independently.
"""

from datetime import date

from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

spark = DatabricksSession.builder.serverless(True).getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")

MODEL = "seasonal_naive"

TAUS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05, 0.10, ..., 0.95
I10, I50, I90 = TAUS.index(0.1), TAUS.index(0.5), TAUS.index(0.9)

TRAIN_START = "2023-01-01"
FIRST_TEST_MONTH = date(2025, 1, 1)
N_FOLDS = 21
EMBARGO_HOURS = 48
SIGMA_WINDOW_DAYS = 7
HOUR = 3600

FOLDS_TABLE = "workspace.gold.backtest_folds"
PREDICTIONS_TABLE = "workspace.gold.backtest_predictions"
METRICS_TABLE = "workspace.gold.backtest_metrics"


def add_months(d: date, n: int) -> date:
    years, month0 = divmod(d.month - 1 + n, 12)
    return date(d.year + years, month0 + 1, 1)


def write_model_rows(df: DataFrame, table: str) -> None:
    """Replace this model's rows only, leaving other models' results untouched."""
    if spark.catalog.tableExists(table):
        (df.write.mode("overwrite").option("replaceWhere", f"model = '{MODEL}'").saveAsTable(table))
    else:
        df.write.saveAsTable(table)


# --- Folds --------------------------------------------------------------------------------

folds = spark.createDataFrame(
    [
        (k + 1, add_months(FIRST_TEST_MONTH, k), add_months(FIRST_TEST_MONTH, k + 1))
        for k in range(N_FOLDS)
    ],
    schema="fold INT, test_start DATE, test_end DATE",
).select(
    "fold",
    F.lit(TRAIN_START).cast("timestamp").alias("train_start"),
    (F.col("test_start").cast("timestamp") - F.expr(f"INTERVAL {EMBARGO_HOURS} HOURS")).alias(
        "train_end"
    ),
    F.col("test_start").cast("timestamp").alias("test_start"),
    F.col("test_end").cast("timestamp").alias("test_end"),
)
folds.write.mode("overwrite").saveAsTable(FOLDS_TABLE)
folds = spark.table(FOLDS_TABLE)

# --- Seasonal-naive inputs ----------------------------------------------------------------

eligible = (
    spark.table("workspace.gold.features")
    .where(~F.col("has_gap") & F.col("lag_168h").isNotNull() & (F.col("demand_mw") > 0))
    .select("region", "period_utc", "hour_local", "demand_mw", "lag_168h")
    .withColumn("error", F.col("demand_mw") - F.col("lag_168h"))
    # rangeBetween needs a numeric ordering column, so order by epoch seconds.
    .withColumn("ts", F.unix_timestamp("period_utc"))
)

recent_week = (
    Window.partitionBy("region")
    .orderBy("ts")
    .rangeBetween(
        -(SIGMA_WINDOW_DAYS * 24 + EMBARGO_HOURS) * HOUR,
        -EMBARGO_HOURS * HOUR,
    )
)
eligible = eligible.withColumn("sigma", F.avg(F.abs("error")).over(recent_week)).where(
    F.col("sigma") > 0
)

# --- Fit: z quantiles per fold, from that fold's training rows only -----------------------

train = eligible.join(
    F.broadcast(folds),
    (F.col("period_utc") >= F.col("train_start")) & (F.col("period_utc") < F.col("train_end")),
)
z_quantiles = (
    train.withColumn("z", F.col("error") / F.col("sigma"))
    .groupBy("fold", "region", "hour_local")
    .agg(
        F.percentile_approx("z", TAUS, 10000).alias("z_q"),
        F.count("*").alias("n_train"),
    )
)

# --- Predict: test rows of each fold ------------------------------------------------------

test = eligible.join(
    F.broadcast(folds),
    (F.col("period_utc") >= F.col("test_start")) & (F.col("period_utc") < F.col("test_end")),
)
predictions = (
    test.join(z_quantiles, on=["fold", "region", "hour_local"])
    .select(
        F.lit(MODEL).alias("model"),
        "fold",
        "region",
        "period_utc",
        "hour_local",
        F.col("demand_mw").alias("actual"),
        # sigma > 0 and z_q is sorted, so the 19 quantiles can never cross.
        F.transform("z_q", lambda z: F.col("lag_168h") + F.col("sigma") * z).alias("quantiles"),
    )
    .withColumn("q10", F.col("quantiles")[I10])
    .withColumn("q50", F.col("quantiles")[I50])
    .withColumn("q90", F.col("quantiles")[I90])
)
write_model_rows(predictions, PREDICTIONS_TABLE)

# --- Score --------------------------------------------------------------------------------

taus = F.array(*[F.lit(t) for t in TAUS])
actual = F.col("actual")
scored = (
    spark.table(PREDICTIONS_TABLE)
    .where(F.col("model") == MODEL)
    .withColumn(
        "pinball",
        F.zip_with(
            "quantiles",
            taus,
            lambda q, tau: F.when(actual >= q, tau * (actual - q)).otherwise(
                (1 - tau) * (q - actual)
            ),
        ),
    )
    .withColumn("crps", 2 * F.aggregate("pinball", F.lit(0.0), lambda acc, x: acc + x) / len(TAUS))
    .withColumn(
        "pinball_served",
        (F.col("pinball")[I10] + F.col("pinball")[I50] + F.col("pinball")[I90]) / 3,
    )
    .withColumn("covered80", ((actual >= F.col("q10")) & (actual <= F.col("q90"))).cast("int"))
    .withColumn("ape", F.abs(actual - F.col("q50")) / actual)
    .withColumn("width80", F.col("q90") - F.col("q10"))
    .withColumn("error_q50", actual - F.col("q50"))
)

METRICS = [
    F.count("*").alias("n"),
    F.avg("crps").alias("crps"),
    F.avg("pinball_served").alias("pinball_served"),
    (100 * F.avg("covered80")).alias("coverage80"),
    F.avg("width80").alias("width80"),
    (100 * F.avg("ape")).alias("mape"),
    F.avg("error_q50").alias("bias"),
]
metrics = scored.groupBy("model", "fold", "region").agg(*METRICS)
write_model_rows(metrics, METRICS_TABLE)

# --- Report -------------------------------------------------------------------------------


def rounded(df: DataFrame) -> DataFrame:
    return df.select(
        *[
            F.round(c, 1).alias(c) if c in {"crps", "coverage80", "mape", "width80", "bias"} else c
            for c in df.columns
        ]
    )


print(f"=== {MODEL}: pooled over all 21 folds, by region ===")
rounded(scored.groupBy("region").agg(*METRICS).orderBy("region")).show()

print(
    "=== All regions, with and without LDWP (CRPS is in MW, so it is dominated by big regions) ==="
)
rounded(
    scored.withColumn("subset", F.lit("all 10"))
    .unionByName(scored.where(F.col("region") != "LDWP").withColumn("subset", F.lit("excl. LDWP")))
    .groupBy("subset")
    .agg(*METRICS)
    .orderBy("subset")
).show()

print("=== Coverage by fold (target 80), excluding LDWP ===")
rounded(
    scored.where(F.col("region") != "LDWP")
    .groupBy("fold")
    .agg(*METRICS)
    .orderBy("fold")
    .select("fold", "n", "coverage80", "mape", "crps")
).show(N_FOLDS)

smallest = z_quantiles.agg(F.min("n_train")).first()[0]
print(f"Smallest training group (fold x region x hour): {smallest} rows")
