"""Scoring and writing shared by every Databricks backtest job (06_backtest, 04_train_fleet).

Metric definitions mirror models/scoring.py exactly (CLAUDE.md section 7): 19 quantiles,
CRPS = 2 x mean pinball loss, coverage of q10..q90, MAPE on q50. One definition, so no two
models can be scored differently.
"""

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

TAUS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05, 0.10, ..., 0.95
I10, I50, I90 = TAUS.index(0.1), TAUS.index(0.5), TAUS.index(0.9)

FOLDS_TABLE = "workspace.gold.backtest_folds"
PREDICTIONS_TABLE = "workspace.gold.backtest_predictions"
METRICS_TABLE = "workspace.gold.backtest_metrics"
N_FOLDS = 21


def write_model_rows(spark: SparkSession, df: DataFrame, table: str, model: str) -> None:
    """Replace this model's rows only, leaving other models' results untouched."""
    if spark.catalog.tableExists(table):
        df.write.mode("overwrite").option("replaceWhere", f"model = '{model}'").saveAsTable(table)
    else:
        df.write.saveAsTable(table)


def metric_columns() -> list[Column]:
    return [
        F.count("*").alias("n"),
        F.avg("crps").alias("crps"),
        F.avg("pinball_served").alias("pinball_served"),
        (100 * F.avg("covered80")).alias("coverage80"),
        F.avg("width80").alias("width80"),
        (100 * F.avg("ape")).alias("mape"),
        F.avg("error_q50").alias("bias"),
    ]


def score(spark: SparkSession, model: str) -> DataFrame:
    """Per-hour scores for one model's saved predictions; writes per-fold metrics."""
    taus = F.array(*[F.lit(t) for t in TAUS])
    actual = F.col("actual")
    scored = (
        spark.table(PREDICTIONS_TABLE)
        .where(F.col("model") == model)
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
        .withColumn(
            "crps", 2 * F.aggregate("pinball", F.lit(0.0), lambda acc, x: acc + x) / len(TAUS)
        )
        .withColumn(
            "pinball_served",
            (F.col("pinball")[I10] + F.col("pinball")[I50] + F.col("pinball")[I90]) / 3,
        )
        .withColumn("covered80", ((actual >= F.col("q10")) & (actual <= F.col("q90"))).cast("int"))
        .withColumn("ape", F.abs(actual - F.col("q50")) / actual)
        .withColumn("width80", F.col("q90") - F.col("q10"))
        .withColumn("error_q50", actual - F.col("q50"))
    )
    metrics = scored.groupBy("model", "fold", "region").agg(*metric_columns())
    write_model_rows(spark, metrics, METRICS_TABLE, model)
    return scored


def _rounded(df: DataFrame) -> DataFrame:
    keep = {"crps", "coverage80", "mape", "width80", "bias"}
    return df.select(*[F.round(c, 1).alias(c) if c in keep else c for c in df.columns])


def report(scored: DataFrame, model: str) -> None:
    print(f"=== {model}: pooled over all folds, by region ===")
    _rounded(scored.groupBy("region").agg(*metric_columns()).orderBy("region")).show()

    print("=== All regions, with and without LDWP (CRPS is in MW, dominated by big regions) ===")
    _rounded(
        scored.withColumn("subset", F.lit("all 10"))
        .unionByName(
            scored.where(F.col("region") != "LDWP").withColumn("subset", F.lit("excl. LDWP"))
        )
        .groupBy("subset")
        .agg(*metric_columns())
        .orderBy("subset")
    ).show()

    print("=== By fold (coverage target 80), excluding LDWP ===")
    _rounded(
        scored.where(F.col("region") != "LDWP")
        .groupBy("fold")
        .agg(*metric_columns())
        .orderBy("fold")
        .select("fold", "n", "coverage80", "mape", "crps")
    ).show(N_FOLDS)
