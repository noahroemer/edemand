"""Fleet job: fit one GAM per (region, fold) in parallel on Spark, via applyInPandas.

The model code is models/gam.py, unchanged: the same function the local harness
(scripts/backtest_local.py) validated. Two things make that possible on serverless:

    DatabricksEnv().withDependencies(...)   installs pygam on the Spark workers
    spark.addArtifacts(models.zip, pyfile)  ships the models/ package to them

Each Spark task receives one region's history up to the end of one test month, fits the
GAM, and returns that month's 19 quantiles per hour. 10 regions x 21 folds = 210 tasks.
Forecasts use the forecast issued a day earlier (previous_day1), the honest setting
(CLAUDE.md section 7). Results land in the shared backtest tables as model 'gam'.

    python databricks/04_train_fleet.py                  # all regions, all folds
    python databricks/04_train_fleet.py CISO 1           # smoke test: one region, one fold
"""

import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from backtest_common import FOLDS_TABLE, I10, I50, I90, PREDICTIONS_TABLE, report, score
from backtest_common import write_model_rows as write_rows
from databricks.connect import DatabricksEnv, DatabricksSession
from pyspark.sql import functions as F

MODEL = "gam"
WEATHER = "temp_c_pd1"
REPO = Path(__file__).resolve().parent.parent

env = DatabricksEnv().withDependencies(["pygam==0.12.0"])
spark = DatabricksSession.builder.withEnvironment(env).serverless(True).getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")

# Ship models/ to the workers as an importable zip.
bundle = Path(tempfile.mkdtemp()) / "models.zip"
with zipfile.ZipFile(bundle, "w") as zf:
    for path in (REPO / "models").glob("*.py"):
        zf.write(path, f"models/{path.name}")
spark.addArtifacts(str(bundle), pyfile=True)

only_region = sys.argv[1] if len(sys.argv) > 1 else None
only_fold = int(sys.argv[2]) if len(sys.argv) > 2 else None

# --- Inputs: gold features + day-ahead forecast temperature ------------------------------

COLUMNS = [
    "region",
    "period_utc",
    "demand_mw",
    "has_gap",
    "local_date",
    "hour_local",
    "dow",
    "is_major_holiday",
    "temp_c",
    "lag_168h",
    "lag_48h",
]
gold = spark.table("workspace.gold.features")
if "parent_region" in gold.columns:  # the 10 balancing authorities, not the CAISO children
    gold = gold.where(F.col("parent_region").isNull())
forecast = (
    spark.table("workspace.silver.weather")
    .where((F.col("source") == "previous_day1") & ~F.col("has_gap"))
    .select("region", "period_utc", F.col("temp_c").alias(WEATHER))
)
history = gold.select(*COLUMNS).join(forecast, ["region", "period_utc"], "left")

folds = spark.table(FOLDS_TABLE)
if only_region:
    history = history.where(F.col("region") == only_region)
if only_fold:
    folds = folds.where(F.col("fold") == only_fold)

# One copy of each region's history per fold, cut at that fold's test_end.
tasks = history.join(F.broadcast(folds), F.col("period_utc") < F.col("test_end"))

# --- Fit: one pandas call per (region, fold) ----------------------------------------------

OUTPUT_SCHEMA = (
    "model STRING, fold INT, region STRING, period_utc TIMESTAMP, hour_local INT, "
    "actual DOUBLE, quantiles ARRAY<DOUBLE>, q10 DOUBLE, q50 DOUBLE, q90 DOUBLE"
)


def fit_one(pdf: pd.DataFrame) -> pd.DataFrame:
    from models import Fold, gam  # resolved on the worker, from the shipped zip

    fold = Fold.from_row(pdf.iloc[0])
    out = gam.fit_predict(pdf, fold, weather=WEATHER)
    q = list(out.quantiles)
    return pd.DataFrame(
        {
            "model": MODEL,
            "fold": fold.fold,
            "region": pdf.region.iloc[0],
            "period_utc": out.period_utc,
            "hour_local": out.hour_local.astype("int32"),
            "actual": out.actual,
            "quantiles": [list(map(float, row)) for row in q],
            "q10": [row[I10] for row in q],
            "q50": [row[I50] for row in q],
            "q90": [row[I90] for row in q],
        }
    )


predictions = tasks.groupBy("region", "fold").applyInPandas(fit_one, schema=OUTPUT_SCHEMA)

if only_region or only_fold:
    sample = predictions.toPandas()
    print(f"smoke test: {len(sample)} hours")
    print(sample[["region", "fold", "period_utc", "actual", "q10", "q50", "q90"]].head())
    sys.exit(0)

write_rows(spark, predictions, PREDICTIONS_TABLE, MODEL)
report(score(spark, MODEL), MODEL)
