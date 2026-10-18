"""Snapshot the modeling tables to local Parquet, so model development doesn't spend
Databricks Free Edition compute (the quota has run out twice).

Writes to data/ (gitignored):
    gold_features.parquet         workspace.gold.features
    forecast_weather.parquet      forecast-source rows of silver.weather (historical_forecast,
                                  previous_day1), for the weather-leakage ablation (section 7)
    backtest_folds.parquet        the 21 fold definitions every model must use
    naive_predictions.parquet     seasonal-naive backtest output, to compare against

Rerun after any pipeline change. The official backtest still runs on Databricks.

    python scripts/export_gold.py
"""

from pathlib import Path

from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

spark = DatabricksSession.builder.serverless(True).getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")

EXPORTS = {
    "gold_features": spark.table("workspace.gold.features"),
    # Forecast weather sources, long format: one row per (region, source, hour).
    "forecast_weather": spark.table("workspace.silver.weather")
    .where(F.col("source").isin("historical_forecast", "previous_day1") & ~F.col("has_gap"))
    .select("region", "source", "period_utc", "temp_c", "humidity_pct"),
    "backtest_folds": spark.table("workspace.gold.backtest_folds"),
    "naive_predictions": spark.table("workspace.gold.backtest_predictions")
    .where(F.col("model") == "seasonal_naive")
    .select("fold", "region", "period_utc", "actual", "q10", "q50", "q90"),
}

for name, df in EXPORTS.items():
    pdf = df.toPandas()
    # Spark Connect stashes query metrics in .attrs, which pandas can't write to Parquet.
    pdf.attrs = {}
    path = OUT / f"{name}.parquet"
    pdf.to_parquet(path, index=False)
    print(f"{name:20s} {len(pdf):>8,} rows  {path.stat().st_size / 1e6:6.1f} MB")
