import sys
from datetime import UTC, datetime, timedelta

from databricks.connect import DatabricksSession

from ingest import open_meteo, regions

spark = DatabricksSession.builder.serverless(True).getOrCreate()

spark.sql("""
    CREATE TABLE IF NOT EXISTS workspace.bronze.weather_raw (
        region        STRING,
        period_utc    TIMESTAMP,
        source        STRING,
        latitude      DOUBLE,
        longitude     DOUBLE,
        temp_c        DOUBLE,
        humidity_pct  DOUBLE,
        fetched_at    TIMESTAMP
    ) USING DELTA
""")

# Three sources, all from the start of 2023:
#   era5_archive        - what the weather actually was. Stops ARCHIVE_LAG_DAYS ago: Open-Meteo
#                         fills the most recent days with forecast data (even for hours that
#                         haven't happened yet), which must not be stored as "actual" weather.
#   historical_forecast - stitched from the start of each forecast run. Scores like actual
#                         weather in the backtest, so it can't measure leakage. Kept for record.
#   previous_day1       - what the forecast said one day earlier (~24-47h lead). The honest
#                         input for backtests (section 7).
# Forecast sources stop at the end of yesterday; future hours belong to the live forecast.
#
# Pass source names to load only those, e.g.  python databricks/01_bronze_weather.py previous_day1
START_DATE = "2023-01-01"
today = datetime.now(UTC).date()
yesterday = today - timedelta(days=1)
ALL_SOURCES = {
    "era5_archive": (
        open_meteo.fetch_archive,
        today - timedelta(days=open_meteo.ARCHIVE_LAG_DAYS),
    ),
    "historical_forecast": (open_meteo.fetch_historical_forecast, yesterday),
    "previous_day1": (open_meteo.fetch_previous_day, yesterday),
}
# Arguments are source names, region codes, or both, in any order. Naming regions matters
# for backfills: adding the CAISO sub-areas should not re-fetch four years of weather for the
# ten regions that already have it.
#     python databricks/01_bronze_weather.py previous_day1
#     python databricks/01_bronze_weather.py PGAE SCE SDGE VEA
args = sys.argv[1:]
unknown = [a for a in args if a not in ALL_SOURCES and a.upper() not in regions.ALL_REGIONS]
if unknown:
    raise SystemExit(
        f"Unknown argument(s) {sorted(unknown)}. "
        f"Sources: {sorted(ALL_SOURCES)}. Regions: {sorted(regions.ALL_REGIONS)}"
    )
SOURCES = {
    name: ALL_SOURCES[name] for name in ([a for a in args if a in ALL_SOURCES] or ALL_SOURCES)
}

# One request per region, so every row is tagged with the region it was asked for.
# (CISO and LDWP share coordinates, so lat/lon alone can't tell them apart.)
# ALL_REGIONS, not REGIONS: the four CAISO sub-areas need their own weather. Their load
# centers are genuinely different climates -- coastal San Diego, inland Pahrump -- which is
# half the reason their temperature responses differ enough to be worth pooling rather than
# merging. SCE's point (Pasadena) sits close to CISO's (LA basin) but is kept separate so
# every region's weather is traceable to its own request.
codes = [a.upper() for a in args if a.upper() in regions.ALL_REGIONS] or list(regions.ALL_REGIONS)
rows = []
for source, (fetch, end_date) in SOURCES.items():
    print(f"{source}, {START_DATE} to {end_date}")
    for i, code in enumerate(codes, start=1):
        print(f"  [{i}/{len(codes)}] {code}...", end=" ", flush=True)
        region_rows = fetch(regions.coordinates([code]), START_DATE, end_date.isoformat())
        for row in region_rows:
            row["region"] = code
            # Open-Meteo sends humidity as a whole number; store every reading as a float.
            for col in ("temp_c", "humidity_pct"):
                if row[col] is not None:
                    row[col] = float(row[col])
        rows.extend(region_rows)
        print(f"{len(region_rows)} rows")
print(f"fetched {len(rows)} rows")

df = spark.createDataFrame(
    rows,
    schema="region STRING, period_utc TIMESTAMP, source STRING, latitude DOUBLE, "
    "longitude DOUBLE, temp_c DOUBLE, humidity_pct DOUBLE, fetched_at TIMESTAMP",
)
df.createOrReplaceTempView("weather_updates")

# source is part of the key: archive and forecast weather for the same hour are different rows.
spark.sql("""
    MERGE INTO workspace.bronze.weather_raw AS target
    USING weather_updates AS source
    ON  target.region     = source.region
    AND target.source     = source.source
    AND target.period_utc = source.period_utc
    WHEN MATCHED     THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

spark.sql("""
    SELECT region, source, COUNT(*) AS n, COUNT(temp_c) AS has_temp,
           MIN(period_utc) AS first, MAX(period_utc) AS last
    FROM workspace.bronze.weather_raw
    GROUP BY region, source ORDER BY region, source
""").show()
