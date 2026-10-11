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

# One request per region, so every row is tagged with the region it was asked for.
# (CISO and LDWP share coordinates, so lat/lon alone can't tell them apart.)
codes = list(regions.REGIONS)
rows = []
for i, code in enumerate(codes, start=1):
    print(f"[{i}/{len(codes)}] fetching {code}...", end=" ", flush=True)
    region_rows = open_meteo.fetch_archive(regions.coordinates([code]), "2024-01-01", "2024-12-31")
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
    SELECT region, COUNT(*) AS n, COUNT(temp_c) AS has_temp
    FROM workspace.bronze.weather_raw
    GROUP BY region ORDER BY region
""").show()
