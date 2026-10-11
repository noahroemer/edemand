from datetime import UTC, datetime

from databricks.connect import DatabricksSession
from dotenv import load_dotenv

from ingest import eia, regions

load_dotenv()
spark = DatabricksSession.builder.serverless(True).getOrCreate()

# 1. Make sure the schema and table exist (safe to rerun).
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.bronze")
spark.sql("""
    CREATE TABLE IF NOT EXISTS workspace.bronze.demand_raw (
        region      STRING,
        period_utc  TIMESTAMP,
        demand_mw   DOUBLE,
        data_type   STRING,
        fetched_at  TIMESTAMP
    ) USING DELTA
""")

# 2. Call the API (runs on your laptop).
# From the start of 2023 up to the current hour. Rerunning re-fetches hours already loaded,
# which is how EIA revisions get picked up; the MERGE below makes that safe.
START = datetime(2023, 1, 1, tzinfo=UTC)
END = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
print(f"fetching demand {START:%Y-%m-%d} to {END:%Y-%m-%d %H:00} UTC")
rows = eia.fetch_many(list(regions.REGIONS), START, END)  # the 10 region codes
print(f"fetched {len(rows)} rows")

# Row count per region, so a short region (missing hours) stands out.
counts: dict[str, int] = {}
for row in rows:
    counts[row["region"]] = counts.get(row["region"], 0) + 1
for code, n in counts.items():
    print(f"  {code}: {n}")

# 3. Python list -> Spark DataFrame, with the schema spelled out.
df = spark.createDataFrame(
    rows,
    schema="region STRING, period_utc TIMESTAMP, demand_mw DOUBLE, data_type STRING, fetched_at TIMESTAMP",
)

# 4. Give the DataFrame a name so SQL can refer to it.
df.createOrReplaceTempView("demand_updates")

# 5. Upsert: update hours we already have, insert hours we don't.
spark.sql("""
    MERGE INTO workspace.bronze.demand_raw AS target
    USING demand_updates AS source
    ON  target.region     = source.region
    AND target.period_utc = source.period_utc
    WHEN MATCHED     THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

spark.sql("SELECT COUNT(*) AS n FROM workspace.bronze.demand_raw").show()
