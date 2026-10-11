import math

from databricks.connect import DatabricksSession
from pyspark.sql import Window
from pyspark.sql import functions as F

from ingest import calendars, regions

spark = DatabricksSession.builder.serverless(True).getOrCreate()

# Spark's local-time functions read timestamps through the session timezone. Pin it to UTC
# so from_utc_timestamp below means exactly "this UTC instant, shown in local time".
spark.conf.set("spark.sql.session.timeZone", "UTC")

# --- Inputs -------------------------------------------------------------------------------

demand = spark.table("workspace.silver.demand")

# Archive weather only, for the baseline. One combined flag for any weather problem.
weather = (
    spark.table("workspace.silver.weather")
    .where(F.col("source") == "era5_archive")
    .select(
        "region",
        "period_utc",
        "temp_c",
        "humidity_pct",
        (F.col("has_gap") | F.col("temp_out_of_range") | F.col("humidity_out_of_range")).alias(
            "weather_gap"
        ),
    )
)

# Small lookup tables built in Python from ingest/, then sent to Spark.
timezones = spark.createDataFrame(
    [(code, r.timezone) for code, r in regions.REGIONS.items()],
    schema="region STRING, timezone STRING",
)

# Local dates near Jan 1 can fall in the previous year, so cover one extra year each side.
first_year, last_year = demand.agg(F.min(F.year("period_utc")), F.max(F.year("period_utc"))).first()
years = list(range(first_year - 1, last_year + 2))
holidays = spark.createDataFrame(
    [
        (code, day, name)
        for code in regions.REGIONS
        for day, name in calendars.holiday_dates(code, years).items()
    ],
    schema="region STRING, local_date DATE, holiday_name STRING",
)

# Days most businesses close: same dates for every region (see calendars.major_holiday_dates).
major_holidays = spark.createDataFrame(
    [(day, True) for day in calendars.major_holiday_dates(years)],
    schema="local_date DATE, is_major_holiday BOOLEAN",
)

# --- Join + calendar features -------------------------------------------------------------

gold = (
    demand.join(weather, on=["region", "period_utc"], how="left")
    # No weather row at all also counts as a weather gap.
    .withColumn("weather_gap", F.coalesce(F.col("weather_gap"), F.lit(True)))
    .join(F.broadcast(timezones), on="region")
    # Local time exists only to derive these features. period_utc stays the key.
    .withColumn("ts_local", F.from_utc_timestamp("period_utc", F.col("timezone")))
    .withColumn("local_date", F.to_date("ts_local"))
    .withColumn("hour_local", F.hour("ts_local"))
    .withColumn("dow", (F.dayofweek("ts_local") + 5) % 7)  # Monday = 0, Sunday = 6
    .withColumn("month", F.month("ts_local"))
    # Holidays are matched on the LOCAL date, not the UTC date.
    # holiday_name: every official holiday, kept for regime analysis.
    # is_major_holiday: only days most businesses close. This is the model feature.
    .join(F.broadcast(holidays), on=["region", "local_date"], how="left")
    .join(F.broadcast(major_holidays), on="local_date", how="left")
    .withColumn("is_major_holiday", F.coalesce(F.col("is_major_holiday"), F.lit(False)))
)

# --- Demand history: lags and rolling stats -----------------------------------------------
# Rows are counted, not hours: "48 rows back" is "48 hours back" only because silver
# guarantees exactly one row per region per hour. Every window ends at least 48h before the
# target hour, so nothing here is unknown at the 13:00 UTC issue time (CLAUDE.md section 9).

by_time = Window.partitionBy("region").orderBy("period_utc")
last_24h_ending_48h_ago = by_time.rowsBetween(-71, -48)
last_168h_ending_48h_ago = by_time.rowsBetween(-215, -48)

gold = (
    gold.withColumn("lag_48h", F.lag("demand_mw", 48).over(by_time))
    .withColumn("lag_168h", F.lag("demand_mw", 168).over(by_time))
    .withColumn("roll_mean_24h", F.avg("demand_mw").over(last_24h_ending_48h_ago))
    .withColumn("roll_mean_168h", F.avg("demand_mw").over(last_168h_ending_48h_ago))
    .withColumn("roll_std_168h", F.stddev("demand_mw").over(last_168h_ending_48h_ago))
)

# --- Cycles (Fourier terms) and degree-hours ---------------------------------------------
# A clock is a circle: sin/cos put 23:00 next to 00:00. Each harmonic k adds finer detail,
# so K pairs describe a smooth repeating shape (e.g. morning bump + evening peak) with 2K
# numbers instead of one category per hour. CLAUDE.md section 6 uses these for SARIMAX and
# for the Bayesian model's daily and weekly seasonality.

CYCLES = {
    # name: (position within the cycle, cycle length, number of harmonics)
    "day": (F.col("hour_local"), 24, 3),
    "week": (F.col("dow") * 24 + F.col("hour_local"), 168, 3),
    "year": (F.dayofyear("ts_local"), 365.25, 2),
}
for name, (position, period, harmonics) in CYCLES.items():
    for k in range(1, harmonics + 1):
        angle = 2 * math.pi * k * position / period
        gold = gold.withColumn(f"{name}_sin_{k}", F.sin(angle)).withColumn(
            f"{name}_cos_{k}", F.cos(angle)
        )

# Heating and cooling degree-hours: how far below / above 18 C. Turns the U-shaped
# temperature response into two features a linear model can use directly.
# greatest() skips nulls, so a missing temperature would silently become 0. Keep it null.
BASE_TEMP_C = 18.0
temp_known = F.col("temp_c").isNotNull()
gold = gold.withColumn(
    "hdh", F.when(temp_known, F.greatest(F.lit(BASE_TEMP_C) - F.col("temp_c"), F.lit(0.0)))
)
gold = gold.withColumn(
    "cdh", F.when(temp_known, F.greatest(F.col("temp_c") - F.lit(BASE_TEMP_C), F.lit(0.0)))
)

fourier_cols = [c for c in gold.columns if "_sin_" in c or "_cos_" in c]
gold = gold.select(
    # key
    "region",
    "period_utc",
    # target and its quality flag
    "demand_mw",
    "has_gap",
    # calendar (local time)
    "local_date",
    "hour_local",
    "dow",
    "month",
    "is_major_holiday",
    "holiday_name",
    # weather
    "temp_c",
    "humidity_pct",
    "hdh",
    "cdh",
    "weather_gap",
    # demand history
    "lag_48h",
    "lag_168h",
    "roll_mean_24h",
    "roll_mean_168h",
    "roll_std_168h",
    # cycles
    *fourier_cols,
)

# --- Save ---------------------------------------------------------------------------------

spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.gold")
(
    gold.write.mode("overwrite")
    .option("overwriteSchema", "true")  # allow columns to change between rebuilds
    .saveAsTable("workspace.gold.features")
)
saved = spark.table("workspace.gold.features")

# --- Checks -------------------------------------------------------------------------------

print("rows:", saved.count(), "(silver.demand has", demand.count(), ")")
print("columns:", len(saved.columns))

# Independent check that row-based lags are really time-based: join each row to the row
# exactly 48 / 168 hours earlier by timestamp, and count disagreements. Should be 0.
earlier = saved.select(
    "region", F.col("period_utc").alias("earlier_utc"), F.col("demand_mw").alias("earlier_mw")
)
for hours, col in ((48, "lag_48h"), (168, "lag_168h")):
    mismatches = (
        saved.join(
            earlier,
            (saved.region == earlier.region)
            & (earlier.earlier_utc == saved.period_utc - F.expr(f"INTERVAL {hours} HOURS")),
            how="left",
        )
        .where(~F.col(col).eqNullSafe(F.col("earlier_mw")))
        .count()
    )
    print(f"{col}: rows that disagree with a timestamp join = {mismatches}")

print("Daily cycle for one CISO day: 23:00 and 00:00 should be neighbors on the circle")
(
    saved.where((F.col("region") == "CISO") & (F.col("local_date") == "2024-07-10"))
    .select(
        "hour_local",
        F.round("day_sin_1", 3).alias("day_sin_1"),
        F.round("day_cos_1", 3).alias("day_cos_1"),
        F.round("temp_c", 1).alias("temp_c"),
        F.round("hdh", 1).alias("hdh"),
        F.round("cdh", 1).alias("cdh"),
    )
    .orderBy("hour_local")
    .show(24)
)
