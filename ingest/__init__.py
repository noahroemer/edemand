"""Data acquisition for the demand forecasting pipeline.

Pure-Python HTTP clients and time/calendar utilities. Deliberately contains **no Spark and
no pandas**: these modules return `list[dict]` records that hand straight to
`spark.createDataFrame(...)`, keeping CLAUDE.md section 12's "write PySpark first" rule
intact and letting the clients be unit-tested off-platform with no cluster.

This package is also independent of the section 13 question 1 outcome — the same clients
work whether ingestion ends up running as a Databricks job or in GitHub Actions.

Modules:
    regions     balancing-authority metadata: codes, timezones, weather coordinates
    timeutils   UTC discipline, DST-safe calendar features, gap detection
    eia         EIA API v2 hourly demand, paginated and revision-aware
    open_meteo  Open-Meteo reanalysis, archived forecasts, and live forecasts
    calendars   US federal and state holiday flags
"""

__version__ = "0.1.0"
