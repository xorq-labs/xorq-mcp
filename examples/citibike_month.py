"""Load a single month of Citibike 2020 trip data from CSV(s).

Usage:
    Set CITIBIKE_GLOB to a glob pattern for the monthly CSV files,
    e.g. "/path/to/202001-citibike-tripdata_*.csv"
"""

import os

import pyarrow as pa
import xorq.api as xo

glob_pattern = os.environ.get("CITIBIKE_GLOB", "")
if not glob_pattern:
    raise RuntimeError("Set CITIBIKE_GLOB to a glob pattern for the monthly CSV files")

# Station IDs are mixed numeric/string (e.g. "7052.01" vs "SYS035"),
# so we provide an explicit schema to avoid type inference issues.
schema = pa.schema(
    [
        ("ride_id", pa.string()),
        ("rideable_type", pa.string()),
        ("started_at", pa.string()),
        ("ended_at", pa.string()),
        ("start_station_name", pa.string()),
        ("start_station_id", pa.string()),
        ("end_station_name", pa.string()),
        ("end_station_id", pa.string()),
        ("start_lat", pa.float64()),
        ("start_lng", pa.float64()),
        ("end_lat", pa.float64()),
        ("end_lng", pa.float64()),
        ("member_casual", pa.string()),
    ]
)

con = xo.connect()
raw = con.read_csv(glob_pattern, schema=schema)
expr = raw.cast({"started_at": "timestamp", "ended_at": "timestamp"})
