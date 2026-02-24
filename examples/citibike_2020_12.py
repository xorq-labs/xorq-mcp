"""Citibike 2020 month 12 trip data."""
import pyarrow as pa
import xorq.api as xo

schema = pa.schema([
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
])

con = xo.connect()
raw = con.read_csv(
    "/Users/paddy/xorq_mcp/data/citibike_2020/2020-citibike-tripdata/202012-citibike-tripdata_*.csv",
    schema=schema,
)
expr = raw.cast({"started_at": "timestamp", "ended_at": "timestamp"})
