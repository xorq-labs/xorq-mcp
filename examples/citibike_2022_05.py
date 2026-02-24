"""Load May 2022 Citibike trip data into xorq catalog."""

import pandas as pd
import pyarrow as pa
import xorq.api as xo
from datetime import datetime, timedelta
import random

# Generate sample May 2022 Citibike data
random.seed(42)

num_trips = 500000
start_date = datetime(2022, 5, 1)
end_date = datetime(2022, 5, 31, 23, 59, 59)

# Build a fixed station registry so each station_name maps to one consistent ID
stations = {f"Station {i}": str(1000 + i) for i in range(1, 501)}
station_names = list(stations.keys())

trip_data = []
for i in range(num_trips):
    started_at = start_date + timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds()))
    )
    duration = random.randint(60, 3600)
    ended_at = started_at + timedelta(seconds=duration)

    start_name = random.choice(station_names)
    end_name = random.choice(station_names)

    trip = {
        "ride_id": f"ride_{i}",
        "rideable_type": random.choice(["classic_bike", "electric_bike", "docked_bike"]),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "start_station_name": start_name,
        "start_station_id": stations[start_name],
        "end_station_name": end_name,
        "end_station_id": stations[end_name],
        "start_lat": 40.7 + random.uniform(-0.1, 0.1),
        "start_lng": -74.0 + random.uniform(-0.1, 0.1),
        "end_lat": 40.7 + random.uniform(-0.1, 0.1),
        "end_lng": -74.0 + random.uniform(-0.1, 0.1),
        "member_casual": random.choice(["member", "casual"]),
    }
    trip_data.append(trip)

# Create DataFrame and save as parquet
df = pd.DataFrame(trip_data)
df.to_parquet("/Users/paddy/xorq_mcp/data/202205-citibike-tripdata.parquet")

# Load into xorq catalog
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
raw = con.read_parquet("/Users/paddy/xorq_mcp/data/202205-citibike-tripdata.parquet")
expr = raw.filter(raw.rideable_type == "electric_bike").cast(
    {"started_at": "timestamp", "ended_at": "timestamp"}
)
