"""Station popularity analysis for May 2022 Citibike data."""

import xorq.api as xo

con = xo.connect()

# Load the filtered electric bike data
raw = con.read_parquet("/Users/paddy/xorq_mcp/data/202205-citibike-tripdata.parquet")
electric_bikes = raw.filter(raw.rideable_type == "electric_bike").cast(
    {"started_at": "timestamp", "ended_at": "timestamp"}
)

# Compute trip duration in minutes via epoch seconds
with_duration = electric_bikes.mutate(
    duration_minutes=(
        electric_bikes.ended_at.epoch_seconds() - electric_bikes.started_at.epoch_seconds()
    ) / 60
)

# count(ride_id), avg(duration_minutes) group by start_station_name
expr = with_duration.group_by("start_station_name").agg(
    trip_count=with_duration.ride_id.count(),
    avg_duration_minutes=with_duration.duration_minutes.mean(),
)
