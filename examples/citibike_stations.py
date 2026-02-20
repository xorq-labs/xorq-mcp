import xorq.api as xo

t = xo.read_parquet("/Users/paddy/buckaroo/docs/example-notebooks/citibike-trips-2016-04.parq")

expr = (
    t.group_by("start station name")
    .agg(
        trip_count=t["start station name"].count(),
        avg_tripduration=t.tripduration.mean(),
    )
    .order_by(xo.desc("trip_count"))
)
