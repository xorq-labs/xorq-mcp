"""Postgres connection helper with defaults matching the docker compose config.

Usage:
    from libs.postgres_helpers import connect_postgres

    pg = connect_postgres()
    pg = connect_postgres(database="caching")
"""

import xorq.api as xo


POSTGRES_DEFAULTS = dict(
    host="localhost",
    user="postgres",
    password="postgres",
    port=5432,
    database="ibis_testing",
)


def connect_postgres(**kwargs):
    """Connect to local postgres with docker compose defaults.

    Override any parameter by passing it as a keyword argument.
    Set POSTGRES_* environment variables to override globally.
    """
    return xo.postgres.connect_env(**{**POSTGRES_DEFAULTS, **kwargs})
