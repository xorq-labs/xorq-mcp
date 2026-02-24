"""Reproduction case: union of multiple read_csv calls fails at build execution time.

Each con.read_csv() registers a table with an auto-generated name in the
DataFusion session context.  When xorq builds and executes the expression,
only the first table survives the build/execute round-trip — subsequent
tables are not re-registered, so DataFusion's planner raises:

    Error during planning: table 'datafusion.public.ibis_read_csv_<hash>' not found

This script reproduces the failure with two tiny in-memory CSV files so no
real data files are needed.  Run with:

    python repro_union_table_not_found.py

Expected (broken): Error during planning: table '...' not found
Workaround:        use a single glob pattern that matches all files instead
                   of separate read_csv() calls joined with .union()
"""

import functools
import tempfile
import textwrap
from pathlib import Path

import pyarrow as pa
import xorq.api as xo
from xorq.ibis_yaml.compiler import build_expr, load_expr

# ---------------------------------------------------------------------------
# Create two tiny CSV files in a temp directory
# ---------------------------------------------------------------------------
SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("value", pa.string()),
    ]
)

CSV_A = textwrap.dedent("""\
    id,value
    1,alpha
    2,beta
""")

CSV_B = textwrap.dedent("""\
    id,value
    3,gamma
    4,delta
""")

tmpdir = Path(tempfile.mkdtemp())
file_a = tmpdir / "part_a.csv"
file_b = tmpdir / "part_b.csv"
file_a.write_text(CSV_A)
file_b.write_text(CSV_B)

print(f"Temp files: {tmpdir}")

# ---------------------------------------------------------------------------
# Failing pattern: two separate read_csv() calls reduced with .union()
# ---------------------------------------------------------------------------
con = xo.connect()

t_a = con.read_csv(str(file_a), schema=SCHEMA)
t_b = con.read_csv(str(file_b), schema=SCHEMA)
expr = functools.reduce(lambda a, b: a.union(b), [t_a, t_b])

print("\n--- FAILING PATTERN (separate read_csv + union) ---")
try:
    build_path = build_expr(expr)
    print(f"  build OK: {build_path}")
    loaded = load_expr(build_path)
    out = Path(tempfile.mktemp(suffix=".parquet"))
    loaded.to_parquet(str(out))   # <-- fails here
    print(f"  execute OK: {out}")
except Exception as e:
    print(f"  ERROR: {e}")

# ---------------------------------------------------------------------------
# Workaround: single read_csv with a glob pattern
# ---------------------------------------------------------------------------
con2 = xo.connect()
expr_glob = con2.read_csv(str(tmpdir / "part_*.csv"), schema=SCHEMA)

print("\n--- WORKAROUND (single glob read_csv) ---")
try:
    build_path2 = build_expr(expr_glob)
    print(f"  build OK: {build_path2}")
    loaded2 = load_expr(build_path2)
    out2 = Path(tempfile.mktemp(suffix=".parquet"))
    loaded2.to_parquet(str(out2))
    print(f"  execute OK: {out2}")
    print("  Workaround succeeded.")
except Exception as e:
    print(f"ERROR: {e}")
