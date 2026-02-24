"""Tests for Buckaroo server reliability, session management, and error handling.

These tests use real Buckaroo + web servers to exercise actual failure modes.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

from tests.conftest import (
    BUCKAROO_TEST_PORT,
    WEB_TEST_PORT,
    _health,
    _load_session,
    _wait_healthy,
)


# ---------------------------------------------------------------------------
# Buckaroo /load API reliability
# ---------------------------------------------------------------------------
class TestBuckarooLoadAPI:
    def test_load_returns_correct_metadata(self, buckaroo_server, small_parquet):
        resp = _load_session(small_parquet, "meta_check", BUCKAROO_TEST_PORT)
        assert resp["session"] == "meta_check"
        assert resp["rows"] == 100
        assert "columns" in resp
        assert "server_pid" in resp

    def test_load_same_session_twice(self, buckaroo_server, small_parquet):
        """Second load of same session should overwrite, not error."""
        r1 = _load_session(small_parquet, "double_load", BUCKAROO_TEST_PORT)
        r2 = _load_session(small_parquet, "double_load", BUCKAROO_TEST_PORT)
        assert r1["rows"] == r2["rows"]

    def test_load_missing_session_field(self, buckaroo_server, small_parquet):
        """Omitting 'session' from the payload should return 400."""
        payload = json.dumps({"path": small_parquet}).encode()
        req = Request(
            f"http://localhost:{BUCKAROO_TEST_PORT}/load",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req, timeout=10)
        assert exc_info.value.code == 400

    def test_load_missing_path_field(self, buckaroo_server):
        """Omitting 'path' from the payload should return 400."""
        payload = json.dumps({"session": "no_path"}).encode()
        req = Request(
            f"http://localhost:{BUCKAROO_TEST_PORT}/load",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req, timeout=10)
        assert exc_info.value.code == 400

    def test_load_empty_dataframe(self, buckaroo_server, tmp_path):
        """An empty DataFrame (0 rows) should load without crashing."""
        df = pd.DataFrame({"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="str")})
        path = tmp_path / "empty.parquet"
        df.to_parquet(str(path))
        resp = _load_session(str(path), "empty_df", BUCKAROO_TEST_PORT)
        assert resp["rows"] == 0

    def test_load_single_row(self, buckaroo_server, tmp_path):
        """A single-row DataFrame should load fine."""
        df = pd.DataFrame({"x": [42]})
        path = tmp_path / "single.parquet"
        df.to_parquet(str(path))
        resp = _load_session(str(path), "single_row", BUCKAROO_TEST_PORT)
        assert resp["rows"] == 1


# ---------------------------------------------------------------------------
# Session data integrity — confirm stale data bug
# ---------------------------------------------------------------------------
class TestSessionDataIntegrity:
    def test_session_update_reflects_new_row_count(self, buckaroo_server, tmp_path):
        """CRITICAL: When you load new data into an existing session_id,
        the row count must reflect the new data, not the old cached data.
        This is the root cause of the 'stale data' bug."""
        # Load 100 rows
        df1 = pd.DataFrame({"v": range(100)})
        p1 = tmp_path / "v1.parquet"
        df1.to_parquet(str(p1))
        r1 = _load_session(str(p1), "integrity", BUCKAROO_TEST_PORT)
        assert r1["rows"] == 100

        # Overwrite same session with 10 rows
        df2 = pd.DataFrame({"v": range(10)})
        p2 = tmp_path / "v2.parquet"
        df2.to_parquet(str(p2))
        r2 = _load_session(str(p2), "integrity", BUCKAROO_TEST_PORT)
        assert r2["rows"] == 10, (
            f"Session 'integrity' should have 10 rows after reload, "
            f"but has {r2['rows']}. Stale session data!"
        )

    def test_session_update_reflects_new_columns(self, buckaroo_server, tmp_path):
        """Column schema should update when session is reloaded."""
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        p1 = tmp_path / "cols1.parquet"
        df1.to_parquet(str(p1))
        r1 = _load_session(str(p1), "col_integrity", BUCKAROO_TEST_PORT)

        df2 = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        p2 = tmp_path / "cols2.parquet"
        df2.to_parquet(str(p2))
        r2 = _load_session(str(p2), "col_integrity", BUCKAROO_TEST_PORT)
        assert len(r2["columns"]) == 3, (
            f"Expected 3 columns after reload, got {len(r2['columns'])}"
        )


# ---------------------------------------------------------------------------
# Server crash recovery
# ---------------------------------------------------------------------------
class TestServerCrashRecovery:
    def test_server_survives_corrupt_file(self, buckaroo_server, tmp_path):
        """Loading a corrupt file should return an error, not crash the server."""
        bad = tmp_path / "corrupt.parquet"
        bad.write_bytes(b"this is not a parquet file at all")

        payload = json.dumps(
            {"session": "corrupt", "path": str(bad), "mode": "lazy"}
        ).encode()
        req = Request(
            f"http://localhost:{BUCKAROO_TEST_PORT}/load",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        # Should return an error, not crash
        try:
            urlopen(req, timeout=10)
        except HTTPError as e:
            assert e.code in (400, 500)

        # Server must still be alive
        assert _health(BUCKAROO_TEST_PORT) is not None, "Server died after corrupt file load"

    def test_server_survives_rapid_loads(self, buckaroo_server, small_parquet):
        """30 rapid sequential loads should not crash the server."""
        for i in range(30):
            _load_session(small_parquet, f"rapid_load_{i}", BUCKAROO_TEST_PORT)

        health = _health(BUCKAROO_TEST_PORT)
        assert health is not None, "Server died after rapid loads"
        assert health["status"] == "ok"


# ---------------------------------------------------------------------------
# Web handler integration — ensure_buckaroo_session
# ---------------------------------------------------------------------------
class TestWebBuckarooIntegration:
    def test_web_health(self, web_server):
        health = _health(WEB_TEST_PORT)
        assert health["status"] == "ok"

    def test_entry_page_returns_200_or_404(self, web_server):
        """Entry pages should return 200 (if entry exists) or 404, never 500."""
        # Test a known-missing entry
        try:
            resp = urlopen(f"http://localhost:{WEB_TEST_PORT}/entry/no_such_entry", timeout=10)
            # If we get here, 200 is fine
        except HTTPError as e:
            assert e.code == 404, f"Expected 404 for missing entry, got {e.code}"

    def test_runs_page_loads(self, web_server):
        resp = urlopen(f"http://localhost:{WEB_TEST_PORT}/runs", timeout=10)
        assert resp.status == 200
        body = resp.read().decode()
        assert "xorq" in body.lower() or "runs" in body.lower()


# ---------------------------------------------------------------------------
# Large dataset stress — break it
# ---------------------------------------------------------------------------
class TestLargeDatasetStress:
    def test_1m_rows_load_time(self, buckaroo_server, large_parquet):
        """1M rows should load within 30 seconds."""
        start = time.time()
        resp = _load_session(large_parquet, "stress_1m", BUCKAROO_TEST_PORT)
        elapsed = time.time() - start
        assert resp["rows"] == 1_000_000
        assert elapsed < 30, f"Loading 1M rows took {elapsed:.1f}s (expected < 30s)"

    def test_wide_dataset_load(self, buckaroo_server, wide_parquet):
        """1000 columns should load without issues."""
        resp = _load_session(wide_parquet, "stress_wide", BUCKAROO_TEST_PORT)
        assert len(resp["columns"]) == 1000

        # Verify health after loading wide data
        health = _health(BUCKAROO_TEST_PORT)
        assert health is not None

    def test_multiple_large_sessions_concurrently(self, buckaroo_server, large_parquet):
        """Loading multiple large datasets should not OOM the server."""
        for i in range(5):
            resp = _load_session(large_parquet, f"large_concurrent_{i}", BUCKAROO_TEST_PORT)
            assert resp["rows"] == 1_000_000

        health = _health(BUCKAROO_TEST_PORT)
        assert health is not None, "Server died under memory pressure from large datasets"
