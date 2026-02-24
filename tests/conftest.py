"""Shared fixtures for xorq_mcp tests."""

import json
import os
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import pytest

# Ports used exclusively by tests — must not clash with dev servers.
BUCKAROO_TEST_PORT = 8655
WEB_TEST_PORT = 8656

VENV_PYTHON = str(Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python3")


def _health(port: int, path: str = "/health") -> dict | None:
    try:
        resp = urlopen(f"http://localhost:{port}{path}", timeout=2)
        if resp.status == 200:
            return json.loads(resp.read())
    except Exception:
        pass
    return None


def _wait_healthy(port: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _health(port):
            return True
        time.sleep(0.25)
    return False


def _load_session(parquet_path: str, session_id: str, port: int, mode: str = "lazy") -> dict:
    """POST a dataset to Buckaroo /load and return the response."""
    payload = json.dumps(
        {"session": session_id, "path": os.path.abspath(parquet_path), "mode": mode}
    ).encode()
    req = Request(
        f"http://localhost:{port}/load",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urlopen(req, timeout=30)
    return json.loads(resp.read())


@pytest.fixture(scope="session")
def buckaroo_server():
    """Start a Buckaroo data server for the test session."""
    cmd = [
        VENV_PYTHON,
        "-m",
        "buckaroo.server",
        "--no-browser",
        "--port",
        str(BUCKAROO_TEST_PORT),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not _wait_healthy(BUCKAROO_TEST_PORT):
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        pytest.fail(
            f"Buckaroo server failed to start on port {BUCKAROO_TEST_PORT}\n"
            f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )
    yield {"port": BUCKAROO_TEST_PORT, "pid": proc.pid, "proc": proc}
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def web_server(buckaroo_server):
    """Start the xorq web server for the test session."""
    cmd = [
        VENV_PYTHON,
        "-m",
        "xorq_web",
        "--port",
        str(WEB_TEST_PORT),
        "--buckaroo-port",
        str(BUCKAROO_TEST_PORT),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not _wait_healthy(WEB_TEST_PORT):
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        pytest.fail(
            f"Web server failed to start on port {WEB_TEST_PORT}\n"
            f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )
    yield {"port": WEB_TEST_PORT, "pid": proc.pid, "proc": proc}
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def small_parquet(tmp_path):
    """Create a small parquet file (100 rows)."""
    df = pd.DataFrame(
        {
            "id": range(100),
            "name": [f"item_{i}" for i in range(100)],
            "value": [i * 1.5 for i in range(100)],
        }
    )
    path = tmp_path / "small.parquet"
    df.to_parquet(str(path))
    return str(path)


@pytest.fixture
def medium_parquet(tmp_path):
    """Create a medium parquet file (100k rows)."""
    import numpy as np

    n = 100_000
    df = pd.DataFrame(
        {
            "id": range(n),
            "category": [f"cat_{i % 50}" for i in range(n)],
            "value": np.random.default_rng(42).standard_normal(n),
            "label": [f"label_{i}" for i in range(n)],
        }
    )
    path = tmp_path / "medium.parquet"
    df.to_parquet(str(path))
    return str(path)


@pytest.fixture
def large_parquet(tmp_path):
    """Create a large parquet file (1M rows, ~50MB)."""
    import numpy as np

    n = 1_000_000
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "id": range(n),
            "category": [f"cat_{i % 200}" for i in range(n)],
            "value_a": rng.standard_normal(n),
            "value_b": rng.standard_normal(n),
            "value_c": rng.standard_normal(n),
            "label": [f"row_{i}_" + "x" * 50 for i in range(n)],
        }
    )
    path = tmp_path / "large.parquet"
    df.to_parquet(str(path))
    return str(path)


@pytest.fixture
def wide_parquet(tmp_path):
    """Create a wide parquet file (1000 columns)."""
    import numpy as np

    n = 10_000
    rng = np.random.default_rng(42)
    data = {f"col_{i:04d}": rng.standard_normal(n) for i in range(1000)}
    df = pd.DataFrame(data)
    path = tmp_path / "wide.parquet"
    df.to_parquet(str(path))
    return str(path)
