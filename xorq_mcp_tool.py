"""xorq MCP tool — lets Claude Code build, run, and inspect xorq expressions
with Buckaroo visualization."""

import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.expanduser("~"), ".xorq", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "mcp_tool.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("xorq.mcp_tool")

# ---------------------------------------------------------------------------
# Server config
# ---------------------------------------------------------------------------
SERVER_PORT = int(os.environ.get("XORQ_PORT", "8455"))
SERVER_URL = f"http://localhost:{SERVER_PORT}"

log.info("MCP tool starting — server=%s", SERVER_URL)

# ---------------------------------------------------------------------------
# Results directory for parquet output
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(tempfile.gettempdir()) / "xorq_mcp_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Server process management (copied from buckaroo_mcp_tool.py)
# ---------------------------------------------------------------------------
_server_proc: subprocess.Popen | None = None
_server_monitor: subprocess.Popen | None = None


def _start_server_monitor(server_pid: int):
    """Spawn a tiny watchdog process that kills the server if we die."""
    global _server_monitor
    monitor_code = (
        "import os, sys, signal\n"
        f"server_pid = {server_pid}\n"
        "sys.stdin.buffer.read()\n"
        "try:\n"
        f"    os.kill(server_pid, signal.SIGTERM)\n"
        "except OSError:\n"
        "    pass\n"
    )
    _server_monitor = subprocess.Popen(
        [sys.executable, "-c", monitor_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log.info(
        "Started server monitor (pid=%d) watching server pid=%d",
        _server_monitor.pid,
        server_pid,
    )


def _cleanup_server():
    """Terminate the data server and monitor if we started them."""
    global _server_proc, _server_monitor
    if _server_proc is not None:
        try:
            if _server_proc.poll() is None:
                log.info("Shutting down server (pid=%d)", _server_proc.pid)
                _server_proc.terminate()
                try:
                    _server_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    log.warning("Server didn't stop after SIGTERM, sending SIGKILL")
                    _server_proc.kill()
        except OSError as exc:
            log.debug("Cleanup error (harmless): %s", exc)
        _server_proc = None
    if _server_monitor is not None:
        try:
            _server_monitor.terminate()
            _server_monitor.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            pass
        _server_monitor = None


atexit.register(_cleanup_server)


def _signal_handler(signum, frame):
    log.info("Received signal %s — cleaning up", signal.Signals(signum).name)
    _cleanup_server()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Health checks & server startup
# ---------------------------------------------------------------------------
def _health_check() -> dict | None:
    try:
        resp = urlopen(f"{SERVER_URL}/health", timeout=2)
        if resp.status == 200:
            data = json.loads(resp.read())
            log.debug("Health check OK: %s", data)
            return data
    except (URLError, OSError) as exc:
        log.debug("Health check failed: %s", exc)
    return None


def _format_startup_failure() -> str:
    server_log = os.path.join(LOG_DIR, "server.log")
    tail = "(server log not found)"
    try:
        if os.path.isfile(server_log):
            with open(server_log) as f:
                lines = f.readlines()
            tail = "".join(lines[-20:])
    except OSError:
        pass

    return (
        f"Buckaroo data server failed to start.\n\n"
        f"## Diagnostic info\n"
        f"- Python: {sys.executable} ({sys.version.split()[0]})\n"
        f"- Server URL: {SERVER_URL}\n"
        f"- Log dir: {LOG_DIR}\n\n"
        f"## Server log (last 20 lines)\n```\n{tail}\n```\n\n"
        f"## What to check\n"
        f"1. Is port {SERVER_PORT} already in use? "
        f"(`lsof -i :{SERVER_PORT}`)\n"
        f"2. Check the full server log: `cat {server_log}`\n"
        f"3. Check the MCP tool log: `cat {LOG_FILE}`\n"
        f"4. Try starting the server manually: "
        f"`{sys.executable} -m buckaroo.server --no-browser --port {SERVER_PORT}`\n"
    )


def ensure_server() -> dict:
    """Start the Buckaroo data server if it isn't already running."""
    import buckaroo

    expected_version = getattr(buckaroo, "__version__", "unknown")

    health = _health_check()
    if health:
        running_version = health.get("version", "unknown")
        if running_version == expected_version:
            log.info(
                "Server already running (v%s) — pid=%s uptime=%.0fs",
                running_version,
                health.get("pid"),
                health.get("uptime_s", 0),
            )
            return {
                "server_status": "reused",
                "server_pid": health.get("pid"),
                "server_uptime_s": health.get("uptime_s", 0),
            }
        else:
            old_pid = health.get("pid")
            log.info(
                "Version mismatch: running=%s expected=%s — killing old server (pid=%s)",
                running_version,
                expected_version,
                old_pid,
            )
            if old_pid:
                try:
                    os.kill(old_pid, signal.SIGTERM)
                    time.sleep(1)
                    if _health_check():
                        os.kill(old_pid, signal.SIGKILL)
                        time.sleep(0.5)
                except OSError as exc:
                    log.debug("Kill old server error (harmless): %s", exc)

    global _server_proc
    cmd = [sys.executable, "-m", "buckaroo.server", "--no-browser", "--port", str(SERVER_PORT)]
    log.info("Starting server: %s", " ".join(cmd))

    server_log = os.path.join(LOG_DIR, "server.log")
    server_log_fh = open(server_log, "a")
    _server_proc = subprocess.Popen(cmd, stdout=server_log_fh, stderr=server_log_fh)
    _start_server_monitor(_server_proc.pid)

    for i in range(20):
        time.sleep(0.25)
        health = _health_check()
        if health:
            log.info(
                "Server ready after %.1fs — pid=%s",
                (i + 1) * 0.25,
                health.get("pid"),
            )
            return {
                "server_status": "started",
                "server_pid": health.get("pid"),
                "server_uptime_s": health.get("uptime_s", 0),
            }

    log.error("Server failed to start within 5s — see %s", server_log)
    raise RuntimeError(_format_startup_failure())


# ---------------------------------------------------------------------------
# View implementation (POST to buckaroo server)
# ---------------------------------------------------------------------------
def _content_id(path: str) -> str:
    """Derive a content-based ID from a file path.

    For parquet files produced by xorq builds, the filename already is a
    content hash (e.g. ``413c1b285ad9.parquet``), so we use the stem directly.
    For other files we hash the absolute path.
    """
    import hashlib

    stem = Path(path).stem
    # If the stem looks like a hex hash (8+ hex chars), use it as-is
    if len(stem) >= 8 and all(c in "0123456789abcdef" for c in stem):
        return stem
    return hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:12]


def _load_in_buckaroo(path: str, session_id: str = "") -> dict:
    """Load a file into Buckaroo and return result metadata (no browser open)."""
    path = os.path.abspath(path)
    session = session_id or _content_id(path)

    server_info = ensure_server()

    payload = json.dumps(
        {"session": session, "path": path, "mode": "buckaroo"}
    ).encode()
    log.debug("POST %s/load payload=%s", SERVER_URL, payload.decode())

    req = Request(
        f"{SERVER_URL}/load",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urlopen(req, timeout=30)
    body = resp.read()
    log.debug("Response status=%d body=%s", resp.status, body[:500])

    result = json.loads(body)
    result["server_info"] = server_info
    result["session"] = session
    result["url"] = f"{SERVER_URL}/s/{session}"
    return result


def _view_impl(path: str) -> str:
    """Load a file in Buckaroo, open browser, and return a text summary."""
    path = os.path.abspath(path)
    log.info("view_data called — path=%s", path)

    try:
        result = _load_in_buckaroo(path)
    except Exception as exc:
        err_body = ""
        if hasattr(exc, "read"):
            try:
                err_body = exc.read().decode(errors="replace")
            except Exception:
                pass
        log.error(
            "view_impl failed: %s body=%s\n%s",
            exc,
            err_body,
            traceback.format_exc(),
        )
        raise

    rows = result["rows"]
    cols = result["columns"]
    col_lines = "\n".join(f"  - {c['name']} ({c['dtype']})" for c in cols)

    url = result["url"]
    server_info = result["server_info"]

    # Open the Buckaroo table in the browser
    import webbrowser
    webbrowser.open(url)
    browser_action = "opened"
    log.info("Opened browser: %s", url)
    server_pid = result.get("server_pid", server_info.get("server_pid", "?"))

    summary = (
        f"Loaded **{os.path.basename(path)}** — "
        f"{rows:,} rows, {len(cols)} columns\n\n"
        f"Columns:\n{col_lines}\n\n"
        f"Interactive view: {url}\n"
        f"Server: pid={server_pid} ({server_info['server_status']}) | "
        f"Browser: {browser_action} | Session: {result['session']}"
    )
    log.info(
        "view_data success — %d rows, %d cols, browser=%s, server=%s(%s)",
        rows,
        len(cols),
        browser_action,
        server_pid,
        server_info["server_status"],
    )
    return summary


# ---------------------------------------------------------------------------
# Parent watcher (handles uvx killing)
# ---------------------------------------------------------------------------
def _start_parent_watcher():
    import threading

    original_ppid = os.getppid()
    log.info("Parent watcher: original ppid=%d", original_ppid)

    def _watcher():
        while True:
            time.sleep(1)
            current_ppid = os.getppid()
            if current_ppid != original_ppid:
                log.info(
                    "Parent changed %d → %d — cleaning up",
                    original_ppid,
                    current_ppid,
                )
                _cleanup_server()
                os._exit(0)

    t = threading.Thread(target=_watcher, daemon=True)
    t.start()


# ===========================================================================
# MCP Server & Tools
# ===========================================================================
mcp = FastMCP(
    "xorq-mcp",
    instructions=(
        "Use xorq_run to build and execute xorq expressions from Python scripts. "
        "Use xorq_view_build to view a previously-built expression from the catalog. "
        "Use view_data to display raw data files (CSV, Parquet, etc.) interactively. "
        "Use xorq_catalog_ls, xorq_catalog_info, xorq_lineage, and xorq_diff_builds "
        "for catalog navigation, lineage inspection, and build comparison."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1: xorq_run
# ---------------------------------------------------------------------------
def _register_in_catalog(build_path, alias=None):
    """Register a build in the xorq catalog with an optional alias."""
    from xorq.catalog import (
        AddBuildRequest,
        CatalogPaths,
        do_copy_build_safely,
        do_ensure_directories,
        do_save_catalog,
        get_now_utc,
        load_catalog,
        process_catalog_update,
        validate_build,
    )

    request = AddBuildRequest(build_path=build_path, alias=alias)
    paths = CatalogPaths.create()
    timestamp = get_now_utc().isoformat()

    build_info = validate_build(request)
    do_ensure_directories(paths)
    do_copy_build_safely(build_info, paths)

    catalog = load_catalog(path=paths.config_path)
    updated_catalog, entry_id, revision_id = process_catalog_update(
        catalog, build_info, request.alias, timestamp
    )
    do_save_catalog(updated_catalog, paths.config_path)

    log.info(
        "Registered build %s as entry=%s rev=%s alias=%s",
        build_info.build_id, entry_id, revision_id, alias,
    )
    return entry_id, revision_id


@mcp.tool()
def xorq_run(script_path: str, expr_name: str = "expr", alias: str = "") -> str:
    """Build, execute, and visualize a xorq expression from a Python script.

    Imports the script, extracts the expression variable, builds the DAG,
    executes it to parquet (cache-aware), and opens the result in Buckaroo.

    Args:
        script_path: Path to a .py or .ipynb file containing a xorq expression.
        expr_name: Name of the expression variable in the script (default: "expr").
        alias: Optional catalog alias to register this build under.
    """
    from xorq.common.utils.caching_utils import get_xorq_cache_dir
    from xorq.common.utils.import_utils import import_from_path
    from xorq.ibis_yaml.compiler import build_expr, load_expr

    log.info("xorq_run called — script=%s expr_name=%s alias=%s", script_path, expr_name, alias)

    # 1. Import the script
    script_path = os.path.abspath(script_path)
    if not os.path.isfile(script_path):
        return f"Error: file not found: {script_path}"

    try:
        module = import_from_path(script_path)
    except Exception as exc:
        log.error("Failed to import script: %s", exc)
        return f"Error importing {script_path}: {exc}"

    expr = getattr(module, expr_name, None)
    if expr is None:
        available = [k for k in dir(module) if not k.startswith("_")]
        return (
            f"Error: variable '{expr_name}' not found in {script_path}.\n"
            f"Available names: {', '.join(available)}"
        )

    # 2. Build the expression DAG
    builds_dir = RESULTS_DIR / "builds"
    builds_dir.mkdir(parents=True, exist_ok=True)
    try:
        build_path = build_expr(expr, builds_dir=builds_dir)
    except Exception as exc:
        log.error("build_expr failed: %s\n%s", exc, traceback.format_exc())
        return f"Error building expression: {exc}"

    build_hash = Path(build_path).name
    log.info("Expression built — hash=%s path=%s", build_hash, build_path)

    # 3. Register in catalog
    catalog_alias = alias or None
    if not catalog_alias:
        # Derive alias from script filename (e.g., "simple_example" from "simple_example.py")
        catalog_alias = Path(script_path).stem
    try:
        entry_id, revision_id = _register_in_catalog(build_path, alias=catalog_alias)
        catalog_msg = f"Catalog: alias={catalog_alias} entry={entry_id} rev={revision_id}"
    except Exception as exc:
        log.error("Catalog registration failed: %s\n%s", exc, traceback.format_exc())
        catalog_msg = f"Catalog registration failed: {exc}"

    # 4. Load the expression back (with cache dir)
    cache_dir = get_xorq_cache_dir()
    try:
        loaded_expr = load_expr(build_path, cache_dir=cache_dir)
    except Exception as exc:
        log.error("load_expr failed: %s\n%s", exc, traceback.format_exc())
        return f"Error loading expression: {exc}"

    # 5. Execute to parquet
    output_path = RESULTS_DIR / f"{build_hash}.parquet"
    try:
        loaded_expr.to_parquet(str(output_path))
    except Exception as exc:
        log.error("to_parquet failed: %s\n%s", exc, traceback.format_exc())
        return f"Error executing expression: {exc}"

    log.info("Expression executed — output=%s", output_path)

    # 6. View in Buckaroo
    view_summary = _view_impl(str(output_path))
    return f"{view_summary}\n{catalog_msg}"


# ---------------------------------------------------------------------------
# Tool 2: xorq_view_build
# ---------------------------------------------------------------------------
@mcp.tool()
def xorq_view_build(target: str) -> str:
    """Load and visualize a previously-built xorq expression from the catalog.

    Resolves the target (path, alias, or entry@revision), loads the expression,
    executes to parquet, and opens in Buckaroo.

    Args:
        target: Build directory path, catalog alias, or entry_id[@revision].
    """
    from xorq.catalog import load_catalog, resolve_build_dir
    from xorq.common.utils.caching_utils import get_xorq_cache_dir
    from xorq.ibis_yaml.compiler import load_expr

    log.info("xorq_view_build called — target=%s", target)

    catalog = load_catalog()
    build_dir = resolve_build_dir(target, catalog)
    if build_dir is None:
        return f"Error: build target not found: {target}"
    if not build_dir.exists() or not build_dir.is_dir():
        return f"Error: build directory not found: {build_dir}"

    # Load expression
    cache_dir = get_xorq_cache_dir()
    try:
        expr = load_expr(build_dir, cache_dir=cache_dir)
    except Exception as exc:
        log.error("load_expr failed: %s\n%s", exc, traceback.format_exc())
        return f"Error loading expression from {build_dir}: {exc}"

    # Execute to parquet
    build_hash = build_dir.name
    output_path = RESULTS_DIR / f"{build_hash}.parquet"
    try:
        expr.to_parquet(str(output_path))
    except Exception as exc:
        log.error("to_parquet failed: %s\n%s", exc, traceback.format_exc())
        return f"Error executing expression: {exc}"

    log.info("Build executed — output=%s", output_path)
    return _view_impl(str(output_path))


# ---------------------------------------------------------------------------
# Tool 3: view_data
# ---------------------------------------------------------------------------
@mcp.tool()
def view_data(path: str) -> str:
    """Load a raw data file (CSV, TSV, Parquet, JSON) in Buckaroo for interactive viewing.

    Opens an interactive table UI in the browser and returns a text summary
    of the dataset (row count, column names and dtypes).
    """
    return _view_impl(path)


# ---------------------------------------------------------------------------
# Tool 4: xorq_catalog_ls
# ---------------------------------------------------------------------------
@mcp.tool()
def xorq_catalog_ls() -> str:
    """List all entries and aliases in the xorq catalog.

    Returns a formatted listing and opens an HTML page in the browser with
    links to Buckaroo detail views for each catalog entry.
    """
    from xorq.catalog import CatalogPaths, load_catalog, resolve_build_dir
    from xorq.common.utils.caching_utils import get_xorq_cache_dir
    from xorq.ibis_yaml.compiler import load_expr

    log.info("xorq_catalog_ls called")

    catalog = load_catalog()

    if not catalog.entries:
        return "Catalog is empty."

    # Build alias lookup: entry_id -> list of alias names
    alias_lookup = {}
    if catalog.aliases:
        for name, alias in catalog.aliases.items():
            alias_lookup.setdefault(alias.entry_id, []).append(name)

    # For each entry: execute to parquet, load into Buckaroo with a unique session
    cache_dir = get_xorq_cache_dir()
    ensure_server()
    entry_rows = []

    for entry in catalog.entries:
        curr_rev = entry.current_revision
        build_id = None
        created_at = None
        for rev in entry.history:
            if rev.revision_id == curr_rev and rev.build:
                build_id = rev.build.build_id
                created_at = str(rev.created_at) if rev.created_at else None
                break

        aliases = alias_lookup.get(entry.entry_id, [])
        display_name = aliases[0] if aliases else entry.entry_id[:12]
        buckaroo_url = None
        col_names = []
        row_count = 0
        error = None

        # Try to resolve, execute, and load into Buckaroo
        target = aliases[0] if aliases else entry.entry_id
        build_dir = resolve_build_dir(target, catalog)
        if build_dir and build_dir.exists():
            try:
                expr = load_expr(build_dir, cache_dir=cache_dir)
                output_path = RESULTS_DIR / f"{build_id}.parquet"
                expr.to_parquet(str(output_path))
                result = _load_in_buckaroo(str(output_path), session_id=build_id)
                buckaroo_url = result["url"]
                row_count = result["rows"]
                col_names = [c["name"] for c in result["columns"]]
            except Exception as exc:
                log.warning("Failed to load entry %s: %s", entry.entry_id, exc)
                error = str(exc)

        entry_rows.append({
            "display_name": display_name,
            "aliases": aliases,
            "entry_id": entry.entry_id,
            "revision": curr_rev,
            "build_id": build_id,
            "created_at": created_at,
            "buckaroo_url": buckaroo_url,
            "columns": col_names,
            "rows": row_count,
            "error": error,
        })

    # Generate HTML
    html = _generate_catalog_html(entry_rows)
    html_path = RESULTS_DIR / "catalog.html"
    html_path.write_text(html)

    import webbrowser
    webbrowser.open(f"file://{html_path}")
    log.info("Opened catalog page: %s", html_path)

    # Also return text summary
    lines = [f"Opened catalog page with {len(entry_rows)} entries\n"]
    for row in entry_rows:
        status = f"({row['rows']} rows, {len(row['columns'])} cols)" if row["buckaroo_url"] else f"(error: {row['error']})"
        lines.append(f"  {row['display_name']}  {row['revision']}  {status}")

    return "\n".join(lines)


def _generate_catalog_html(entries: list) -> str:
    """Generate an HTML catalog page with links to Buckaroo detail views."""
    cards = []
    for e in entries:
        alias_str = ", ".join(e["aliases"]) if e["aliases"] else "—"
        cols_str = ", ".join(e["columns"][:8])
        if len(e["columns"]) > 8:
            cols_str += f", ... (+{len(e['columns']) - 8} more)"

        if e["buckaroo_url"]:
            link = f'<a href="{e["buckaroo_url"]}" class="view-btn">View in Buckaroo</a>'
            stats = f'{e["rows"]:,} rows &middot; {len(e["columns"])} columns'
        else:
            link = f'<span class="error">Error: {e["error"]}</span>'
            stats = "—"

        cards.append(f"""
        <div class="card">
            <div class="card-header">
                <h2>{e["display_name"]}</h2>
                {link}
            </div>
            <div class="meta">
                <span class="badge">{e["revision"]}</span>
                <span class="stats">{stats}</span>
            </div>
            <div class="details">
                <div><strong>Aliases:</strong> {alias_str}</div>
                <div><strong>Build:</strong> <code>{e["build_id"]}</code></div>
                <div><strong>Columns:</strong> {cols_str}</div>
                <div><strong>Created:</strong> {e["created_at"] or "—"}</div>
            </div>
        </div>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>xorq Catalog</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    padding: 2rem;
  }}
  h1 {{
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
    color: #fff;
  }}
  .subtitle {{
    color: #888;
    margin-bottom: 2rem;
    font-size: 0.95rem;
  }}
  .card {{
    background: #1a1d27;
    border: 1px solid #2a2d37;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
  }}
  .card:hover {{
    border-color: #4a7dff;
  }}
  .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
  }}
  .card-header h2 {{
    font-size: 1.2rem;
    color: #fff;
  }}
  .view-btn {{
    background: #4a7dff;
    color: #fff;
    text-decoration: none;
    padding: 0.4rem 1rem;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    transition: background 0.2s;
  }}
  .view-btn:hover {{
    background: #3a6dee;
  }}
  .meta {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
  }}
  .badge {{
    background: #2a2d37;
    color: #8ab4f8;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-family: monospace;
  }}
  .stats {{
    color: #999;
    font-size: 0.85rem;
  }}
  .details {{
    font-size: 0.85rem;
    color: #aaa;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.35rem 1.5rem;
  }}
  .details code {{
    background: #2a2d37;
    padding: 0.1rem 0.35rem;
    border-radius: 3px;
    font-size: 0.8rem;
  }}
  .error {{
    color: #ff6b6b;
    font-size: 0.85rem;
  }}
</style>
</head>
<body>
  <h1>xorq Catalog</h1>
  <p class="subtitle">{len(entries)} entries</p>
  {"".join(cards)}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Tool 5: xorq_catalog_info
# ---------------------------------------------------------------------------
@mcp.tool()
def xorq_catalog_info(target: str) -> str:
    """Show detailed information about a catalog entry or alias.

    Displays revision history, build paths, and schema for the resolved target.

    Args:
        target: Catalog alias or entry_id[@revision].
    """
    from xorq.catalog import load_catalog, resolve_build_dir
    from xorq.ibis_yaml.compiler import load_expr

    log.info("xorq_catalog_info called — target=%s", target)

    catalog = load_catalog()

    # Resolve the target to find entry + revision info
    resolved = catalog.resolve_target(target)
    if resolved is None:
        return f"Target not found: {target}"

    entry = catalog.maybe_get_entry(resolved.entry_id)
    if entry is None:
        return f"Entry not found: {resolved.entry_id}"

    lines = [f"## Entry: {entry.entry_id}"]
    lines.append(f"Current revision: {entry.current_revision}")
    if resolved.alias:
        lines.append(f"Resolved via alias: {target}")

    lines.append(f"\n### Revision history ({len(entry.history)} revisions)")
    for rev in entry.history:
        build_info = ""
        if rev.build:
            build_info = f"  build={rev.build.build_id}"
            if rev.build.path:
                build_info += f"  path={rev.build.path}"
        lines.append(f"  {rev.revision_id}  created={rev.created_at}{build_info}")

    # Try to show schema from the current build
    build_dir = resolve_build_dir(target, catalog)
    if build_dir and build_dir.exists():
        try:
            expr = load_expr(build_dir)
            if hasattr(expr, "schema"):
                schema = expr.schema()
                lines.append("\n### Schema")
                for col_name, col_type in zip(schema.names, schema.types):
                    lines.append(f"  {col_name}: {col_type}")
        except Exception as exc:
            lines.append(f"\n(Could not load schema: {exc})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 6: xorq_lineage
# ---------------------------------------------------------------------------
@mcp.tool()
def xorq_lineage(target: str, column: str = "") -> str:
    """Show column lineage for a built xorq expression.

    Traces how each output column is derived from source data.

    Args:
        target: Build directory path, catalog alias, or entry_id[@revision].
        column: Optional specific column name. If empty, shows all columns.
    """
    from xorq.catalog import load_catalog, resolve_build_dir
    from xorq.common.utils.lineage_utils import build_column_trees
    from xorq.ibis_yaml.compiler import load_expr

    log.info("xorq_lineage called — target=%s column=%s", target, column)

    catalog = load_catalog()
    build_dir = resolve_build_dir(target, catalog)
    if build_dir is None:
        return f"Error: build target not found: {target}"
    if not build_dir.exists() or not build_dir.is_dir():
        return f"Error: build directory not found: {build_dir}"

    try:
        expr = load_expr(build_dir)
    except Exception as exc:
        return f"Error loading expression: {exc}"

    trees = build_column_trees(expr)

    if column:
        if column not in trees:
            return (
                f"Column '{column}' not found. "
                f"Available columns: {', '.join(trees.keys())}"
            )
        trees = {column: trees[column]}

    lines = []
    for col_name, tree_node in trees.items():
        lines.append(f"## Lineage: {col_name}")
        lines.append(_format_lineage_text(tree_node, indent=0))
        lines.append("")

    return "\n".join(lines)


def _format_lineage_text(node, indent: int = 0) -> str:
    """Format a GenericNode lineage tree as indented text."""
    op = node.op
    prefix = "  " * indent

    # Determine a readable label
    name = op.__class__.__name__
    if hasattr(op, "name"):
        label = f"{name}: {op.name}"
    else:
        label = name

    lines = [f"{prefix}- {label}"]
    for child in node.children:
        lines.append(_format_lineage_text(child, indent + 1))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 7: xorq_diff_builds
# ---------------------------------------------------------------------------
@mcp.tool()
def xorq_diff_builds(left: str, right: str) -> str:
    """Compare two xorq builds by diffing their expr.yaml files.

    Args:
        left: First build target (path, alias, or entry@revision).
        right: Second build target (path, alias, or entry@revision).
    """
    from xorq.catalog import load_catalog, resolve_build_dir

    log.info("xorq_diff_builds called — left=%s right=%s", left, right)

    catalog = load_catalog()
    left_dir = resolve_build_dir(left, catalog)
    right_dir = resolve_build_dir(right, catalog)

    if left_dir is None:
        return f"Error: build target not found: {left}"
    if right_dir is None:
        return f"Error: build target not found: {right}"
    if not left_dir.exists() or not left_dir.is_dir():
        return f"Error: build directory not found: {left_dir}"
    if not right_dir.exists() or not right_dir.is_dir():
        return f"Error: build directory not found: {right_dir}"

    left_expr = left_dir / "expr.yaml"
    right_expr = right_dir / "expr.yaml"

    if not left_expr.exists():
        return f"Error: expr.yaml not found in {left_dir}"
    if not right_expr.exists():
        return f"Error: expr.yaml not found in {right_dir}"

    try:
        result = subprocess.run(
            ["git", "diff", "--no-index", "--", str(left_expr), str(right_expr)],
            capture_output=True,
            text=True,
        )
        diff_output = result.stdout
    except FileNotFoundError:
        return "Error: git command not found. git is required for diff."

    if result.returncode == 0:
        return f"No differences between builds.\n  left:  {left_dir}\n  right: {right_dir}"

    return (
        f"## Diff: expr.yaml\n"
        f"left:  {left_dir}\n"
        f"right: {right_dir}\n\n"
        f"```diff\n{diff_output}\n```"
    )


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    _start_parent_watcher()
    mcp.run()


if __name__ == "__main__":
    main()
