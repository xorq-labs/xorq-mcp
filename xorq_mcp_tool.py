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
WEB_PORT = int(os.environ.get("XORQ_WEB_PORT", str(SERVER_PORT + 1)))
WEB_URL = f"http://localhost:{WEB_PORT}"

log.info("MCP tool starting — server=%s web=%s", SERVER_URL, WEB_URL)

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
_web_server_proc: subprocess.Popen | None = None
_web_server_monitor: subprocess.Popen | None = None


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
    """Terminate the data server, web server, and their monitors."""
    global _server_proc, _server_monitor, _web_server_proc, _web_server_monitor
    for label, proc_attr, monitor_attr in [
        ("buckaroo", "_server_proc", "_server_monitor"),
        ("web", "_web_server_proc", "_web_server_monitor"),
    ]:
        proc = globals()[proc_attr]
        monitor = globals()[monitor_attr]
        if proc is not None:
            try:
                if proc.poll() is None:
                    log.info("Shutting down %s server (pid=%d)", label, proc.pid)
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        log.warning("%s server didn't stop after SIGTERM, sending SIGKILL", label)
                        proc.kill()
            except OSError as exc:
                log.debug("Cleanup error (harmless): %s", exc)
            globals()[proc_attr] = None
        if monitor is not None:
            try:
                monitor.terminate()
                monitor.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
            globals()[monitor_attr] = None
    _server_proc = None
    _server_monitor = None
    _web_server_proc = None
    _web_server_monitor = None


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


def _web_health_check() -> dict | None:
    try:
        resp = urlopen(f"{WEB_URL}/health", timeout=2)
        if resp.status == 200:
            data = json.loads(resp.read())
            log.debug("Web health check OK: %s", data)
            return data
    except (URLError, OSError) as exc:
        log.debug("Web health check failed: %s", exc)
    return None


def ensure_web_server() -> dict:
    """Start the xorq web server if it isn't already running."""
    health = _web_health_check()
    if health:
        log.info("Web server already running — pid=%s", health.get("pid"))
        return {"web_status": "reused", "web_pid": health.get("pid")}

    global _web_server_proc
    cmd = [
        sys.executable,
        "-m",
        "xorq_web",
        "--port",
        str(WEB_PORT),
        "--buckaroo-port",
        str(SERVER_PORT),
    ]
    log.info("Starting web server: %s", " ".join(cmd))

    web_log = os.path.join(LOG_DIR, "web_server.log")
    web_log_fh = open(web_log, "a")
    _web_server_proc = subprocess.Popen(cmd, stdout=web_log_fh, stderr=web_log_fh)
    _start_server_monitor(_web_server_proc.pid)

    for i in range(20):
        time.sleep(0.25)
        health = _web_health_check()
        if health:
            log.info(
                "Web server ready after %.1fs — pid=%s",
                (i + 1) * 0.25,
                health.get("pid"),
            )
            return {"web_status": "started", "web_pid": health.get("pid")}

    log.error("Web server failed to start within 5s — see %s", web_log)
    raise RuntimeError(
        f"xorq web server failed to start.\nCheck log: {web_log}\nTry manually: {' '.join(cmd)}"
    )


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

    payload = json.dumps({"session": session, "path": path, "mode": "buckaroo"}).encode()
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
def _register_in_catalog(build_path, alias=None, metadata=None):
    """Register a build in the xorq catalog with an optional alias.

    Args:
        build_path: Path to the build directory.
        alias: Optional catalog alias.
        metadata: Optional dict to store on the revision (e.g. {"prompt": "..."}).
    """
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

    # Workaround: upstream make_revision() doesn't accept metadata yet
    # (see https://github.com/xorq-labs/xorq/issues/1598).
    # We evolve the revision after the fact and re-save.
    if metadata:
        entry = updated_catalog.maybe_get_entry(entry_id)
        if entry:
            rev = entry.maybe_get_revision(revision_id)
            if rev:
                updated_rev = rev.evolve(metadata=metadata)
                updated_history = tuple(
                    updated_rev if r.revision_id == revision_id else r for r in entry.history
                )
                updated_entry = entry.evolve(history=updated_history)
                updated_entries = tuple(
                    updated_entry if e.entry_id == entry_id else e for e in updated_catalog.entries
                )
                updated_catalog = updated_catalog.evolve(entries=updated_entries)

    do_save_catalog(updated_catalog, paths.config_path)

    log.info(
        "Registered build %s as entry=%s rev=%s alias=%s metadata=%s",
        build_info.build_id,
        entry_id,
        revision_id,
        alias,
        list(metadata.keys()) if metadata else None,
    )
    return entry_id, revision_id


def _update_revision_metadata(entry_id, revision_id, updates):
    """Merge additional keys into a revision's metadata dict and re-save the catalog."""
    from xorq.catalog import CatalogPaths, do_save_catalog, load_catalog

    paths = CatalogPaths.create()
    catalog = load_catalog(path=paths.config_path)
    entry = catalog.maybe_get_entry(entry_id)
    if not entry:
        return
    rev = entry.maybe_get_revision(revision_id)
    if not rev:
        return

    merged = dict(rev.metadata or {}, **updates)
    updated_rev = rev.evolve(metadata=merged)
    updated_history = tuple(
        updated_rev if r.revision_id == revision_id else r for r in entry.history
    )
    updated_entry = entry.evolve(history=updated_history)
    updated_entries = tuple(updated_entry if e.entry_id == entry_id else e for e in catalog.entries)
    updated_catalog = catalog.evolve(entries=updated_entries)
    do_save_catalog(updated_catalog, paths.config_path)


@mcp.tool()
def xorq_run(script_path: str, expr_name: str = "expr", alias: str = "", prompt: str = "") -> str:
    """Build, execute, and visualize a xorq expression from a Python script.

    Imports the script, extracts the expression variable, builds the DAG,
    executes it to parquet (cache-aware), and opens the result in Buckaroo.

    Args:
        script_path: Path to a .py or .ipynb file containing a xorq expression.
        expr_name: Name of the expression variable in the script (default: "expr").
        alias: Optional catalog alias to register this build under.
        prompt: Optional prompt/description that produced this expression
            (stored as revision metadata).
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
    revision_metadata = {"prompt": prompt} if prompt else None
    entry_id = revision_id = None
    try:
        entry_id, revision_id = _register_in_catalog(
            build_path, alias=catalog_alias, metadata=revision_metadata
        )
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

    # 5. Execute to parquet (timed)
    output_path = RESULTS_DIR / f"{build_hash}.parquet"
    t0 = time.monotonic()
    try:
        loaded_expr.to_parquet(str(output_path))
    except Exception as exc:
        log.error("to_parquet failed: %s\n%s", exc, traceback.format_exc())
        return f"Error executing expression: {exc}"
    execute_seconds = round(time.monotonic() - t0, 3)

    log.info("Expression executed in %.3fs — output=%s", execute_seconds, output_path)

    # Store execute time on the revision metadata
    if entry_id and revision_id:
        _update_revision_metadata(entry_id, revision_id, {"execute_seconds": execute_seconds})

    # 6. Open in xorq web UI
    import webbrowser

    ensure_server()
    ensure_web_server()
    web_url = f"{WEB_URL}/entry/{catalog_alias}"
    webbrowser.open(web_url)
    log.info("Opened web UI: %s", web_url)

    # Return a text summary
    try:
        result = _load_in_buckaroo(str(output_path))
        rows = result["rows"]
        cols = result["columns"]
        col_lines = "\n".join(f"  - {c['name']} ({c['dtype']})" for c in cols)
        summary = (
            f"Loaded **{catalog_alias}** — "
            f"{rows:,} rows, {len(cols)} columns\n\n"
            f"Columns:\n{col_lines}\n\n"
            f"Web UI: {web_url}\n"
        )
    except Exception:
        summary = f"Expression executed — output at {output_path}\nWeb UI: {web_url}\n"

    return f"{summary}\n{catalog_msg}"


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

    # Open in xorq web UI
    import webbrowser

    ensure_server()
    ensure_web_server()
    web_url = f"{WEB_URL}/entry/{target}"
    webbrowser.open(web_url)
    log.info("Opened web UI: %s", web_url)

    # Return text summary
    try:
        result = _load_in_buckaroo(str(output_path))
        rows = result["rows"]
        cols = result["columns"]
        col_lines = "\n".join(f"  - {c['name']} ({c['dtype']})" for c in cols)
        summary = (
            f"Loaded **{target}** — "
            f"{rows:,} rows, {len(cols)} columns\n\n"
            f"Columns:\n{col_lines}\n\n"
            f"Web UI: {web_url}"
        )
    except Exception:
        summary = f"Expression executed — output at {output_path}\nWeb UI: {web_url}"

    return summary


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
    from xorq.catalog import load_catalog

    log.info("xorq_catalog_ls called")

    catalog = load_catalog()

    if not catalog.entries:
        return "Catalog is empty."

    # Build alias lookup: entry_id -> list of alias names
    alias_lookup: dict[str, list[str]] = {}
    if catalog.aliases:
        for name, alias in catalog.aliases.items():
            alias_lookup.setdefault(alias.entry_id, []).append(name)

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
        entry_rows.append(
            {
                "display_name": display_name,
                "aliases": aliases,
                "entry_id": entry.entry_id,
                "revision": curr_rev,
                "build_id": build_id,
                "created_at": created_at,
            }
        )

    # Open the xorq web catalog page
    import webbrowser

    ensure_server()
    ensure_web_server()
    web_url = f"{WEB_URL}/"
    webbrowser.open(web_url)
    log.info("Opened catalog page: %s", web_url)

    # Return text summary
    lines = [f"Opened catalog page with {len(entry_rows)} entries\n"]
    for row in entry_rows:
        lines.append(f"  {row['display_name']}  {row['revision']}  build={row['build_id']}")

    return "\n".join(lines)


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
            return f"Column '{column}' not found. Available columns: {', '.join(trees.keys())}"
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
        f"## Diff: expr.yaml\nleft:  {left_dir}\nright: {right_dir}\n\n```diff\n{diff_output}\n```"
    )


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    _start_parent_watcher()
    mcp.run()


if __name__ == "__main__":
    main()
