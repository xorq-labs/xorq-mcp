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
# Browser open rate limiting
# ---------------------------------------------------------------------------
_last_browser_open: float = 0.0
_BROWSER_OPEN_MIN_INTERVAL = 10  # seconds between tab opens


def _open_browser(url: str) -> None:
    """Open *url* in the browser, but no more than once every
    ``_BROWSER_OPEN_MIN_INTERVAL`` seconds to avoid spamming tabs during
    batch runs."""
    global _last_browser_open
    import webbrowser

    now = time.monotonic()
    if now - _last_browser_open >= _BROWSER_OPEN_MIN_INTERVAL:
        webbrowser.open(url)
        _last_browser_open = now
        log.info("Opened browser: %s", url)
    else:
        log.info("Skipped browser open (rate limited): %s", url)


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
    """Terminate the data server, web server, and their monitors, and clean up session file."""
    global _server_proc, _server_monitor, _web_server_proc, _web_server_monitor

    # Clean up session manifest for this process
    try:
        from xorq_web.session_store import cleanup_session

        cleanup_session()
    except Exception:
        pass

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
    _open_browser(url)
    browser_action = "opened"
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
        "for catalog navigation, lineage inspection, and build comparison. "
        "Use xorq_doctor to diagnose catalog issues; pass fix=True to auto-repair."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1: xorq_run
# ---------------------------------------------------------------------------
def _register_in_catalog(build_path, alias=None, metadata=None):
    """Register a build in the xorq catalog with an optional alias.

    Delegates to shared :func:`xorq_web.catalog_utils.register_in_catalog`.
    """
    from xorq_web.catalog_utils import register_in_catalog

    return register_in_catalog(build_path, alias=alias, metadata=metadata)


def _update_revision_metadata(entry_id, revision_id, updates):
    """Merge additional keys into a revision's metadata dict and re-save the catalog.

    Delegates to shared :func:`xorq_web.catalog_utils.update_revision_metadata`.
    """
    from xorq_web.catalog_utils import update_revision_metadata

    update_revision_metadata(entry_id, revision_id, updates)


@mcp.tool()
def xorq_run(
    script_path: str,
    expr_name: str = "expr",
    alias: str = "",
    prompt: str = "",
    catalog: bool = True,
) -> str:
    """Build, execute, and visualize a xorq expression from a Python script.

    Imports the script, extracts the expression variable, builds the DAG,
    executes it to parquet (cache-aware), and opens the result in Buckaroo.

    Args:
        script_path: Path to a .py or .ipynb file containing a xorq expression.
        expr_name: Name of the expression variable in the script (default: "expr").
        alias: Optional catalog alias to register this build under.
        prompt: Optional prompt/description that produced this expression
            (stored as revision metadata).
        catalog: If True (default), register in the permanent catalog.
            If False, keep as a session-local draft visible in the web UI
            under /session/{name} that can be promoted or discarded.
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

    # 3. Determine display name
    catalog_alias = alias or None
    if not catalog_alias:
        # Derive alias from script filename (e.g., "simple_example" from "simple_example.py")
        catalog_alias = Path(script_path).stem

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

    # 6. Register in catalog or session store
    if catalog:
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

        # Store execute time on the revision metadata
        if entry_id and revision_id:
            _update_revision_metadata(entry_id, revision_id, {"execute_seconds": execute_seconds})

        web_path = f"/entry/{catalog_alias}"
    else:
        from xorq_web.session_store import add_session_entry

        try:
            add_session_entry(
                name=catalog_alias,
                build_path=str(build_path),
                build_id=build_hash,
                prompt=prompt,
                execute_seconds=execute_seconds,
            )
            catalog_msg = f"Session: name={catalog_alias} (not in permanent catalog)"
        except Exception as exc:
            log.error("Session registration failed: %s\n%s", exc, traceback.format_exc())
            catalog_msg = f"Session registration failed: {exc}"

        web_path = f"/session/{catalog_alias}"

    # 7. Open in xorq web UI
    ensure_server()
    ensure_web_server()
    web_url = f"{WEB_URL}{web_path}"
    _open_browser(web_url)
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
    from xorq.catalog import load_expr_from_tgz

    from xorq_web.catalog_utils import CatalogSnapshot, resolve_target

    log.info("xorq_view_build called — target=%s", target)

    snapshot = CatalogSnapshot()
    entry = resolve_target(target, snapshot)
    if entry is None:
        return f"Error: build target not found: {target}"
    if not entry.exists():
        return f"Error: catalog entry not found: {entry.name}"

    # Load expression from tgz
    try:
        expr = load_expr_from_tgz(entry.catalog_path)
    except Exception as exc:
        log.error("load_expr_from_tgz failed: %s\n%s", exc, traceback.format_exc())
        return f"Error loading expression for {target}: {exc}"

    # Execute to parquet
    output_path = RESULTS_DIR / f"{entry.name}.parquet"
    try:
        expr.to_parquet(str(output_path))
    except Exception as exc:
        log.error("to_parquet failed: %s\n%s", exc, traceback.format_exc())
        return f"Error executing expression: {exc}"

    log.info("Build executed — output=%s", output_path)

    # Open in xorq web UI
    ensure_server()
    ensure_web_server()
    display_name = snapshot.display_name_for(entry.name)
    web_url = f"{WEB_URL}/entry/{display_name}"
    _open_browser(web_url)
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
    from xorq_web.catalog_utils import CatalogSnapshot

    log.info("xorq_catalog_ls called")

    snapshot = CatalogSnapshot()

    if not snapshot.entries:
        return "Catalog is empty."

    entry_rows = []
    for entry in snapshot.entries:
        aliases = snapshot.aliases_for(entry.name)
        display_name = snapshot.display_name_for(entry.name)
        entry_rows.append(
            {
                "display_name": display_name,
                "aliases": aliases,
                "entry_id": entry.name,
                "build_id": entry.name,
            }
        )

    # Open the xorq web catalog page
    ensure_server()
    ensure_web_server()
    web_url = f"{WEB_URL}/"
    _open_browser(web_url)
    log.info("Opened catalog page: %s", web_url)

    # Return text summary
    lines = [f"Opened catalog page with {len(entry_rows)} entries\n"]
    for row in entry_rows:
        lines.append(f"  {row['display_name']}  build={row['build_id']}")

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
    from xorq.catalog import load_expr_from_tgz

    from xorq_web.catalog_utils import (
        CatalogSnapshot,
        _read_entry_metadata,
        resolve_target,
    )

    log.info("xorq_catalog_info called — target=%s", target)

    snapshot = CatalogSnapshot()
    entry = resolve_target(target, snapshot)
    if entry is None:
        return f"Target not found: {target}"

    aliases = snapshot.aliases_for(entry.name)
    display_name = snapshot.display_name_for(entry.name)

    lines = [f"## Entry: {entry.name}"]
    if aliases:
        lines.append(f"Aliases: {', '.join(aliases)}")
    if display_name != entry.name:
        lines.append(f"Resolved via alias: {target}")

    # Revision history from alias
    alias_objs = entry.aliases
    if alias_objs:
        try:
            revisions = alias_objs[0].list_revisions()
            lines.append(f"\n### Revision history ({len(revisions)} revisions)")
            for rev_entry, commit in revisions:
                lines.append(f"  {commit.hexsha[:12]}  created={commit.authored_datetime}")
        except Exception as exc:
            lines.append(f"\n(Could not load revision history: {exc})")

    # Metadata
    meta = _read_entry_metadata(entry)
    if meta:
        lines.append("\n### Metadata")
        for k, v in meta.items():
            lines.append(f"  {k}: {v}")

    # Try to show schema
    try:
        expr = load_expr_from_tgz(entry.catalog_path)
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
    from xorq.catalog import load_expr_from_tgz
    from xorq.common.utils.lineage_utils import build_column_trees

    from xorq_web.catalog_utils import CatalogSnapshot, resolve_target

    log.info("xorq_lineage called — target=%s column=%s", target, column)

    snapshot = CatalogSnapshot()
    entry = resolve_target(target, snapshot)
    if entry is None:
        return f"Error: build target not found: {target}"
    if not entry.exists():
        return f"Error: catalog entry not found: {entry.name}"

    try:
        expr = load_expr_from_tgz(entry.catalog_path)
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
    from xorq.catalog.tar_utils import extract_build_tgz_context

    from xorq_web.catalog_utils import CatalogSnapshot, resolve_target

    log.info("xorq_diff_builds called — left=%s right=%s", left, right)

    snapshot = CatalogSnapshot()
    left_entry = resolve_target(left, snapshot)
    right_entry = resolve_target(right, snapshot)

    if left_entry is None:
        return f"Error: build target not found: {left}"
    if right_entry is None:
        return f"Error: build target not found: {right}"
    if not left_entry.exists():
        return f"Error: catalog entry not found: {left_entry.name}"
    if not right_entry.exists():
        return f"Error: catalog entry not found: {right_entry.name}"

    # Extract both tgz files — nested context managers
    with extract_build_tgz_context(left_entry.catalog_path) as left_dir:
        with extract_build_tgz_context(right_entry.catalog_path) as right_dir:
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
                return (
                    f"No differences between builds.\n"
                    f"  left:  {left_entry.name}\n  right: {right_entry.name}"
                )

            return (
                f"## Diff: expr.yaml\n"
                f"left:  {left_entry.name}\nright: {right_entry.name}\n\n"
                f"```diff\n{diff_output}\n```"
            )


# ---------------------------------------------------------------------------
# Tool 8: xorq_doctor
# ---------------------------------------------------------------------------
@mcp.tool()
def xorq_doctor(fix: bool = False) -> str:
    """Diagnose catalog health, server reachability, and environment info.

    Returns a markdown report. Pass fix=True to attempt automatic repair
    when a catalog desync is detected.

    Args:
        fix: If True and a desync is detected, rebuild catalog.yaml automatically.
    """
    from xorq_web.catalog_utils import catalog_health_check, repair_catalog_yaml

    log.info("xorq_doctor called — fix=%s", fix)

    lines = ["# xorq doctor\n"]

    # --- Catalog ---
    report = catalog_health_check()
    status = "OK" if report.healthy else "UNHEALTHY"
    lines.append(f"## Catalog: {status}\n")
    lines.append(f"- Path: `{report.repo_path or 'not found'}`")
    lines.append(f"- catalog.yaml: {'exists' if report.catalog_yaml_exists else 'missing'}")
    lines.append(f"- YAML entries/aliases: {report.yaml_entry_count} / {report.yaml_alias_count}")
    lines.append(f"- Filesystem entries/aliases: {report.fs_entry_count} / {report.fs_alias_count}")
    if report.is_desync:
        lines.append("\n**Desync detected.**")
        if report.missing_from_yaml:
            lines.append(
                f"- Missing from YAML ({len(report.missing_from_yaml)}): "
                f"{', '.join(report.missing_from_yaml[:10])}"
                f"{'...' if len(report.missing_from_yaml) > 10 else ''}"
            )
        if report.extra_in_yaml:
            lines.append(
                f"- Extra in YAML ({len(report.extra_in_yaml)}): "
                f"{', '.join(report.extra_in_yaml[:10])}"
                f"{'...' if len(report.extra_in_yaml) > 10 else ''}"
            )
    if not report.healthy:
        lines.append(f"\nError: {report.error_type}: {report.error_message}")

    # --- Repair ---
    if fix and report.is_desync:
        lines.append("\n## Repair\n")
        try:
            result = repair_catalog_yaml()
            lines.append(result)
            # Re-check after repair
            post = catalog_health_check()
            lines.append(f"\nPost-repair status: {'OK' if post.healthy else 'still unhealthy'}")
        except Exception as exc:
            lines.append(f"Repair failed: {exc}")
    elif fix and not report.is_desync:
        lines.append("\n## Repair\n")
        lines.append("No desync detected — nothing to repair.")

    # --- Servers ---
    lines.append("\n## Servers\n")

    buckaroo_health = _health_check()
    if buckaroo_health:
        lines.append(
            f"- Buckaroo ({SERVER_URL}): OK "
            f"(pid={buckaroo_health.get('pid')}, "
            f"uptime={buckaroo_health.get('uptime_s', 0):.0f}s)"
        )
    else:
        lines.append(f"- Buckaroo ({SERVER_URL}): not reachable")

    web_health = _web_health_check()
    if web_health:
        lines.append(f"- Web ({WEB_URL}): OK (pid={web_health.get('pid')})")
    else:
        lines.append(f"- Web ({WEB_URL}): not reachable")

    # --- Environment ---
    lines.append("\n## Environment\n")
    lines.append(f"- Python: {report.python_version}")
    lines.append(f"- xorq: {report.xorq_version}")
    try:
        import xorq_web

        mcp_version = getattr(xorq_web, "__version__", "dev")
    except Exception:
        mcp_version = "unknown"
    lines.append(f"- xorq-mcp: {mcp_version}")
    lines.append(f"- Log dir: `{LOG_DIR}`")

    return "\n".join(lines)


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    _start_parent_watcher()
    mcp.run()


if __name__ == "__main__":
    main()
