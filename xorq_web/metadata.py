"""Utility functions to load xorq build data for templates."""

import json
import logging
import os
import threading
from pathlib import Path
from urllib.request import Request, urlopen

log = logging.getLogger("xorq.web")


def load_build_metadata(build_dir: Path) -> dict:
    """Read metadata.json from a build directory.

    Returns dict with keys like current_library_version, git_state, etc.
    Returns empty dict if metadata.json is missing or unreadable.
    """
    meta_path = build_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception as exc:
        log.warning("Failed to read metadata.json in %s: %s", build_dir, exc)
        return {}


def load_build_schema(build_dir: Path) -> list[tuple[str, str]]:
    """Load expression schema as a list of (col_name, col_type) tuples.

    Returns empty list if expression cannot be loaded.
    """
    try:
        from xorq.ibis_yaml.compiler import load_expr

        expr = load_expr(build_dir)
        schema = expr.schema()
        return list(zip(schema.names, [str(t) for t in schema.types]))
    except Exception as exc:
        log.warning("Failed to load schema from %s: %s", build_dir, exc)
        return []


def load_lineage_html(build_dir: Path) -> dict[str, str]:
    """Build column lineage trees and render each as nested HTML.

    Returns dict mapping column name to HTML string of <ul>/<li> tree.
    Returns empty dict if lineage cannot be computed or times out.

    build_column_trees has no memoization and can loop for an effectively
    infinite time on complex multi-join expression graphs.  We run it in a
    daemon thread and abandon it after 5 s so the page always loads.
    """
    try:
        from xorq.common.utils.lineage_utils import build_column_trees
        from xorq.ibis_yaml.compiler import load_expr

        expr = load_expr(build_dir)

        result: dict = {}
        exc_holder: list = []

        def _run() -> None:
            try:
                result["trees"] = build_column_trees(expr)
            except Exception as e:  # noqa: BLE001
                exc_holder.append(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=5.0)

        if t.is_alive():
            log.warning(
                "build_column_trees timed out for %s — skipping lineage", build_dir
            )
            return {}

        if exc_holder:
            raise exc_holder[0]

        return {col: _render_node_html(node) for col, node in result["trees"].items()}
    except Exception as exc:
        log.warning("Failed to build lineage from %s: %s", build_dir, exc)
        return {}


def _render_node_html(node) -> str:
    """Recursively render a GenericNode as nested <ul>/<li> HTML."""
    op = node.op
    name = op.__class__.__name__
    if hasattr(op, "name"):
        label = f"<strong>{name}</strong>: {op.name}"
    else:
        label = f"<strong>{name}</strong>"

    if not node.children:
        return f"<li>{label}</li>"

    children_html = "\n".join(_render_node_html(c) for c in node.children)
    return f"<li>{label}\n<ul>{children_html}</ul></li>"


def _buckaroo_health_check(port: int) -> bool:
    """Return True if the Buckaroo server on *port* is healthy."""
    try:
        resp = urlopen(f"http://localhost:{port}/health", timeout=2)
        return resp.status == 200
    except Exception:
        return False


def _start_buckaroo_server(port: int) -> None:
    """Start a Buckaroo server on *port* and wait until it's healthy."""
    import subprocess
    import sys
    import time

    cmd = [sys.executable, "-m", "buckaroo.server", "--no-browser", "--port", str(port)]
    log.info("Starting buckaroo server: %s", " ".join(cmd))

    log_dir = os.path.join(os.path.expanduser("~"), ".xorq", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_fh = open(os.path.join(log_dir, "server.log"), "a")
    subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh)

    for _ in range(20):
        time.sleep(0.25)
        if _buckaroo_health_check(port):
            log.info("Buckaroo server ready on port %d", port)
            return

    raise RuntimeError(f"Buckaroo server failed to start on port {port} within 5s")


def ensure_buckaroo_server(port: int) -> None:
    """Ensure the Buckaroo server is running on *port*, starting it if necessary."""
    if not _buckaroo_health_check(port):
        _start_buckaroo_server(port)


def ensure_buckaroo_session(parquet_path: str, session_id: str, buckaroo_port: int) -> dict:
    """POST to Buckaroo /load to ensure a session exists for this data.

    Starts the Buckaroo server if it isn't already running.
    Returns the Buckaroo response dict with session, rows, columns, etc.
    """
    import os

    ensure_buckaroo_server(buckaroo_port)

    payload = json.dumps(
        {"session": session_id, "path": os.path.abspath(parquet_path), "mode": "buckaroo"}
    ).encode()

    req = Request(
        f"http://localhost:{buckaroo_port}/load",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urlopen(req, timeout=30)
    return json.loads(resp.read())


def get_all_runs() -> list[dict]:
    """Return every revision across all entries, sorted newest-first.

    Each dict: {display_name, entry_id, revision_id, build_id, created_at,
                prompt, execute_seconds}
    """
    from xorq.catalog import load_catalog

    catalog = load_catalog()
    if not catalog.entries:
        return []

    alias_lookup: dict[str, list[str]] = {}
    if catalog.aliases:
        for name, alias in catalog.aliases.items():
            alias_lookup.setdefault(alias.entry_id, []).append(name)

    runs = []
    for entry in catalog.entries:
        aliases = alias_lookup.get(entry.entry_id, [])
        display_name = aliases[0] if aliases else entry.entry_id[:12]

        for rev in entry.history:
            meta = rev.metadata or {}
            runs.append(
                {
                    "display_name": display_name,
                    "entry_id": entry.entry_id,
                    "revision_id": rev.revision_id,
                    "build_id": rev.build.build_id if rev.build else None,
                    "created_at": str(rev.created_at) if rev.created_at else None,
                    "prompt": meta.get("prompt"),
                    "execute_seconds": meta.get("execute_seconds"),
                }
            )

    runs.sort(key=lambda r: r["created_at"] or "", reverse=True)
    return runs


def get_entry_revisions(target: str) -> list[dict]:
    """Return all revisions for the entry that target resolves to.

    Each dict: {revision_id, build_id, created_at, is_current}
    Returns [] if target can't be resolved.
    """
    from xorq.catalog import Target, load_catalog

    catalog = load_catalog()
    resolved = Target.from_str(target, catalog)
    if resolved is None:
        return []

    entry = catalog.maybe_get_entry(resolved.entry_id)
    if entry is None:
        return []

    return [
        {
            "revision_id": rev.revision_id,
            "build_id": rev.build.build_id if rev.build else None,
            "created_at": str(rev.created_at) if rev.created_at else None,
            "is_current": rev.revision_id == entry.current_revision,
        }
        for rev in entry.history
    ]


def get_catalog_entries() -> list[dict]:
    """Load catalog and return a list of entry dicts for navigation.

    Each dict has: alias, entry_id, revision, build_id, created_at, col_count.
    """
    from xorq.catalog import load_catalog

    catalog = load_catalog()
    if not catalog.entries:
        return []

    # Build alias lookup: entry_id -> list of alias names
    alias_lookup: dict[str, list[str]] = {}
    if catalog.aliases:
        for name, alias in catalog.aliases.items():
            alias_lookup.setdefault(alias.entry_id, []).append(name)

    entries = []
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

        entries.append(
            {
                "display_name": display_name,
                "aliases": aliases,
                "entry_id": entry.entry_id,
                "revision": curr_rev,
                "build_id": build_id,
                "created_at": created_at,
            }
        )

    return entries


def get_all_entries() -> list[dict]:
    """Return catalog entries + session entries, each tagged with ``source``.

    Catalog entries get ``source: "catalog"``, session entries get ``source: "session"``.
    """
    from xorq_web.session_store import get_session_entries

    catalog = [dict(e, source="catalog") for e in get_catalog_entries()]
    session = [
        {
            "display_name": e["name"],
            "aliases": [],
            "entry_id": None,
            "revision": None,
            "build_id": e.get("build_id"),
            "created_at": e.get("created_at"),
            "source": "session",
        }
        for e in get_session_entries()
    ]
    return catalog + session


def get_all_runs_merged() -> list[dict]:
    """Return catalog runs + session entries merged, each tagged with ``source``."""
    from xorq_web.session_store import get_session_entries

    catalog_runs = [dict(r, source="catalog") for r in get_all_runs()]
    session_runs = [
        {
            "display_name": e["name"],
            "entry_id": None,
            "revision_id": None,
            "build_id": e.get("build_id"),
            "created_at": e.get("created_at"),
            "prompt": e.get("prompt"),
            "execute_seconds": e.get("execute_seconds"),
            "source": "session",
        }
        for e in get_session_entries()
    ]
    merged = catalog_runs + session_runs
    merged.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return merged
