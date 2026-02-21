"""Session-local expression store.

Each MCP process gets a JSON manifest at ``~/.xorq/sessions/{pid}.json``.
Entries track builds that haven't been promoted to the permanent catalog yet.
Stale session files (dead PIDs) are automatically cleaned up on read.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("xorq.web.session_store")

SESSIONS_DIR = Path(os.path.expanduser("~")) / ".xorq" / "sessions"


def _session_path(pid: int | None = None) -> Path:
    """Return the manifest path for the given (or current) PID."""
    pid = pid or os.getpid()
    return SESSIONS_DIR / f"{pid}.json"


def _load_manifest(path: Path) -> list[dict]:
    """Load a session manifest, returning [] if missing or corrupt."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
    except Exception as exc:
        log.warning("Failed to read session manifest %s: %s", path, exc)
    return []


def _save_manifest(entries: list[dict], path: Path) -> None:
    """Atomically write *entries* to a session manifest file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    tmp.rename(path)


def _is_pid_alive(pid: int) -> bool:
    """Check whether a PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def add_session_entry(
    name: str,
    build_path: str,
    build_id: str,
    prompt: str = "",
    execute_seconds: float | None = None,
) -> dict:
    """Append a new entry to the current process's session manifest.

    Returns the entry dict that was added.
    """
    entry = {
        "name": name,
        "build_path": str(build_path),
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "execute_seconds": execute_seconds,
    }
    path = _session_path()
    entries = _load_manifest(path)
    # Replace existing entry with the same name (re-run overwrites)
    entries = [e for e in entries if e.get("name") != name]
    entries.append(entry)
    _save_manifest(entries, path)
    log.info("Added session entry %s (build=%s)", name, build_id)
    return entry


def get_session_entries() -> list[dict]:
    """Load entries from all *live* session manifests.

    Stale manifests (dead PIDs) are automatically removed.
    Each returned dict gets an extra ``session_pid`` key.
    """
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    all_entries: list[dict] = []
    for manifest in SESSIONS_DIR.glob("*.json"):
        try:
            pid = int(manifest.stem)
        except ValueError:
            continue
        if not _is_pid_alive(pid):
            log.info("Cleaning stale session manifest for pid %d", pid)
            manifest.unlink(missing_ok=True)
            continue
        for entry in _load_manifest(manifest):
            entry["session_pid"] = pid
            all_entries.append(entry)
    all_entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
    return all_entries


def remove_session_entry(name: str) -> bool:
    """Remove an entry by name from any live session manifest.

    Returns True if found and removed.
    """
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for manifest in SESSIONS_DIR.glob("*.json"):
        try:
            pid = int(manifest.stem)
        except ValueError:
            continue
        if not _is_pid_alive(pid):
            manifest.unlink(missing_ok=True)
            continue
        entries = _load_manifest(manifest)
        filtered = [e for e in entries if e.get("name") != name]
        if len(filtered) < len(entries):
            _save_manifest(filtered, manifest)
            log.info("Removed session entry %s from pid %d manifest", name, pid)
            return True
    return False


def get_session_entry(name: str) -> dict | None:
    """Look up a single session entry by name across all live sessions."""
    for entry in get_session_entries():
        if entry.get("name") == name:
            return entry
    return None


def update_session_entry_metadata(name: str, updates: dict) -> bool:
    """Merge *updates* into the named entry's fields.

    Returns True if the entry was found and updated.
    """
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for manifest in SESSIONS_DIR.glob("*.json"):
        try:
            pid = int(manifest.stem)
        except ValueError:
            continue
        if not _is_pid_alive(pid):
            manifest.unlink(missing_ok=True)
            continue
        entries = _load_manifest(manifest)
        changed = False
        for entry in entries:
            if entry.get("name") == name:
                entry.update(updates)
                changed = True
                break
        if changed:
            _save_manifest(entries, manifest)
            log.info("Updated session entry %s metadata: %s", name, list(updates.keys()))
            return True
    return False


def cleanup_session(pid: int | None = None) -> None:
    """Remove the session manifest for the given (or current) PID."""
    path = _session_path(pid)
    if path.exists():
        path.unlink(missing_ok=True)
        log.info("Cleaned up session manifest %s", path)
