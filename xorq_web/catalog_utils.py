"""Shared catalog registration and resolution utilities.

Provides CatalogSnapshot (request-scoped read-once cache), resolve_target(),
metadata read/write helpers, catalog registration functions, and catalog
health-check / repair utilities.
"""

import logging
import platform
import traceback as tb_mod
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger("xorq.web.catalog_utils")


class CatalogSnapshot:
    """Request-scoped read-once cache for catalog entries and aliases.

    ``catalog_entries`` and ``catalog_aliases`` are not cached on the Catalog
    object — every property access re-scans the git repo.  CatalogSnapshot
    reads both once and exposes a dict-like interface for fast repeated lookups.
    """

    def __init__(self, catalog=None):
        if catalog is None:
            from xorq.catalog import Catalog

            catalog = Catalog.from_default()
        self._catalog = catalog
        self._entries = tuple(catalog.catalog_entries)
        self._aliases = tuple(catalog.catalog_aliases)

        # entry_name -> CatalogEntry
        self._entry_map: dict = {}
        for e in self._entries:
            self._entry_map[e.name] = e

        # entry_name -> sorted list of alias strings
        self._alias_lookup: dict[str, list[str]] = {}
        # alias_string -> CatalogAlias
        self._alias_map: dict = {}
        for ca in self._aliases:
            entry_name = ca.catalog_entry.name
            self._alias_lookup.setdefault(entry_name, []).append(ca.alias)
            self._alias_map[ca.alias] = ca
        for k in self._alias_lookup:
            self._alias_lookup[k].sort()

    @property
    def catalog(self):
        return self._catalog

    @property
    def entries(self):
        return self._entries

    @property
    def aliases(self):
        return self._aliases

    def contains(self, name: str) -> bool:
        return name in self._entry_map or name in self._alias_map

    def get_catalog_entry(self, name: str):
        """Resolve name to a CatalogEntry. Checks entry names first, then aliases."""
        if name in self._entry_map:
            return self._entry_map[name]
        ca = self._alias_map.get(name)
        if ca is not None:
            return ca.catalog_entry
        return None

    def get_catalog_alias(self, alias: str):
        """Return the CatalogAlias object for the given alias string."""
        return self._alias_map.get(alias)

    def aliases_for(self, entry_name: str) -> list[str]:
        """Return sorted list of alias strings for an entry name."""
        return self._alias_lookup.get(entry_name, [])

    def display_name_for(self, entry_name: str) -> str:
        """Return the display name for an entry: first alias or truncated entry name."""
        aliases = self.aliases_for(entry_name)
        return aliases[0] if aliases else entry_name[:12]


def resolve_target(target: str, snapshot: "CatalogSnapshot | None" = None):
    """Resolve a target string to a CatalogEntry.

    Target can be: entry name, alias name, or ``alias@revision`` (the @revision
    part is stripped for resolution — callers must handle revision separately).

    Returns CatalogEntry or None.
    """
    if snapshot is None:
        snapshot = CatalogSnapshot()

    base_name = target.split("@")[0]
    return snapshot.get_catalog_entry(base_name)


def _read_entry_metadata(entry) -> dict:
    """Read .metadata.yaml from a CatalogEntry.

    Returns {} on missing or malformed files.
    """
    try:
        if not entry.metadata_path.exists():
            return {}
        text = entry.metadata_path.read_text()
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.warning("Failed to read metadata for %s: %s", entry.name, exc)
        return {}


def _write_entry_metadata(entry, updates: dict) -> None:
    """Read-merge-write .metadata.yaml on a CatalogEntry."""
    existing = _read_entry_metadata(entry)
    merged = {**existing, **updates}
    try:
        entry.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        entry.metadata_path.write_text(yaml.dump(merged, default_flow_style=False))
    except Exception as exc:
        log.warning("Failed to write metadata for %s: %s", entry.name, exc)


def register_in_catalog(build_path, alias=None, metadata=None):
    """Register a build in the xorq catalog with an optional alias.

    Args:
        build_path: Path to the build directory.
        alias: Optional catalog alias.
        metadata: Optional dict to store in .metadata.yaml (e.g. {"prompt": "..."}).

    Returns:
        (entry_name, entry_name) tuple for backwards compatibility.
    """
    from pathlib import Path

    from xorq.catalog import Catalog

    catalog = Catalog.from_default()
    aliases = (alias,) if alias else ()
    catalog.add(Path(build_path), aliases=aliases)

    # Resolve the entry we just added to write metadata
    snapshot = CatalogSnapshot(catalog)
    if alias:
        entry = snapshot.get_catalog_entry(alias)
    else:
        # Without an alias, find by build hash
        build_name = Path(build_path).name
        entry = snapshot.get_catalog_entry(build_name)

    entry_name = entry.name if entry else Path(build_path).name

    if metadata and entry:
        _write_entry_metadata(entry, metadata)

    log.info(
        "Registered build %s as entry=%s alias=%s metadata=%s",
        Path(build_path).name,
        entry_name,
        alias,
        list(metadata.keys()) if metadata else None,
    )
    return entry_name, entry_name


def update_revision_metadata(entry_id, revision_id, updates):
    """Merge additional keys into an entry's metadata and re-save.

    In the new git-backed catalog, metadata is per-entry (not per-revision).
    The revision_id parameter is accepted for API compatibility but not used.
    """
    from xorq.catalog import Catalog

    catalog = Catalog.from_default()
    snapshot = CatalogSnapshot(catalog)
    entry = snapshot.get_catalog_entry(entry_id)
    if not entry:
        # Try looking up by alias
        for ca in snapshot.aliases:
            if ca.catalog_entry.name == entry_id:
                entry = ca.catalog_entry
                break
    if not entry:
        log.warning("update_revision_metadata: entry not found: %s", entry_id)
        return

    _write_entry_metadata(entry, updates)


# -----------------------------------------------------------------------
# Catalog health check & repair
# -----------------------------------------------------------------------


@dataclass
class CatalogHealthReport:
    """Diagnostic snapshot of catalog health — safe to construct even when broken."""

    healthy: bool = False
    error_type: str = ""
    error_message: str = ""
    traceback_text: str = ""
    repo_path: str = ""
    catalog_yaml_exists: bool = False
    yaml_entry_count: int = 0
    yaml_alias_count: int = 0
    fs_entry_count: int = 0
    fs_alias_count: int = 0
    fs_metadata_count: int = 0
    is_desync: bool = False
    missing_from_yaml: list[str] = field(default_factory=list)
    extra_in_yaml: list[str] = field(default_factory=list)
    xorq_version: str = ""
    python_version: str = ""
    repair_available: bool = False
    repair_hint: str = ""
    summary: str = ""
    mitigation_steps: list[str] = field(default_factory=list)


def _find_catalog_repo_path() -> Path | None:
    """Find the default catalog git repo path, returning None on failure."""
    try:
        from xorq.catalog import Catalog

        return Catalog.by_name_base_path / "default"
    except Exception:
        fallback = Path.home() / ".local" / "share" / "xorq" / "git-catalogs" / "default"
        return fallback if fallback.exists() else None


def _scan_catalog_fs(repo_path: Path) -> dict:
    """Count entries, aliases, and metadata files on the filesystem."""
    entries_dir = repo_path / "entry"
    aliases_dir = repo_path / "alias"
    metadata_dir = repo_path / "metadata"

    entry_names = sorted(p.stem for p in entries_dir.glob("*.tgz")) if entries_dir.is_dir() else []
    alias_names = sorted(p.stem for p in aliases_dir.glob("*.tgz")) if aliases_dir.is_dir() else []
    metadata_count = len(list(metadata_dir.glob("*.metadata.yaml"))) if metadata_dir.is_dir() else 0

    return {
        "entry_names": entry_names,
        "alias_names": alias_names,
        "fs_entry_count": len(entry_names),
        "fs_alias_count": len(alias_names),
        "fs_metadata_count": metadata_count,
    }


def _read_catalog_yaml(repo_path: Path) -> dict | None:
    """Read catalog.yaml and return its contents, or None if missing/broken."""
    yaml_path = repo_path / "catalog.yaml"
    if not yaml_path.exists():
        return None
    try:
        data = yaml.safe_load(yaml_path.read_text())
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def catalog_health_check() -> CatalogHealthReport:
    """Run a non-destructive health check on the default catalog.

    Safe to call when the catalog is broken — never raises.
    """
    report = CatalogHealthReport()

    # Environment info
    report.python_version = platform.python_version()
    try:
        import xorq

        report.xorq_version = getattr(xorq, "__version__", "unknown")
    except Exception:
        report.xorq_version = "not installed"

    # Find repo
    repo_path = _find_catalog_repo_path()
    if repo_path is not None:
        report.repo_path = str(repo_path)
    else:
        report.healthy = False
        report.summary = "Could not locate the catalog repository."
        report.mitigation_steps = [
            "Run a xorq expression to initialise the catalog.",
            "Check that xorq is installed: pip show xorq",
        ]
        return report

    # Filesystem scan
    fs = _scan_catalog_fs(repo_path)
    report.fs_entry_count = fs["fs_entry_count"]
    report.fs_alias_count = fs["fs_alias_count"]
    report.fs_metadata_count = fs["fs_metadata_count"]

    # catalog.yaml
    yaml_data = _read_catalog_yaml(repo_path)
    report.catalog_yaml_exists = yaml_data is not None
    if yaml_data is not None:
        report.yaml_entry_count = len(yaml_data.get("entry", []))
        report.yaml_alias_count = len(yaml_data.get("alias", []))

    # Desync detection
    if yaml_data is not None:
        yaml_entries = set(yaml_data.get("entry", []))
        fs_entries = set(fs["entry_names"])
        report.missing_from_yaml = sorted(fs_entries - yaml_entries)
        report.extra_in_yaml = sorted(yaml_entries - fs_entries)
        report.is_desync = bool(report.missing_from_yaml or report.extra_in_yaml)
    elif report.fs_entry_count > 0:
        report.is_desync = True
        report.missing_from_yaml = fs["entry_names"]

    # Try loading the catalog
    try:
        from xorq.catalog import Catalog

        Catalog.from_default()
        report.healthy = True
        report.summary = "Catalog is healthy."
    except Exception as exc:
        report.healthy = False
        report.error_type = type(exc).__name__
        report.error_message = str(exc)
        report.traceback_text = tb_mod.format_exc()
        report.summary = f"Catalog failed to load: {type(exc).__name__}: {exc}"

    # Repair availability
    if report.is_desync and not report.healthy:
        report.repair_available = True
        report.repair_hint = (
            "Run xorq_doctor(fix=True) or: "
            'python -c "from xorq_web.catalog_utils import repair_catalog_yaml; '
            'print(repair_catalog_yaml())"'
        )

    # Mitigation steps
    if not report.healthy:
        steps = []
        if report.is_desync:
            steps.append(
                "catalog.yaml is out of sync with the filesystem "
                f"({report.yaml_entry_count} yaml entries vs "
                f"{report.fs_entry_count} on disk). "
                "Use xorq_doctor(fix=True) to rebuild it."
            )
        steps.append(f"Check the catalog repo at: {report.repo_path}")
        steps.append("File a bug at https://github.com/xorq-labs/xorq/issues/new")
        report.mitigation_steps = steps

    return report


def repair_catalog_yaml() -> str:
    """Rebuild catalog.yaml from filesystem state and commit.

    Returns a human-readable summary of what was done.
    """
    import subprocess

    repo_path = _find_catalog_repo_path()
    if repo_path is None:
        return "Error: could not locate the catalog repository."

    fs = _scan_catalog_fs(repo_path)
    entry_names = fs["entry_names"]
    alias_names = fs["alias_names"]

    yaml_path = repo_path / "catalog.yaml"
    new_data = {"entry": entry_names, "alias": alias_names}
    yaml_path.write_text(yaml.safe_dump(new_data, default_flow_style=False))

    # Git add + commit
    try:
        subprocess.run(
            ["git", "add", "catalog.yaml"],
            cwd=str(repo_path),
            check=True,
            capture_output=True,
        )
        n_entries, n_aliases = len(entry_names), len(alias_names)
        msg = f"repair: rebuild catalog.yaml ({n_entries} entries, {n_aliases} aliases)"
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(repo_path),
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        return (
            f"Wrote catalog.yaml but git commit failed: {exc.stderr.decode().strip()}\n"
            f"Path: {yaml_path}"
        )

    return (
        f"Repaired catalog.yaml: {len(entry_names)} entries, {len(alias_names)} aliases.\n"
        f"Path: {yaml_path}"
    )
