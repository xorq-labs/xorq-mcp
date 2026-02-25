"""Shared catalog registration and resolution utilities.

Provides CatalogSnapshot (request-scoped read-once cache), resolve_target(),
metadata read/write helpers, and catalog registration functions.
"""

import logging

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
