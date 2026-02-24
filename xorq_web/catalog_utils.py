"""Shared catalog registration utilities.

Extracted from xorq_mcp_tool.py so both the MCP tool and web handlers
(e.g. session promote) can register builds without circular imports.
"""

import logging

log = logging.getLogger("xorq.web.catalog_utils")


def register_in_catalog(build_path, alias=None, metadata=None):
    """Register a build in the xorq catalog with an optional alias.

    Args:
        build_path: Path to the build directory.
        alias: Optional catalog alias.
        metadata: Optional dict to store on the revision (e.g. {"prompt": "..."}).

    Returns:
        (entry_id, revision_id) tuple.
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


def update_revision_metadata(entry_id, revision_id, updates):
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
