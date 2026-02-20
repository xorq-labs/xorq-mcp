"""Tornado request handlers for the xorq web server."""

import logging
import os
import tempfile
import traceback
from pathlib import Path

import tornado.web

from xorq_web.metadata import (
    ensure_buckaroo_session,
    get_all_runs,
    get_catalog_entries,
    get_entry_revisions,
    load_build_metadata,
    load_lineage_html,
)

log = logging.getLogger("xorq.web")

RESULTS_DIR = Path(tempfile.gettempdir()) / "xorq_mcp_results"


class CatalogIndexHandler(tornado.web.RequestHandler):
    def get(self):
        nav_entries = get_catalog_entries()
        self.render(
            "catalog_index.html",
            entries=nav_entries,
            nav_entries=nav_entries,
            current_entry=None,
        )


class ExpressionDetailHandler(tornado.web.RequestHandler):
    def get(self, target: str):
        from xorq.catalog import Target, load_catalog, resolve_build_dir
        from xorq.common.utils.caching_utils import get_xorq_cache_dir
        from xorq.ibis_yaml.compiler import load_expr

        buckaroo_port = self.application.settings["buckaroo_port"]
        nav_entries = get_catalog_entries()

        catalog = load_catalog()
        build_dir = resolve_build_dir(target, catalog)
        if build_dir is None or not build_dir.exists():
            self.set_status(404)
            self.write(f"Build target not found: {target}")
            return

        # Parse base name (strip @rN) for display and revision nav links
        base_name = target.split("@")[0]

        # Resolve the target to get the actual revision being viewed
        resolved = Target.from_str(target, catalog)
        current_rev_id = resolved.rev if resolved else None

        # Find the matching catalog entry for display info
        display_name = base_name
        revision = current_rev_id
        build_id = build_dir.name
        created_at = None
        for e in nav_entries:
            if e["display_name"] == base_name or e["entry_id"] == base_name:
                display_name = e["display_name"]
                build_id = e["build_id"] or build_dir.name
                break

        # Look up created_at and metadata from the specific revision
        revision_metadata = None
        if resolved:
            entry = catalog.maybe_get_entry(resolved.entry_id)
            if entry:
                rev_obj = entry.maybe_get_revision(current_rev_id)
                if rev_obj:
                    created_at = str(rev_obj.created_at) if rev_obj.created_at else None
                    if rev_obj.build:
                        build_id = rev_obj.build.build_id
                    revision_metadata = rev_obj.metadata

        # Compute revision navigation
        revisions = get_entry_revisions(target)
        prev_url = None
        next_url = None
        prev_rev_id = None
        next_rev_id = None
        if revisions:
            current_idx = next(
                (i for i, r in enumerate(revisions) if r["revision_id"] == current_rev_id),
                None,
            )
            if current_idx is not None and current_idx > 0:
                prev_rev_id = revisions[current_idx - 1]["revision_id"]
                prev_url = f"/entry/{display_name}@{prev_rev_id}"
            if current_idx is not None and current_idx < len(revisions) - 1:
                next_rev_id = revisions[current_idx + 1]["revision_id"]
                next_url = f"/entry/{display_name}@{next_rev_id}"

        # Load metadata
        metadata = load_build_metadata(build_dir)

        # Execute to parquet and load into Buckaroo
        buckaroo_session = None
        try:
            cache_dir = get_xorq_cache_dir()
            expr = load_expr(build_dir, cache_dir=cache_dir)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            output_path = RESULTS_DIR / f"{build_id}.parquet"
            expr.to_parquet(str(output_path))
            result = ensure_buckaroo_session(
                str(output_path), build_id, buckaroo_port
            )
            buckaroo_session = build_id
        except Exception as exc:
            log.error(
                "Failed to load expression for %s: %s\n%s",
                target, exc, traceback.format_exc(),
            )

        # Load lineage
        lineage = load_lineage_html(build_dir)

        self.render(
            "expression_detail.html",
            nav_entries=nav_entries,
            current_entry=display_name,
            display_name=display_name,
            revision=revision,
            build_id=build_id,
            created_at=created_at,
            metadata=metadata,
            buckaroo_port=buckaroo_port,
            buckaroo_session=buckaroo_session,
            lineage=lineage,
            revisions=revisions,
            current_rev_id=current_rev_id,
            prev_url=prev_url,
            next_url=next_url,
            prev_rev_id=prev_rev_id,
            next_rev_id=next_rev_id,
            revision_metadata=revision_metadata,
        )


class RunsHandler(tornado.web.RequestHandler):
    def get(self):
        nav_entries = get_catalog_entries()
        runs = get_all_runs()
        self.render(
            "runs.html",
            nav_entries=nav_entries,
            current_entry="__runs__",
            runs=runs,
        )


class HealthHandler(tornado.web.RequestHandler):
    def get(self):
        self.write({"status": "ok", "service": "xorq-web", "pid": os.getpid()})
