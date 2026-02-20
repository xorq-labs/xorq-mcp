"""Tornado request handlers for the xorq web server."""

import logging
import os
import tempfile
import traceback
from pathlib import Path

import tornado.web

from xorq_web.metadata import (
    ensure_buckaroo_session,
    get_catalog_entries,
    load_build_metadata,
    load_build_schema,
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
        from xorq.catalog import load_catalog, resolve_build_dir
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

        # Find the matching catalog entry for display info
        display_name = target
        revision = None
        build_id = build_dir.name
        created_at = None
        for e in nav_entries:
            if e["display_name"] == target or e["entry_id"] == target:
                display_name = e["display_name"]
                revision = e["revision"]
                build_id = e["build_id"] or build_dir.name
                created_at = e["created_at"]
                break

        # Load metadata
        metadata = load_build_metadata(build_dir)
        schema = load_build_schema(build_dir)

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
            schema=schema,
            buckaroo_port=buckaroo_port,
            buckaroo_session=buckaroo_session,
            lineage=lineage,
        )


class HealthHandler(tornado.web.RequestHandler):
    def get(self):
        self.write({"status": "ok", "service": "xorq-web", "pid": os.getpid()})
