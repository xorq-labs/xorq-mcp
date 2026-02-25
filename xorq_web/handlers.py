"""Tornado request handlers for the xorq web server."""

import asyncio
import functools
import logging
import os
import tempfile
import traceback
from pathlib import Path

import tornado.web

from xorq_web.catalog_utils import (
    CatalogSnapshot,
    _read_entry_metadata,
    resolve_target,
)
from xorq_web.metadata import (
    ensure_buckaroo_session,
    get_all_entries,
    get_all_runs_merged,
    get_catalog_entries,
    get_entry_revisions,
    load_build_metadata,
    load_lineage_html,
)
from xorq_web.session_store import get_session_entries

log = logging.getLogger("xorq.web")

RESULTS_DIR = Path(tempfile.gettempdir()) / "xorq_mcp_results"


def _load_execute_and_register(
    build_dir: Path,
    build_id: str,
    output_path: Path,
    buckaroo_port: int,
) -> None:
    """Load, execute, and register an expression with Buckaroo in one thread.

    Running load_expr → to_parquet → ensure_buckaroo_session as a single
    unit in the executor guarantees that the expression object never crosses
    a thread boundary between creation and materialisation (addresses the
    cross-thread safety concern raised in PR #4 review).
    """
    from xorq.common.utils.caching_utils import get_xorq_cache_dir
    from xorq.ibis_yaml.compiler import load_expr

    cache_dir = get_xorq_cache_dir()
    expr = load_expr(build_dir, cache_dir=cache_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expr.to_parquet(str(output_path))
    ensure_buckaroo_session(str(output_path), build_id, buckaroo_port)


class BaseHandler(tornado.web.RequestHandler):
    """Injects ``session_nav_entries`` into every template render."""

    def get_template_namespace(self):
        ns = super().get_template_namespace()
        ns["session_nav_entries"] = get_session_entries()
        return ns


class CatalogIndexHandler(BaseHandler):
    def get(self):
        snapshot = CatalogSnapshot()
        all_entries = get_all_entries(snapshot)
        catalog_entries = [e for e in all_entries if e.get("source") == "catalog"]
        session_entries = [e for e in all_entries if e.get("source") == "session"]
        nav_entries = get_catalog_entries(snapshot)
        self.render(
            "catalog_index.html",
            entries=catalog_entries,
            session_entries=session_entries,
            nav_entries=nav_entries,
            current_entry="__catalog__",
        )


class ExpressionDetailHandler(BaseHandler):
    async def get(self, target: str):
        from xorq.catalog.tar_utils import extract_build_tgz_context

        loop = asyncio.get_running_loop()
        buckaroo_port = self.application.settings["buckaroo_port"]

        snapshot = CatalogSnapshot()
        nav_entries = get_catalog_entries(snapshot)

        entry = resolve_target(target, snapshot)
        if entry is None or not entry.exists():
            self.set_status(404)
            self.write(f"Build target not found: {target}")
            return

        display_name = snapshot.display_name_for(entry.name)
        build_id = entry.name
        revision_metadata = _read_entry_metadata(entry)

        # Get revision info from alias history
        current_rev_id = None
        created_at = None
        aliases = entry.aliases
        if aliases:
            try:
                revisions_list = aliases[0].list_revisions()
                if revisions_list:
                    _, latest_commit = revisions_list[-1]
                    current_rev_id = latest_commit.hexsha[:12]
                    created_at = str(latest_commit.authored_datetime)
            except Exception:
                pass

        # Compute revision navigation
        revisions = get_entry_revisions(target, snapshot)
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

        # Extract build dir from tgz — all build-dir work runs inside this context
        with extract_build_tgz_context(entry.catalog_path) as build_dir:
            # Load metadata
            metadata = load_build_metadata(build_dir)

            # Execute to parquet and load into Buckaroo
            buckaroo_session = None
            buckaroo_error = None
            output_path = RESULTS_DIR / f"{build_id}.parquet"
            try:
                await loop.run_in_executor(
                    None,
                    functools.partial(
                        _load_execute_and_register, build_dir, build_id, output_path, buckaroo_port
                    ),
                )
                buckaroo_session = build_id
            except Exception as exc:
                buckaroo_error = str(exc)
                log.error(
                    "Failed to load expression for %s: %s\n%s",
                    target,
                    exc,
                    traceback.format_exc(),
                )

            # Load lineage in executor
            lineage = await loop.run_in_executor(
                None, functools.partial(load_lineage_html, build_dir)
            )

        self.render(
            "expression_detail.html",
            nav_entries=nav_entries,
            current_entry=display_name,
            display_name=display_name,
            revision=current_rev_id,
            build_id=build_id,
            created_at=created_at,
            metadata=metadata,
            buckaroo_port=buckaroo_port,
            buckaroo_session=buckaroo_session,
            buckaroo_error=buckaroo_error,
            lineage=lineage,
            revisions=revisions,
            current_rev_id=current_rev_id,
            prev_url=prev_url,
            next_url=next_url,
            prev_rev_id=prev_rev_id,
            next_rev_id=next_rev_id,
            revision_metadata=revision_metadata or None,
            is_session=False,
            session_name=None,
        )


class SessionExpressionDetailHandler(BaseHandler):
    """Render expression detail for a session-local (non-catalog) build."""

    async def get(self, name: str):
        from xorq_web.session_store import get_session_entry

        loop = asyncio.get_running_loop()
        buckaroo_port = self.application.settings["buckaroo_port"]

        snapshot = CatalogSnapshot()
        nav_entries = get_catalog_entries(snapshot)

        entry = get_session_entry(name)
        if entry is None:
            self.set_status(404)
            self.write(f"Session expression not found: {name}")
            return

        build_dir = Path(entry["build_path"])
        if not build_dir.exists():
            self.set_status(404)
            self.write(f"Build directory missing: {build_dir}")
            return

        build_id = entry.get("build_id", build_dir.name)

        # Load metadata
        metadata = load_build_metadata(build_dir)

        # Execute to parquet and load into Buckaroo — single executor task so
        # load_expr, to_parquet, and ensure_buckaroo_session all run in the
        # same thread (no expression object crossing thread boundaries).
        buckaroo_session = None
        buckaroo_error = None
        output_path = RESULTS_DIR / f"{build_id}.parquet"
        try:
            await loop.run_in_executor(
                None,
                functools.partial(
                    _load_execute_and_register, build_dir, build_id, output_path, buckaroo_port
                ),
            )
            buckaroo_session = build_id
        except Exception as exc:
            buckaroo_error = str(exc)
            log.error(
                "Failed to load session expression for %s: %s\n%s",
                name,
                exc,
                traceback.format_exc(),
            )

        # Load lineage in executor (calls load_expr a second time)
        lineage = await loop.run_in_executor(None, functools.partial(load_lineage_html, build_dir))

        # Build revision_metadata-like dict from session entry
        revision_metadata = {}
        if entry.get("prompt"):
            revision_metadata["prompt"] = entry["prompt"]
        if entry.get("execute_seconds") is not None:
            revision_metadata["execute_seconds"] = entry["execute_seconds"]

        self.render(
            "expression_detail.html",
            nav_entries=nav_entries,
            current_entry=name,
            display_name=name,
            revision=None,
            build_id=build_id,
            created_at=entry.get("created_at"),
            metadata=metadata,
            buckaroo_port=buckaroo_port,
            buckaroo_session=buckaroo_session,
            buckaroo_error=buckaroo_error,
            lineage=lineage,
            revisions=[],
            current_rev_id=None,
            prev_url=None,
            next_url=None,
            prev_rev_id=None,
            next_rev_id=None,
            revision_metadata=revision_metadata or None,
            is_session=True,
            session_name=name,
        )


class PromoteHandler(BaseHandler):
    """Promote a session expression to the permanent catalog."""

    def post(self, name: str):
        from xorq_web.catalog_utils import register_in_catalog
        from xorq_web.session_store import get_session_entry, remove_session_entry

        entry = get_session_entry(name)
        if entry is None:
            self.set_status(404)
            self.write(f"Session expression not found: {name}")
            return

        build_path = entry["build_path"]
        metadata = {}
        if entry.get("prompt"):
            metadata["prompt"] = entry["prompt"]
        if entry.get("execute_seconds") is not None:
            metadata["execute_seconds"] = entry["execute_seconds"]

        try:
            register_in_catalog(build_path, alias=name, metadata=metadata or None)
        except Exception as exc:
            log.error("Failed to promote session entry %s: %s", name, exc)
            self.set_status(500)
            self.write(f"Promote failed: {exc}")
            return

        remove_session_entry(name)
        self.redirect(f"/entry/{name}")


class DiscardHandler(BaseHandler):
    """Remove a session expression without promoting it."""

    def post(self, name: str):
        from xorq_web.session_store import remove_session_entry

        remove_session_entry(name)
        self.redirect("/")


class RunsHandler(BaseHandler):
    def get(self):
        snapshot = CatalogSnapshot()
        nav_entries = get_catalog_entries(snapshot)
        runs = get_all_runs_merged(snapshot)
        self.render(
            "runs.html",
            nav_entries=nav_entries,
            current_entry="__runs__",
            runs=runs,
        )


class HealthHandler(tornado.web.RequestHandler):
    def get(self):
        self.write({"status": "ok", "service": "xorq-web", "pid": os.getpid()})
