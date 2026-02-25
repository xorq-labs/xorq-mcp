"""Tests for the xorq_web server package."""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import tornado.testing
import tornado.web

from xorq_web.app import make_app
from xorq_web.metadata import (
    _render_node_html,
    get_all_runs,
    get_catalog_entries,
    get_entry_revisions,
    load_build_metadata,
    load_build_schema,
    load_lineage_html,
)


# -----------------------------------------------------------------------
# Test helpers — mock catalog objects
# -----------------------------------------------------------------------
def _mock_empty_snapshot():
    """Return a MagicMock CatalogSnapshot with no entries or aliases."""
    snap = MagicMock()
    snap.entries = ()
    snap.aliases = ()
    snap.contains.return_value = False
    snap.get_catalog_entry.return_value = None
    snap.get_catalog_alias.return_value = None
    snap.aliases_for.return_value = []
    snap.display_name_for.return_value = ""
    return snap


def _mock_catalog_entry(name, aliases=None, exists=True, catalog_path=None, metadata_path=None):
    """Return a MagicMock CatalogEntry."""
    entry = MagicMock()
    entry.name = name
    entry.exists.return_value = exists
    entry.catalog_path = catalog_path or Path(f"/fake/entries/{name}.tgz")
    entry.metadata_path = metadata_path or Path(f"/fake/metadata/{name}.tgz.metadata.yaml")

    # Build alias objects
    alias_mocks = []
    for alias_name in aliases or []:
        ca = MagicMock()
        ca.alias = alias_name
        ca.catalog_entry = entry
        ca.list_revisions.return_value = []
        alias_mocks.append(ca)
    entry.aliases = tuple(alias_mocks)

    return entry


def _mock_snapshot_with_entries(entries_with_aliases):
    """Build a CatalogSnapshot mock from a list of (entry_name, [aliases]) tuples."""
    snap = _mock_empty_snapshot()
    entry_objs = []
    alias_objs = []
    entry_map = {}
    alias_map = {}
    alias_lookup = {}

    for entry_name, aliases in entries_with_aliases:
        entry = _mock_catalog_entry(entry_name, aliases=aliases)
        entry_objs.append(entry)
        entry_map[entry_name] = entry
        alias_lookup[entry_name] = sorted(aliases) if aliases else []
        for alias_name in aliases or []:
            for ca in entry.aliases:
                if ca.alias == alias_name:
                    alias_objs.append(ca)
                    alias_map[alias_name] = ca
                    break

    snap.entries = tuple(entry_objs)
    snap.aliases = tuple(alias_objs)

    def _contains(name):
        return name in entry_map or name in alias_map

    def _get_entry(name):
        if name in entry_map:
            return entry_map[name]
        ca = alias_map.get(name)
        return ca.catalog_entry if ca else None

    def _get_alias(name):
        return alias_map.get(name)

    def _aliases_for(entry_name):
        return alias_lookup.get(entry_name, [])

    def _display_name_for(entry_name):
        als = alias_lookup.get(entry_name, [])
        return als[0] if als else entry_name[:12]

    snap.contains.side_effect = _contains
    snap.get_catalog_entry.side_effect = _get_entry
    snap.get_catalog_alias.side_effect = _get_alias
    snap.aliases_for.side_effect = _aliases_for
    snap.display_name_for.side_effect = _display_name_for
    return snap


@contextmanager
def _fake_extract_build_tgz(build_dir):
    """Context manager that yields a pre-created build directory."""
    yield build_dir


# -----------------------------------------------------------------------
# metadata.py — load_build_metadata
# -----------------------------------------------------------------------
class TestLoadBuildMetadata:
    def test_returns_dict_from_valid_json(self, tmp_path):
        meta = {
            "current_library_version": "0.3.7",
            "git_state": {"commit": "abc123"},
            "sys-version_info": [3, 13, 0, "final", 0],
        }
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        result = load_build_metadata(tmp_path)
        assert result == meta

    def test_returns_empty_dict_when_missing(self, tmp_path):
        result = load_build_metadata(tmp_path)
        assert result == {}

    def test_returns_empty_dict_on_invalid_json(self, tmp_path):
        (tmp_path / "metadata.json").write_text("not json{{{")
        result = load_build_metadata(tmp_path)
        assert result == {}


# -----------------------------------------------------------------------
# metadata.py — load_build_schema
# -----------------------------------------------------------------------
class TestLoadBuildSchema:
    @patch("xorq.ibis_yaml.compiler.load_expr")
    def test_returns_column_tuples(self, mock_load):
        mock_schema = MagicMock()
        mock_schema.names = ["id", "name", "value"]
        mock_schema.types = ["int64", "string", "float64"]
        mock_expr = MagicMock()
        mock_expr.schema.return_value = mock_schema
        mock_load.return_value = mock_expr

        result = load_build_schema(Path("/fake/build"))
        assert result == [("id", "int64"), ("name", "string"), ("value", "float64")]

    @patch("xorq.ibis_yaml.compiler.load_expr", side_effect=Exception("boom"))
    def test_returns_empty_on_error(self, mock_load):
        result = load_build_schema(Path("/fake/build"))
        assert result == []


# -----------------------------------------------------------------------
# metadata.py — load_lineage_html
# -----------------------------------------------------------------------
class TestLoadLineageHtml:
    @patch("xorq.common.utils.lineage_utils.build_column_trees")
    @patch("xorq.ibis_yaml.compiler.load_expr")
    def test_returns_html_per_column(self, mock_load, mock_trees):
        node = MagicMock()
        node.op.__class__.__name__ = "Field"
        node.op.name = "col_a"
        node.children = ()
        mock_trees.return_value = {"col_a": node}

        result = load_lineage_html(Path("/fake/build"))
        assert "col_a" in result
        assert "<li>" in result["col_a"]
        assert "<strong>Field</strong>" in result["col_a"]

    @patch("xorq.ibis_yaml.compiler.load_expr", side_effect=Exception("no expr"))
    def test_returns_empty_on_error(self, mock_load):
        result = load_lineage_html(Path("/fake/build"))
        assert result == {}


# -----------------------------------------------------------------------
# metadata.py — load_lineage_html timeout
# -----------------------------------------------------------------------
class TestLoadLineageHtmlTimeout:
    """load_lineage_html must not hang when build_column_trees is slow.

    Reproduces the hang on complex multi-join expressions (e.g. good_deal_by_position)
    where build_column_trees traverses a shared-subgraph expression DAG without
    memoization, leading to exponential traversal time that never returns.
    """

    @patch("xorq.ibis_yaml.compiler.load_expr")
    @patch("xorq.common.utils.lineage_utils.build_column_trees")
    def test_returns_empty_dict_when_build_column_trees_hangs(
        self, mock_trees, mock_load, tmp_path
    ):
        import threading

        mock_load.return_value = MagicMock()
        never_done = threading.Event()
        mock_trees.side_effect = lambda expr: never_done.wait() or {}

        result_box = {}

        def _call():
            result_box["value"] = load_lineage_html(tmp_path)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=8.0)  # 5s internal timeout + 3s slack
        never_done.set()  # release hanging mock thread

        assert not t.is_alive(), (
            "load_lineage_html is still running after 8s — build_column_trees has no timeout"
        )
        assert result_box.get("value") == {}, (
            f"Expected empty dict on timeout, got {result_box.get('value')!r}"
        )


# -----------------------------------------------------------------------
# metadata.py — _render_node_html
# -----------------------------------------------------------------------
class TestRenderNodeHtml:
    def test_leaf_node(self):
        node = MagicMock()
        node.op.__class__.__name__ = "Field"
        node.op.name = "col_a"
        node.children = ()
        html = _render_node_html(node)
        assert html == "<li><strong>Field</strong>: col_a</li>"

    def test_node_without_name(self):
        node = MagicMock(spec=[])
        node.op = MagicMock(spec=[])
        node.op.__class__ = type("Filter", (), {})
        node.children = ()
        html = _render_node_html(node)
        assert html == "<li><strong>Filter</strong></li>"

    def test_nested_nodes(self):
        child = MagicMock()
        child.op.__class__.__name__ = "DatabaseTable"
        child.op.name = "src"
        child.children = ()

        parent = MagicMock()
        parent.op.__class__.__name__ = "Field"
        parent.op.name = "x"
        parent.children = (child,)

        html = _render_node_html(parent)
        assert "<strong>Field</strong>: x" in html
        assert "<ul>" in html
        assert "<strong>DatabaseTable</strong>: src" in html


# -----------------------------------------------------------------------
# metadata.py — get_catalog_entries
# -----------------------------------------------------------------------
class TestGetCatalogEntries:
    def test_empty_catalog(self):
        snap = _mock_empty_snapshot()
        result = get_catalog_entries(snap)
        assert result == []

    def test_entry_with_alias(self):
        snap = _mock_snapshot_with_entries([("entry-uuid-1234", ["my_alias"])])
        result = get_catalog_entries(snap)
        assert len(result) == 1
        assert result[0]["display_name"] == "my_alias"
        assert result[0]["entry_id"] == "entry-uuid-1234"
        assert "my_alias" in result[0]["aliases"]

    def test_entry_without_alias_uses_truncated_id(self):
        snap = _mock_snapshot_with_entries([("abcdef123456789000", [])])
        result = get_catalog_entries(snap)
        assert len(result) == 1
        assert result[0]["display_name"] == "abcdef123456"


# -----------------------------------------------------------------------
# metadata.py — get_all_runs
# -----------------------------------------------------------------------
class TestGetAllRuns:
    def test_empty_catalog(self):
        snap = _mock_empty_snapshot()
        result = get_all_runs(snap)
        assert result == []

    def test_returns_entries(self):
        snap = _mock_snapshot_with_entries(
            [
                ("entry-a", ["expr_a"]),
                ("entry-b", ["expr_b"]),
            ]
        )
        # Mock _read_entry_metadata for each entry
        with patch("xorq_web.catalog_utils._read_entry_metadata") as mock_meta:
            mock_meta.side_effect = lambda e: (
                {"prompt": "create expr A"} if e.name == "entry-a" else {"execute_seconds": 1.5}
            )
            result = get_all_runs(snap)

        assert len(result) == 2
        # Both entries should appear
        names = {r["display_name"] for r in result}
        assert "expr_a" in names
        assert "expr_b" in names


# -----------------------------------------------------------------------
# metadata.py — get_entry_revisions
# -----------------------------------------------------------------------
class TestGetEntryRevisions:
    def test_returns_empty_for_unknown_target(self):
        snap = _mock_empty_snapshot()
        with patch("xorq_web.catalog_utils.resolve_target", return_value=None):
            result = get_entry_revisions("nonexistent", snap)
            assert result == []

    def test_returns_revisions_from_alias(self):
        snap = _mock_snapshot_with_entries([("entry-uuid", ["my_expr"])])
        entry = snap.get_catalog_entry("my_expr")

        # Add revision history to alias
        mock_commit1 = MagicMock()
        mock_commit1.hexsha = "aabbcc112233"
        mock_commit1.authored_datetime = "2025-01-01T00:00:00"
        mock_commit2 = MagicMock()
        mock_commit2.hexsha = "ddeeff445566"
        mock_commit2.authored_datetime = "2025-02-01T00:00:00"
        entry.aliases[0].list_revisions.return_value = [
            (entry, mock_commit1),
            (entry, mock_commit2),
        ]

        with patch("xorq_web.catalog_utils.resolve_target", return_value=entry):
            result = get_entry_revisions("my_expr", snap)

        assert len(result) == 2
        assert result[0]["revision_id"] == "aabbcc112233"
        assert result[1]["revision_id"] == "ddeeff445566"
        assert result[1]["is_current"] is True
        assert result[0]["is_current"] is False


# -----------------------------------------------------------------------
# app.py — make_app
# -----------------------------------------------------------------------
class TestMakeApp:
    def test_returns_tornado_application(self):
        app = make_app(buckaroo_port=9999)
        assert isinstance(app, tornado.web.Application)
        assert app.settings["buckaroo_port"] == 9999

    def test_default_buckaroo_port(self):
        app = make_app()
        assert app.settings["buckaroo_port"] == 8455

    def test_has_template_and_static_paths(self):
        app = make_app()
        assert "template_path" in app.settings
        assert "static_path" in app.settings
        assert os.path.isdir(app.settings["template_path"])
        assert os.path.isdir(app.settings["static_path"])


# -----------------------------------------------------------------------
# handlers.py — HealthHandler (via Tornado test client)
# -----------------------------------------------------------------------
class TestHealthHandler(tornado.testing.AsyncHTTPTestCase):
    def get_app(self):
        return make_app(buckaroo_port=8455)

    def test_health_returns_ok(self):
        resp = self.fetch("/health")
        assert resp.code == 200
        body = json.loads(resp.body)
        assert body["status"] == "ok"
        assert body["service"] == "xorq-web"
        assert "pid" in body


# -----------------------------------------------------------------------
# handlers.py — CatalogIndexHandler (via Tornado test client)
# -----------------------------------------------------------------------
class TestCatalogIndexHandler(tornado.testing.AsyncHTTPTestCase):
    def get_app(self):
        return make_app(buckaroo_port=8455)

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.get_all_entries", return_value=[])
    @patch("xorq_web.handlers.get_catalog_entries", return_value=[])
    @patch("xorq_web.handlers.CatalogSnapshot")
    def test_empty_catalog_renders(self, mock_snap, mock_entries, mock_all, mock_session):
        mock_snap.return_value = _mock_empty_snapshot()
        resp = self.fetch("/")
        assert resp.code == 200
        body = resp.body.decode()
        assert "xorq" in body
        assert "No catalog entries" in body

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.get_all_entries")
    @patch("xorq_web.handlers.get_catalog_entries")
    @patch("xorq_web.handlers.CatalogSnapshot")
    def test_catalog_with_entries(self, mock_snap, mock_entries, mock_all, mock_session):
        mock_snap.return_value = _mock_empty_snapshot()
        entry = {
            "display_name": "test_expr",
            "aliases": ["test_expr"],
            "entry_id": "uuid-1",
            "revision": "r1",
            "build_id": "abc123",
            "created_at": "2025-01-01",
            "source": "catalog",
        }
        mock_entries.return_value = [
            {
                "display_name": "test_expr",
                "aliases": ["test_expr"],
                "entry_id": "uuid-1",
                "revision": "r1",
                "build_id": "abc123",
                "created_at": "2025-01-01",
            },
        ]
        mock_all.return_value = [entry]
        resp = self.fetch("/")
        assert resp.code == 200
        body = resp.body.decode()
        assert "test_expr" in body
        assert "/entry/test_expr" in body


# -----------------------------------------------------------------------
# handlers.py — RunsHandler (via Tornado test client)
# -----------------------------------------------------------------------
class TestRunsHandler(tornado.testing.AsyncHTTPTestCase):
    def get_app(self):
        return make_app(buckaroo_port=8455)

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.get_catalog_entries", return_value=[])
    @patch("xorq_web.handlers.get_all_runs_merged", return_value=[])
    @patch("xorq_web.handlers.CatalogSnapshot")
    def test_empty_runs_renders(self, mock_snap, mock_runs, mock_entries, mock_session):
        mock_snap.return_value = _mock_empty_snapshot()
        resp = self.fetch("/runs")
        assert resp.code == 200
        body = resp.body.decode()
        assert "No runs yet" in body

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.get_catalog_entries", return_value=[])
    @patch("xorq_web.handlers.get_all_runs_merged")
    @patch("xorq_web.handlers.CatalogSnapshot")
    def test_runs_with_data(self, mock_snap, mock_runs, mock_entries, mock_session):
        mock_snap.return_value = _mock_empty_snapshot()
        mock_runs.return_value = [
            {
                "display_name": "my_expr",
                "entry_id": "uuid-1",
                "revision_id": "r2",
                "build_id": "build-r2",
                "created_at": "2025-06-01",
                "prompt": "add more rows",
                "execute_seconds": 2.5,
                "source": "catalog",
            },
            {
                "display_name": "my_expr",
                "entry_id": "uuid-1",
                "revision_id": "r1",
                "build_id": "build-r1",
                "created_at": "2025-01-01",
                "prompt": None,
                "execute_seconds": None,
                "source": "catalog",
            },
        ]
        resp = self.fetch("/runs")
        assert resp.code == 200
        body = resp.body.decode()
        assert "my_expr" in body
        assert "/entry/my_expr@r2" in body
        assert "/entry/my_expr@r1" in body
        assert "add more rows" in body
        assert "2.5s" in body


# -----------------------------------------------------------------------
# handlers.py — ExpressionDetailHandler (via Tornado test client)
# -----------------------------------------------------------------------
class TestExpressionDetailHandler(tornado.testing.AsyncHTTPTestCase):
    def get_app(self):
        return make_app(buckaroo_port=8455)

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.get_catalog_entries", return_value=[])
    @patch("xorq_web.handlers.resolve_target", return_value=None)
    @patch("xorq_web.handlers.CatalogSnapshot")
    def test_missing_target_returns_404(self, mock_snap, mock_resolve, mock_entries, mock_session):
        mock_snap.return_value = _mock_empty_snapshot()
        resp = self.fetch("/entry/nonexistent")
        assert resp.code == 404

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.load_lineage_html", return_value={})
    @patch(
        "xorq_web.handlers.load_build_metadata",
        return_value={"current_library_version": "0.3.7"},
    )
    @patch("xorq_web.handlers.ensure_buckaroo_session", return_value={"session": "abc"})
    @patch("xorq_web.handlers.get_entry_revisions", return_value=[])
    @patch("xorq_web.handlers.get_catalog_entries")
    @patch("xorq_web.handlers._read_entry_metadata", return_value={})
    @patch("xorq_web.handlers.resolve_target")
    @patch("xorq_web.handlers.CatalogSnapshot")
    @patch("xorq.ibis_yaml.compiler.load_expr")
    @patch("xorq.common.utils.caching_utils.get_xorq_cache_dir", return_value="/tmp/cache")
    def test_valid_target_renders_detail(
        self,
        mock_cache,
        mock_load_expr,
        mock_snap,
        mock_resolve,
        mock_meta_entry,
        mock_entries,
        mock_revisions,
        mock_bk_session,
        mock_meta,
        mock_lineage,
        mock_session,
    ):
        import tempfile

        mock_snap.return_value = _mock_snapshot_with_entries([("abc123", ["my_expr"])])

        mock_entry = _mock_catalog_entry("abc123", aliases=["my_expr"])
        mock_resolve.return_value = mock_entry

        mock_entries.return_value = [
            {
                "display_name": "my_expr",
                "aliases": ["my_expr"],
                "entry_id": "abc123",
                "revision": None,
                "build_id": "abc123",
                "created_at": "2025-06-01",
            },
        ]
        mock_load_expr.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as td:
            build_dir = Path(td) / "abc123"
            build_dir.mkdir()

            with patch(
                "xorq.catalog.tar_utils.extract_build_tgz_context",
                return_value=_fake_extract_build_tgz(build_dir),
            ):
                resp = self.fetch("/entry/my_expr")
                assert resp.code == 200
                body = resp.body.decode()
                assert "my_expr" in body
                assert "abc123" in body
                assert "0.3.7" in body

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.load_lineage_html", return_value={})
    @patch("xorq_web.handlers.load_build_metadata", return_value={})
    @patch("xorq_web.handlers.ensure_buckaroo_session", return_value={"session": "s1"})
    @patch("xorq_web.handlers.get_entry_revisions")
    @patch("xorq_web.handlers.get_catalog_entries")
    @patch("xorq_web.handlers._read_entry_metadata", return_value={})
    @patch("xorq_web.handlers.resolve_target")
    @patch("xorq_web.handlers.CatalogSnapshot")
    @patch("xorq.ibis_yaml.compiler.load_expr")
    @patch("xorq.common.utils.caching_utils.get_xorq_cache_dir", return_value="/tmp/cache")
    def test_revision_nav_renders_prev_next(
        self,
        mock_cache,
        mock_load_expr,
        mock_snap,
        mock_resolve,
        mock_meta_entry,
        mock_entries,
        mock_revisions,
        mock_bk_session,
        mock_meta,
        mock_lineage,
        mock_session,
    ):
        import tempfile

        snap = _mock_snapshot_with_entries([("uuid-1", ["my_expr"])])
        mock_snap.return_value = snap

        mock_entry = _mock_catalog_entry("uuid-1", aliases=["my_expr"])
        # Set up revision history on the alias
        mock_commit = MagicMock()
        mock_commit.hexsha = "r2r2r2r2r2r2"
        mock_commit.authored_datetime = "2025-06-01"
        mock_entry.aliases[0].list_revisions.return_value = [(mock_entry, mock_commit)]
        mock_resolve.return_value = mock_entry

        mock_revisions.return_value = [
            {
                "revision_id": "r1r1r1r1r1r1",
                "build_id": "build-r1",
                "created_at": "2025-01-01",
                "is_current": False,
            },
            {
                "revision_id": "r2r2r2r2r2r2",
                "build_id": "build-r2",
                "created_at": "2025-06-01",
                "is_current": False,
            },
            {
                "revision_id": "r3r3r3r3r3r3",
                "build_id": "build-r3",
                "created_at": "2025-09-01",
                "is_current": True,
            },
        ]
        mock_entries.return_value = [
            {
                "display_name": "my_expr",
                "aliases": ["my_expr"],
                "entry_id": "uuid-1",
                "revision": None,
                "build_id": "uuid-1",
                "created_at": "2025-09-01",
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            build_dir = Path(td) / "build-r2"
            build_dir.mkdir()
            mock_load_expr.return_value = MagicMock()

            with patch(
                "xorq.catalog.tar_utils.extract_build_tgz_context",
                return_value=_fake_extract_build_tgz(build_dir),
            ):
                resp = self.fetch("/entry/my_expr@r2")
                assert resp.code == 200
                body = resp.body.decode()

                # Should show revision nav with prev and next
                assert "my_expr@r1" in body
                assert "my_expr@r3" in body

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.load_lineage_html", return_value={})
    @patch("xorq_web.handlers.load_build_metadata", return_value={})
    @patch("xorq_web.handlers.ensure_buckaroo_session", return_value={"session": "s1"})
    @patch("xorq_web.handlers.get_entry_revisions")
    @patch("xorq_web.handlers.get_catalog_entries")
    @patch("xorq_web.handlers._read_entry_metadata", return_value={})
    @patch("xorq_web.handlers.resolve_target")
    @patch("xorq_web.handlers.CatalogSnapshot")
    @patch("xorq.ibis_yaml.compiler.load_expr")
    @patch("xorq.common.utils.caching_utils.get_xorq_cache_dir", return_value="/tmp/cache")
    def test_no_revision_nav_for_single_revision(
        self,
        mock_cache,
        mock_load_expr,
        mock_snap,
        mock_resolve,
        mock_meta_entry,
        mock_entries,
        mock_revisions,
        mock_bk_session,
        mock_meta,
        mock_lineage,
        mock_session,
    ):
        import tempfile

        snap = _mock_snapshot_with_entries([("uuid-1", ["solo_expr"])])
        mock_snap.return_value = snap

        mock_entry = _mock_catalog_entry("uuid-1", aliases=["solo_expr"])
        mock_resolve.return_value = mock_entry

        mock_revisions.return_value = [
            {
                "revision_id": "r1r1r1r1r1r1",
                "build_id": "build-1",
                "created_at": "2025-01-01",
                "is_current": True,
            },
        ]
        mock_entries.return_value = [
            {
                "display_name": "solo_expr",
                "aliases": ["solo_expr"],
                "entry_id": "uuid-1",
                "revision": None,
                "build_id": "uuid-1",
                "created_at": "2025-01-01",
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            build_dir = Path(td) / "build-1"
            build_dir.mkdir()
            mock_load_expr.return_value = MagicMock()

            with patch(
                "xorq.catalog.tar_utils.extract_build_tgz_context",
                return_value=_fake_extract_build_tgz(build_dir),
            ):
                resp = self.fetch("/entry/solo_expr")
                assert resp.code == 200
                body = resp.body.decode()
                # Should NOT render the revision nav bar for a single revision
                assert "revision-nav" not in body

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.load_lineage_html", return_value={})
    @patch("xorq_web.handlers.load_build_metadata", return_value={})
    @patch("xorq_web.handlers.ensure_buckaroo_session", return_value={"session": "s1"})
    @patch("xorq_web.handlers.get_entry_revisions", return_value=[])
    @patch("xorq_web.handlers.get_catalog_entries")
    @patch("xorq_web.handlers._read_entry_metadata")
    @patch("xorq_web.handlers.resolve_target")
    @patch("xorq_web.handlers.CatalogSnapshot")
    @patch("xorq.ibis_yaml.compiler.load_expr")
    @patch("xorq.common.utils.caching_utils.get_xorq_cache_dir", return_value="/tmp/cache")
    def test_prompt_metadata_displayed(
        self,
        mock_cache,
        mock_load_expr,
        mock_snap,
        mock_resolve,
        mock_meta_entry,
        mock_entries,
        mock_revisions,
        mock_bk_session,
        mock_meta,
        mock_lineage,
        mock_session,
    ):
        import tempfile

        snap = _mock_snapshot_with_entries([("uuid-1", ["my_expr"])])
        mock_snap.return_value = snap

        mock_entry = _mock_catalog_entry("uuid-1", aliases=["my_expr"])
        mock_resolve.return_value = mock_entry
        mock_meta_entry.return_value = {"prompt": "expand the number of rows to 20"}

        mock_entries.return_value = [
            {
                "display_name": "my_expr",
                "aliases": ["my_expr"],
                "entry_id": "uuid-1",
                "revision": None,
                "build_id": "uuid-1",
                "created_at": "2025-06-01",
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            build_dir = Path(td) / "build-r2"
            build_dir.mkdir()
            mock_load_expr.return_value = MagicMock()

            with patch(
                "xorq.catalog.tar_utils.extract_build_tgz_context",
                return_value=_fake_extract_build_tgz(build_dir),
            ):
                resp = self.fetch("/entry/my_expr@r2")
                assert resp.code == 200
                body = resp.body.decode()
                assert "prompt-block" in body
                assert "expand the number of rows to 20" in body

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.load_lineage_html", return_value={})
    @patch("xorq_web.handlers.load_build_metadata", return_value={})
    @patch("xorq_web.handlers.ensure_buckaroo_session", return_value={"session": "s1"})
    @patch("xorq_web.handlers.get_entry_revisions", return_value=[])
    @patch("xorq_web.handlers.get_catalog_entries")
    @patch("xorq_web.handlers._read_entry_metadata", return_value={})
    @patch("xorq_web.handlers.resolve_target")
    @patch("xorq_web.handlers.CatalogSnapshot")
    @patch("xorq.ibis_yaml.compiler.load_expr")
    @patch("xorq.common.utils.caching_utils.get_xorq_cache_dir", return_value="/tmp/cache")
    def test_no_prompt_block_when_no_metadata(
        self,
        mock_cache,
        mock_load_expr,
        mock_snap,
        mock_resolve,
        mock_meta_entry,
        mock_entries,
        mock_revisions,
        mock_bk_session,
        mock_meta,
        mock_lineage,
        mock_session,
    ):
        import tempfile

        snap = _mock_snapshot_with_entries([("uuid-1", ["my_expr"])])
        mock_snap.return_value = snap

        mock_entry = _mock_catalog_entry("uuid-1", aliases=["my_expr"])
        mock_resolve.return_value = mock_entry

        mock_entries.return_value = [
            {
                "display_name": "my_expr",
                "aliases": ["my_expr"],
                "entry_id": "uuid-1",
                "revision": None,
                "build_id": "uuid-1",
                "created_at": "2025-01-01",
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            build_dir = Path(td) / "build-1"
            build_dir.mkdir()
            mock_load_expr.return_value = MagicMock()

            with patch(
                "xorq.catalog.tar_utils.extract_build_tgz_context",
                return_value=_fake_extract_build_tgz(build_dir),
            ):
                resp = self.fetch("/entry/my_expr")
                assert resp.code == 200
                body = resp.body.decode()
                assert "prompt-block" not in body


# -----------------------------------------------------------------------
# xorq_mcp_tool.py — web server management
# -----------------------------------------------------------------------
class TestWebServerConfig:
    def test_web_port_defaults_to_buckaroo_plus_one(self):
        from xorq_mcp_tool import SERVER_PORT, WEB_PORT

        assert WEB_PORT == SERVER_PORT + 1

    def test_web_url_uses_web_port(self):
        from xorq_mcp_tool import WEB_PORT, WEB_URL

        assert str(WEB_PORT) in WEB_URL
        assert WEB_URL == f"http://localhost:{WEB_PORT}"


class TestEnsureWebServer:
    @patch("xorq_mcp_tool._web_health_check")
    def test_reuses_running_server(self, mock_health):
        mock_health.return_value = {"status": "ok", "pid": 1234}
        from xorq_mcp_tool import ensure_web_server

        result = ensure_web_server()
        assert result["web_status"] == "reused"
        assert result["web_pid"] == 1234

    @patch("xorq_mcp_tool._start_server_monitor")
    @patch("xorq_mcp_tool.subprocess.Popen")
    @patch("xorq_mcp_tool._web_health_check")
    def test_starts_server_when_not_running(self, mock_health, mock_popen, mock_monitor):
        # First call: not running. Second call (in poll loop): running.
        mock_health.side_effect = [None, {"status": "ok", "pid": 5678}]
        mock_proc = MagicMock()
        mock_proc.pid = 5678
        mock_popen.return_value = mock_proc

        from xorq_mcp_tool import ensure_web_server

        with patch("builtins.open", MagicMock()):
            result = ensure_web_server()

        assert result["web_status"] == "started"
        assert result["web_pid"] == 5678
        mock_popen.assert_called_once()
        # Verify the command includes xorq_web module
        cmd_args = mock_popen.call_args[0][0]
        assert "-m" in cmd_args
        assert "xorq_web" in cmd_args


class TestCleanupServer:
    @patch("xorq_mcp_tool._web_server_monitor", None)
    @patch("xorq_mcp_tool._web_server_proc", None)
    @patch("xorq_mcp_tool._server_monitor", None)
    @patch("xorq_mcp_tool._server_proc", None)
    def test_cleanup_with_no_servers(self):
        """Cleanup should not raise when no servers are running."""
        from xorq_mcp_tool import _cleanup_server

        _cleanup_server()  # should not raise


# -----------------------------------------------------------------------
# xorq_mcp_tool.py — catalog_ls opens web URL
# -----------------------------------------------------------------------
class TestCatalogLsOpensWebUrl:
    @patch("webbrowser.open")
    @patch("xorq_mcp_tool.ensure_web_server", return_value={"web_status": "reused"})
    @patch("xorq_mcp_tool.ensure_server", return_value={"server_status": "reused"})
    def test_opens_web_catalog_url(self, mock_srv, mock_web, mock_browser):
        snap = _mock_snapshot_with_entries([("entry-1", ["my_entry"])])

        with patch("xorq_web.catalog_utils.CatalogSnapshot", return_value=snap):
            from xorq_mcp_tool import WEB_URL, xorq_catalog_ls

            result = xorq_catalog_ls()

        mock_browser.assert_called_once()
        opened_url = mock_browser.call_args[0][0]
        assert opened_url.startswith(WEB_URL)
        assert "1 entries" in result


# -----------------------------------------------------------------------
# session_store.py
# -----------------------------------------------------------------------
class TestSessionStore:
    def test_add_and_get_entries(self, tmp_path, monkeypatch):
        from xorq_web import session_store

        monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
        session_store.add_session_entry(
            name="my_draft", build_path="/tmp/build/abc", build_id="abc123", prompt="test"
        )
        entries = session_store.get_session_entries()
        assert len(entries) == 1
        assert entries[0]["name"] == "my_draft"
        assert entries[0]["build_id"] == "abc123"
        assert entries[0]["prompt"] == "test"

    def test_add_overwrites_same_name(self, tmp_path, monkeypatch):
        from xorq_web import session_store

        monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
        session_store.add_session_entry(name="draft", build_path="/tmp/build/aaa", build_id="aaa")
        session_store.add_session_entry(name="draft", build_path="/tmp/build/bbb", build_id="bbb")
        entries = session_store.get_session_entries()
        assert len(entries) == 1
        assert entries[0]["build_id"] == "bbb"

    def test_remove_entry(self, tmp_path, monkeypatch):
        from xorq_web import session_store

        monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
        session_store.add_session_entry(name="to_remove", build_path="/tmp/x", build_id="x")
        assert session_store.remove_session_entry("to_remove") is True
        assert session_store.get_session_entries() == []

    def test_remove_nonexistent_returns_false(self, tmp_path, monkeypatch):
        from xorq_web import session_store

        monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
        assert session_store.remove_session_entry("nope") is False

    def test_get_session_entry(self, tmp_path, monkeypatch):
        from xorq_web import session_store

        monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
        session_store.add_session_entry(name="findme", build_path="/tmp/b", build_id="b1")
        entry = session_store.get_session_entry("findme")
        assert entry is not None
        assert entry["build_id"] == "b1"
        assert session_store.get_session_entry("nope") is None

    def test_update_metadata(self, tmp_path, monkeypatch):
        from xorq_web import session_store

        monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
        session_store.add_session_entry(name="upd", build_path="/tmp/u", build_id="u1")
        assert session_store.update_session_entry_metadata("upd", {"execute_seconds": 1.5})
        entry = session_store.get_session_entry("upd")
        assert entry["execute_seconds"] == 1.5

    def test_cleanup_session(self, tmp_path, monkeypatch):
        from xorq_web import session_store

        monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
        session_store.add_session_entry(name="clean", build_path="/tmp/c", build_id="c1")
        assert (tmp_path / f"{os.getpid()}.json").exists()
        session_store.cleanup_session()
        assert not (tmp_path / f"{os.getpid()}.json").exists()

    def test_stale_pid_cleanup(self, tmp_path, monkeypatch):
        from xorq_web import session_store

        monkeypatch.setattr(session_store, "SESSIONS_DIR", tmp_path)
        # Write a manifest for a dead PID
        dead_pid = 99999999
        manifest = tmp_path / f"{dead_pid}.json"
        manifest.write_text(json.dumps([{"name": "stale", "build_path": "/x", "build_id": "x"}]))
        entries = session_store.get_session_entries()
        # Should be empty (stale PID cleaned up)
        assert len(entries) == 0
        assert not manifest.exists()


# -----------------------------------------------------------------------
# handlers.py — SessionExpressionDetailHandler
# -----------------------------------------------------------------------
class TestSessionExpressionDetailHandler(tornado.testing.AsyncHTTPTestCase):
    def get_app(self):
        return make_app(buckaroo_port=8455)

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.get_catalog_entries", return_value=[])
    @patch("xorq_web.handlers.CatalogSnapshot")
    @patch("xorq_web.session_store.get_session_entry", return_value=None)
    def test_missing_session_returns_404(self, mock_entry, mock_snap, mock_entries, mock_session):
        mock_snap.return_value = _mock_empty_snapshot()
        resp = self.fetch("/session/nonexistent")
        assert resp.code == 404

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.load_lineage_html", return_value={})
    @patch("xorq_web.handlers.load_build_metadata", return_value={})
    @patch("xorq_web.handlers.ensure_buckaroo_session", return_value={"session": "abc"})
    @patch("xorq_web.handlers.get_catalog_entries", return_value=[])
    @patch("xorq_web.handlers.CatalogSnapshot")
    @patch("xorq.ibis_yaml.compiler.load_expr")
    @patch("xorq.common.utils.caching_utils.get_xorq_cache_dir", return_value="/tmp/cache")
    def test_valid_session_renders(
        self,
        mock_cache,
        mock_load_expr,
        mock_snap,
        mock_entries,
        mock_bk_session,
        mock_meta,
        mock_lineage,
        mock_session,
    ):
        import tempfile

        mock_snap.return_value = _mock_empty_snapshot()
        mock_load_expr.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as td:
            build_dir = Path(td) / "abc123"
            build_dir.mkdir()

            with patch("xorq_web.session_store.get_session_entry") as mock_entry:
                mock_entry.return_value = {
                    "name": "my_draft",
                    "build_path": str(build_dir),
                    "build_id": "abc123",
                    "created_at": "2025-06-01",
                    "prompt": "test prompt",
                    "execute_seconds": 1.2,
                }
                resp = self.fetch("/session/my_draft")

        assert resp.code == 200
        body = resp.body.decode()
        assert "my_draft" in body
        assert "session-actions" in body
        assert "Promote to Catalog" in body
        assert "Discard" in body
        assert "test prompt" in body


# -----------------------------------------------------------------------
# handlers.py — DiscardHandler
# -----------------------------------------------------------------------
class TestDiscardHandler(tornado.testing.AsyncHTTPTestCase):
    def get_app(self):
        return make_app(buckaroo_port=8455)

    @patch("xorq_web.session_store.remove_session_entry", return_value=True)
    def test_discard_redirects_to_index(self, mock_remove):
        resp = self.fetch(
            "/session/my_draft/discard", method="POST", body=b"", follow_redirects=False
        )
        assert resp.code == 302
        assert resp.headers["Location"] == "/"
        mock_remove.assert_called_once_with("my_draft")


# -----------------------------------------------------------------------
# handlers.py — async handler properties
# -----------------------------------------------------------------------
class TestHandlersAreAsync:
    """Verify that the detail handlers are coroutines (async def), not plain methods.

    This is the regression guard for GitHub issue #1: blocking I/O in synchronous
    Tornado handlers caused the server to hang while executing expressions.
    """

    def test_expression_detail_handler_is_async(self):
        import asyncio

        from xorq_web.handlers import ExpressionDetailHandler

        assert asyncio.iscoroutinefunction(ExpressionDetailHandler.get)

    def test_session_expression_detail_handler_is_async(self):
        import asyncio

        from xorq_web.handlers import SessionExpressionDetailHandler

        assert asyncio.iscoroutinefunction(SessionExpressionDetailHandler.get)


class TestQueryParamPassthrough(tornado.testing.AsyncHTTPTestCase):
    """Verify that unknown query params (e.g. ?s=SESSION_ID) are silently ignored.

    The xorq-mcp browser focus feature (issue #3) appends ?s=<session_id> to
    all web URLs. The server must serve normal responses regardless.
    """

    def get_app(self):
        return make_app(buckaroo_port=8455)

    def test_health_ignores_query_param(self):
        resp = self.fetch("/health?s=abc123")
        assert resp.code == 200
        body = resp.json() if hasattr(resp, "json") else __import__("json").loads(resp.body)
        assert body["status"] == "ok"

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.get_all_entries", return_value=[])
    @patch("xorq_web.handlers.get_catalog_entries", return_value=[])
    @patch("xorq_web.handlers.CatalogSnapshot")
    def test_catalog_index_ignores_query_param(
        self, mock_snap, mock_entries, mock_all, mock_session
    ):
        mock_snap.return_value = _mock_empty_snapshot()
        resp = self.fetch("/?s=abc123")
        assert resp.code == 200


# -----------------------------------------------------------------------
# catalog_utils.py — CatalogSnapshot
# -----------------------------------------------------------------------
class TestCatalogSnapshot:
    def test_empty_snapshot(self):
        mock_catalog = MagicMock()
        mock_catalog.catalog_entries = ()
        mock_catalog.catalog_aliases = ()

        from xorq_web.catalog_utils import CatalogSnapshot

        snap = CatalogSnapshot(mock_catalog)
        assert snap.entries == ()
        assert snap.aliases == ()
        assert snap.contains("anything") is False
        assert snap.get_catalog_entry("anything") is None
        assert snap.aliases_for("anything") == []

    def test_resolve_by_entry_name(self):
        mock_entry = MagicMock()
        mock_entry.name = "abc123"
        mock_entry.aliases = ()

        mock_catalog = MagicMock()
        mock_catalog.catalog_entries = (mock_entry,)
        mock_catalog.catalog_aliases = ()

        from xorq_web.catalog_utils import CatalogSnapshot

        snap = CatalogSnapshot(mock_catalog)
        assert snap.contains("abc123") is True
        assert snap.get_catalog_entry("abc123") is mock_entry

    def test_resolve_by_alias(self):
        mock_entry = MagicMock()
        mock_entry.name = "abc123"
        mock_entry.aliases = ()

        mock_alias = MagicMock()
        mock_alias.alias = "my_alias"
        mock_alias.catalog_entry = mock_entry

        mock_catalog = MagicMock()
        mock_catalog.catalog_entries = (mock_entry,)
        mock_catalog.catalog_aliases = (mock_alias,)

        from xorq_web.catalog_utils import CatalogSnapshot

        snap = CatalogSnapshot(mock_catalog)
        assert snap.contains("my_alias") is True
        assert snap.get_catalog_entry("my_alias") is mock_entry
        assert snap.aliases_for("abc123") == ["my_alias"]
        assert snap.display_name_for("abc123") == "my_alias"

    def test_display_name_truncates_without_alias(self):
        mock_entry = MagicMock()
        mock_entry.name = "abcdef123456789000"
        mock_entry.aliases = ()

        mock_catalog = MagicMock()
        mock_catalog.catalog_entries = (mock_entry,)
        mock_catalog.catalog_aliases = ()

        from xorq_web.catalog_utils import CatalogSnapshot

        snap = CatalogSnapshot(mock_catalog)
        assert snap.display_name_for("abcdef123456789000") == "abcdef123456"


# -----------------------------------------------------------------------
# catalog_utils.py — resolve_target
# -----------------------------------------------------------------------
class TestResolveTarget:
    def test_returns_none_for_unknown(self):
        from xorq_web.catalog_utils import resolve_target

        snap = _mock_empty_snapshot()
        assert resolve_target("nope", snap) is None

    def test_resolves_entry_name(self):
        from xorq_web.catalog_utils import resolve_target

        snap = _mock_snapshot_with_entries([("abc123", ["my_alias"])])
        entry = resolve_target("abc123", snap)
        assert entry is not None
        assert entry.name == "abc123"

    def test_resolves_alias(self):
        from xorq_web.catalog_utils import resolve_target

        snap = _mock_snapshot_with_entries([("abc123", ["my_alias"])])
        entry = resolve_target("my_alias", snap)
        assert entry is not None
        assert entry.name == "abc123"

    def test_strips_revision_suffix(self):
        from xorq_web.catalog_utils import resolve_target

        snap = _mock_snapshot_with_entries([("abc123", ["my_alias"])])
        entry = resolve_target("my_alias@r2", snap)
        assert entry is not None
        assert entry.name == "abc123"


# -----------------------------------------------------------------------
# catalog_utils.py — metadata read/write
# -----------------------------------------------------------------------
class TestEntryMetadata:
    def test_read_returns_empty_on_missing(self):
        from xorq_web.catalog_utils import _read_entry_metadata

        entry = MagicMock()
        entry.metadata_path = Path("/nonexistent/metadata.yaml")
        entry.name = "test"
        result = _read_entry_metadata(entry)
        assert result == {}

    def test_read_returns_dict(self, tmp_path):
        import yaml

        from xorq_web.catalog_utils import _read_entry_metadata

        meta_path = tmp_path / "meta.yaml"
        meta_path.write_text(yaml.dump({"prompt": "hello", "execute_seconds": 1.5}))

        entry = MagicMock()
        entry.metadata_path = meta_path
        entry.name = "test"
        result = _read_entry_metadata(entry)
        assert result == {"prompt": "hello", "execute_seconds": 1.5}

    def test_read_returns_empty_on_malformed(self, tmp_path):
        from xorq_web.catalog_utils import _read_entry_metadata

        meta_path = tmp_path / "meta.yaml"
        meta_path.write_text("not: valid: yaml: {{{}}")

        entry = MagicMock()
        entry.metadata_path = meta_path
        entry.name = "test"
        result = _read_entry_metadata(entry)
        assert result == {}

    def test_write_merges(self, tmp_path):
        import yaml

        from xorq_web.catalog_utils import _read_entry_metadata, _write_entry_metadata

        meta_path = tmp_path / "meta.yaml"
        meta_path.write_text(yaml.dump({"prompt": "hello"}))

        entry = MagicMock()
        entry.metadata_path = meta_path
        entry.name = "test"
        _write_entry_metadata(entry, {"execute_seconds": 2.0})

        result = _read_entry_metadata(entry)
        assert result == {"prompt": "hello", "execute_seconds": 2.0}
