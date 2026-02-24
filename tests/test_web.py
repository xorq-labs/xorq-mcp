"""Tests for the xorq_web server package."""

import json
import os
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
        from xorq.catalog import XorqCatalog

        with patch(
            "xorq.catalog.load_catalog",
            return_value=XorqCatalog(),
        ):
            result = get_catalog_entries()
            assert result == []

    def test_entry_with_alias(self):
        """Entries with aliases use the alias as display_name."""
        mock_build = MagicMock()
        mock_build.build_id = "abc123"

        mock_rev = MagicMock()
        mock_rev.revision_id = "r1"
        mock_rev.build = mock_build
        mock_rev.created_at = "2025-01-01T00:00:00"

        mock_entry = MagicMock()
        mock_entry.entry_id = "entry-uuid-1234"
        mock_entry.current_revision = "r1"
        mock_entry.history = [mock_rev]

        mock_alias = MagicMock()
        mock_alias.entry_id = "entry-uuid-1234"

        mock_catalog = MagicMock()
        mock_catalog.entries = [mock_entry]
        mock_catalog.aliases = {"my_alias": mock_alias}

        with patch("xorq.catalog.load_catalog", return_value=mock_catalog):
            result = get_catalog_entries()
            assert len(result) == 1
            assert result[0]["display_name"] == "my_alias"
            assert result[0]["build_id"] == "abc123"
            assert result[0]["revision"] == "r1"
            assert "my_alias" in result[0]["aliases"]

    def test_entry_without_alias_uses_truncated_id(self):
        mock_build = MagicMock()
        mock_build.build_id = "def456"

        mock_rev = MagicMock()
        mock_rev.revision_id = "r1"
        mock_rev.build = mock_build
        mock_rev.created_at = None

        mock_entry = MagicMock()
        mock_entry.entry_id = "abcdef123456789000"
        mock_entry.current_revision = "r1"
        mock_entry.history = [mock_rev]

        mock_catalog = MagicMock()
        mock_catalog.entries = [mock_entry]
        mock_catalog.aliases = {}

        with patch("xorq.catalog.load_catalog", return_value=mock_catalog):
            result = get_catalog_entries()
            assert len(result) == 1
            assert result[0]["display_name"] == "abcdef123456"
            assert result[0]["created_at"] is None


# -----------------------------------------------------------------------
# metadata.py — get_all_runs
# -----------------------------------------------------------------------
class TestGetAllRuns:
    def test_empty_catalog(self):
        mock_catalog = MagicMock()
        mock_catalog.entries = []
        mock_catalog.aliases = {}
        with patch("xorq.catalog.load_catalog", return_value=mock_catalog):
            result = get_all_runs()
            assert result == []

    def test_returns_all_revisions_across_entries(self):
        mock_build_a = MagicMock()
        mock_build_a.build_id = "build-a"
        mock_rev_a = MagicMock()
        mock_rev_a.revision_id = "r1"
        mock_rev_a.build = mock_build_a
        mock_rev_a.created_at = "2025-01-01"
        mock_rev_a.metadata = {"prompt": "create expr A"}

        mock_build_b = MagicMock()
        mock_build_b.build_id = "build-b"
        mock_rev_b = MagicMock()
        mock_rev_b.revision_id = "r1"
        mock_rev_b.build = mock_build_b
        mock_rev_b.created_at = "2025-02-01"
        mock_rev_b.metadata = {"execute_seconds": 1.5}

        mock_entry_a = MagicMock()
        mock_entry_a.entry_id = "entry-a"
        mock_entry_a.history = [mock_rev_a]
        mock_entry_b = MagicMock()
        mock_entry_b.entry_id = "entry-b"
        mock_entry_b.history = [mock_rev_b]

        mock_alias_a = MagicMock()
        mock_alias_a.entry_id = "entry-a"
        mock_alias_b = MagicMock()
        mock_alias_b.entry_id = "entry-b"

        mock_catalog = MagicMock()
        mock_catalog.entries = [mock_entry_a, mock_entry_b]
        mock_catalog.aliases = {"expr_a": mock_alias_a, "expr_b": mock_alias_b}

        with patch("xorq.catalog.load_catalog", return_value=mock_catalog):
            result = get_all_runs()

        assert len(result) == 2
        # Sorted newest first
        assert result[0]["display_name"] == "expr_b"
        assert result[0]["execute_seconds"] == 1.5
        assert result[1]["display_name"] == "expr_a"
        assert result[1]["prompt"] == "create expr A"


# -----------------------------------------------------------------------
# metadata.py — get_entry_revisions
# -----------------------------------------------------------------------
class TestGetEntryRevisions:
    def _make_revision(self, revision_id, build_id=None, created_at=None):
        mock_rev = MagicMock()
        mock_rev.revision_id = revision_id
        mock_rev.created_at = created_at
        if build_id:
            mock_rev.build = MagicMock()
            mock_rev.build.build_id = build_id
        else:
            mock_rev.build = None
        return mock_rev

    def _make_catalog(self, entry_id, revisions, alias_name=None, current_revision=None):
        mock_entry = MagicMock()
        mock_entry.entry_id = entry_id
        mock_entry.current_revision = current_revision or revisions[-1].revision_id
        mock_entry.history = revisions

        mock_catalog = MagicMock()
        mock_catalog.entries = [mock_entry]

        if alias_name:
            mock_alias = MagicMock()
            mock_alias.entry_id = entry_id
            mock_alias.revision_id = current_revision or revisions[-1].revision_id
            mock_catalog.aliases = {alias_name: mock_alias}
        else:
            mock_catalog.aliases = {}

        # Wire up maybe_get_entry
        mock_catalog.maybe_get_entry = lambda eid: mock_entry if eid == entry_id else None

        return mock_catalog

    def test_returns_empty_for_unknown_target(self):
        from xorq.catalog import Target

        with patch("xorq.catalog.load_catalog") as mock_load:
            mock_catalog = MagicMock()
            mock_catalog.entries = []
            mock_catalog.aliases = {}
            mock_load.return_value = mock_catalog
            with patch.object(Target, "from_str", return_value=None):
                result = get_entry_revisions("nonexistent")
                assert result == []

    def test_returns_all_revisions_for_alias(self):
        from xorq.catalog import Target

        r1 = self._make_revision("r1", build_id="build-aaa", created_at="2025-01-01")
        r2 = self._make_revision("r2", build_id="build-bbb", created_at="2025-02-01")
        r3 = self._make_revision("r3", build_id="build-ccc", created_at="2025-03-01")
        catalog = self._make_catalog("entry-uuid", [r1, r2, r3], alias_name="my_expr")

        mock_target = MagicMock()
        mock_target.entry_id = "entry-uuid"
        mock_target.rev = "r3"

        with patch("xorq.catalog.load_catalog", return_value=catalog):
            with patch.object(Target, "from_str", return_value=mock_target):
                result = get_entry_revisions("my_expr")

        assert len(result) == 3
        assert result[0]["revision_id"] == "r1"
        assert result[1]["revision_id"] == "r2"
        assert result[2]["revision_id"] == "r3"
        assert result[2]["is_current"] is True
        assert result[0]["is_current"] is False
        assert result[1]["build_id"] == "build-bbb"

    def test_marks_correct_current_revision(self):
        from xorq.catalog import Target

        r1 = self._make_revision("r1", build_id="b1")
        r2 = self._make_revision("r2", build_id="b2")
        catalog = self._make_catalog("e1", [r1, r2], alias_name="expr", current_revision="r2")

        mock_target = MagicMock()
        mock_target.entry_id = "e1"

        with patch("xorq.catalog.load_catalog", return_value=catalog):
            with patch.object(Target, "from_str", return_value=mock_target):
                result = get_entry_revisions("expr")

        assert result[0]["is_current"] is False
        assert result[1]["is_current"] is True


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
    def test_empty_catalog_renders(self, mock_entries, mock_all, mock_session):
        resp = self.fetch("/")
        assert resp.code == 200
        body = resp.body.decode()
        assert "xorq" in body
        assert "No catalog entries" in body

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.get_all_entries")
    @patch("xorq_web.handlers.get_catalog_entries")
    def test_catalog_with_entries(self, mock_entries, mock_all, mock_session):
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
    def test_empty_runs_renders(self, mock_runs, mock_entries, mock_session):
        resp = self.fetch("/runs")
        assert resp.code == 200
        body = resp.body.decode()
        assert "No runs yet" in body

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.get_catalog_entries", return_value=[])
    @patch("xorq_web.handlers.get_all_runs_merged")
    def test_runs_with_data(self, mock_runs, mock_entries, mock_session):
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
    @patch("xorq.catalog.resolve_build_dir", return_value=None)
    @patch("xorq.catalog.load_catalog")
    def test_missing_target_returns_404(self, mock_cat, mock_resolve, mock_entries, mock_session):
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
    @patch("xorq.catalog.resolve_build_dir")
    @patch("xorq.catalog.load_catalog")
    @patch("xorq.catalog.Target.from_str")
    @patch("xorq.ibis_yaml.compiler.load_expr")
    @patch("xorq.common.utils.caching_utils.get_xorq_cache_dir", return_value="/tmp/cache")
    def test_valid_target_renders_detail(
        self,
        mock_cache,
        mock_load_expr,
        mock_target_from_str,
        mock_cat,
        mock_resolve,
        mock_entries,
        mock_revisions,
        mock_bk_session,
        mock_meta,
        mock_lineage,
        mock_session,
    ):
        import tempfile

        mock_resolved = MagicMock()
        mock_resolved.entry_id = "uuid-1"
        mock_resolved.rev = "r2"
        mock_target_from_str.return_value = mock_resolved

        mock_entry = MagicMock()
        mock_entry.entry_id = "uuid-1"
        mock_entry.maybe_get_revision.return_value = None
        mock_cat.return_value.maybe_get_entry.return_value = mock_entry

        with tempfile.TemporaryDirectory() as td:
            build_dir = Path(td) / "abc123"
            build_dir.mkdir()
            mock_resolve.return_value = build_dir
            mock_entries.return_value = [
                {
                    "display_name": "my_expr",
                    "aliases": ["my_expr"],
                    "entry_id": "uuid-1",
                    "revision": "r2",
                    "build_id": "abc123",
                    "created_at": "2025-06-01",
                },
            ]
            mock_expr = MagicMock()
            mock_load_expr.return_value = mock_expr

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
    @patch("xorq.catalog.resolve_build_dir")
    @patch("xorq.catalog.load_catalog")
    @patch("xorq.catalog.Target.from_str")
    @patch("xorq.ibis_yaml.compiler.load_expr")
    @patch("xorq.common.utils.caching_utils.get_xorq_cache_dir", return_value="/tmp/cache")
    def test_revision_nav_renders_prev_next(
        self,
        mock_cache,
        mock_load_expr,
        mock_target_from_str,
        mock_cat,
        mock_resolve,
        mock_entries,
        mock_revisions,
        mock_bk_session,
        mock_meta,
        mock_lineage,
        mock_session,
    ):
        import tempfile

        # Simulate viewing r2 of 3 revisions
        mock_resolved = MagicMock()
        mock_resolved.entry_id = "uuid-1"
        mock_resolved.rev = "r2"
        mock_target_from_str.return_value = mock_resolved

        mock_rev_obj = MagicMock()
        mock_rev_obj.created_at = "2025-06-01"
        mock_rev_obj.build = MagicMock()
        mock_rev_obj.build.build_id = "build-r2"
        mock_entry = MagicMock()
        mock_entry.entry_id = "uuid-1"
        mock_entry.maybe_get_revision.return_value = mock_rev_obj
        mock_cat.return_value.maybe_get_entry.return_value = mock_entry

        mock_revisions.return_value = [
            {
                "revision_id": "r1",
                "build_id": "build-r1",
                "created_at": "2025-01-01",
                "is_current": False,
            },
            {
                "revision_id": "r2",
                "build_id": "build-r2",
                "created_at": "2025-06-01",
                "is_current": False,
            },
            {
                "revision_id": "r3",
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
                "revision": "r3",
                "build_id": "build-r3",
                "created_at": "2025-09-01",
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            build_dir = Path(td) / "build-r2"
            build_dir.mkdir()
            mock_resolve.return_value = build_dir
            mock_load_expr.return_value = MagicMock()

            resp = self.fetch("/entry/my_expr@r2")
            assert resp.code == 200
            body = resp.body.decode()

            # Should show revision nav with prev (r1) and next (r3)
            assert "my_expr@r1" in body
            assert "my_expr@r2" in body
            assert "my_expr@r3" in body
            # Prev arrow should link to r1
            assert "/entry/my_expr@r1" in body
            # Next arrow should link to r3
            assert "/entry/my_expr@r3" in body

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.load_lineage_html", return_value={})
    @patch("xorq_web.handlers.load_build_metadata", return_value={})
    @patch("xorq_web.handlers.ensure_buckaroo_session", return_value={"session": "s1"})
    @patch("xorq_web.handlers.get_entry_revisions")
    @patch("xorq_web.handlers.get_catalog_entries")
    @patch("xorq.catalog.resolve_build_dir")
    @patch("xorq.catalog.load_catalog")
    @patch("xorq.catalog.Target.from_str")
    @patch("xorq.ibis_yaml.compiler.load_expr")
    @patch("xorq.common.utils.caching_utils.get_xorq_cache_dir", return_value="/tmp/cache")
    def test_no_revision_nav_for_single_revision(
        self,
        mock_cache,
        mock_load_expr,
        mock_target_from_str,
        mock_cat,
        mock_resolve,
        mock_entries,
        mock_revisions,
        mock_bk_session,
        mock_meta,
        mock_lineage,
        mock_session,
    ):
        import tempfile

        mock_resolved = MagicMock()
        mock_resolved.entry_id = "uuid-1"
        mock_resolved.rev = "r1"
        mock_target_from_str.return_value = mock_resolved

        mock_entry = MagicMock()
        mock_entry.entry_id = "uuid-1"
        mock_entry.maybe_get_revision.return_value = None
        mock_cat.return_value.maybe_get_entry.return_value = mock_entry

        mock_revisions.return_value = [
            {
                "revision_id": "r1",
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
                "revision": "r1",
                "build_id": "build-1",
                "created_at": "2025-01-01",
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            build_dir = Path(td) / "build-1"
            build_dir.mkdir()
            mock_resolve.return_value = build_dir
            mock_load_expr.return_value = MagicMock()

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
    @patch("xorq.catalog.resolve_build_dir")
    @patch("xorq.catalog.load_catalog")
    @patch("xorq.catalog.Target.from_str")
    @patch("xorq.ibis_yaml.compiler.load_expr")
    @patch("xorq.common.utils.caching_utils.get_xorq_cache_dir", return_value="/tmp/cache")
    def test_prompt_metadata_displayed(
        self,
        mock_cache,
        mock_load_expr,
        mock_target_from_str,
        mock_cat,
        mock_resolve,
        mock_entries,
        mock_revisions,
        mock_bk_session,
        mock_meta,
        mock_lineage,
        mock_session,
    ):
        import tempfile

        mock_resolved = MagicMock()
        mock_resolved.entry_id = "uuid-1"
        mock_resolved.rev = "r2"
        mock_target_from_str.return_value = mock_resolved

        mock_rev_obj = MagicMock()
        mock_rev_obj.created_at = "2025-06-01"
        mock_rev_obj.build = MagicMock()
        mock_rev_obj.build.build_id = "build-r2"
        mock_rev_obj.metadata = {"prompt": "expand the number of rows to 20"}
        mock_entry = MagicMock()
        mock_entry.entry_id = "uuid-1"
        mock_entry.maybe_get_revision.return_value = mock_rev_obj
        mock_cat.return_value.maybe_get_entry.return_value = mock_entry

        mock_entries.return_value = [
            {
                "display_name": "my_expr",
                "aliases": ["my_expr"],
                "entry_id": "uuid-1",
                "revision": "r2",
                "build_id": "build-r2",
                "created_at": "2025-06-01",
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            build_dir = Path(td) / "build-r2"
            build_dir.mkdir()
            mock_resolve.return_value = build_dir
            mock_load_expr.return_value = MagicMock()

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
    @patch("xorq.catalog.resolve_build_dir")
    @patch("xorq.catalog.load_catalog")
    @patch("xorq.catalog.Target.from_str")
    @patch("xorq.ibis_yaml.compiler.load_expr")
    @patch("xorq.common.utils.caching_utils.get_xorq_cache_dir", return_value="/tmp/cache")
    def test_no_prompt_block_when_no_metadata(
        self,
        mock_cache,
        mock_load_expr,
        mock_target_from_str,
        mock_cat,
        mock_resolve,
        mock_entries,
        mock_revisions,
        mock_bk_session,
        mock_meta,
        mock_lineage,
        mock_session,
    ):
        import tempfile

        mock_resolved = MagicMock()
        mock_resolved.entry_id = "uuid-1"
        mock_resolved.rev = "r1"
        mock_target_from_str.return_value = mock_resolved

        mock_entry = MagicMock()
        mock_entry.entry_id = "uuid-1"
        mock_entry.maybe_get_revision.return_value = None
        mock_cat.return_value.maybe_get_entry.return_value = mock_entry

        mock_entries.return_value = [
            {
                "display_name": "my_expr",
                "aliases": ["my_expr"],
                "entry_id": "uuid-1",
                "revision": "r1",
                "build_id": "build-1",
                "created_at": "2025-01-01",
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            build_dir = Path(td) / "build-1"
            build_dir.mkdir()
            mock_resolve.return_value = build_dir
            mock_load_expr.return_value = MagicMock()

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
        mock_build = MagicMock()
        mock_build.build_id = "abc"

        mock_rev = MagicMock()
        mock_rev.revision_id = "r1"
        mock_rev.build = mock_build
        mock_rev.created_at = None

        mock_entry = MagicMock()
        mock_entry.entry_id = "entry-1"
        mock_entry.current_revision = "r1"
        mock_entry.history = [mock_rev]

        mock_catalog = MagicMock()
        mock_catalog.entries = [mock_entry]
        mock_catalog.aliases = {}

        with patch("xorq.catalog.load_catalog", return_value=mock_catalog):
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
    @patch("xorq_web.session_store.get_session_entry", return_value=None)
    def test_missing_session_returns_404(self, mock_entry, mock_entries, mock_session):
        resp = self.fetch("/session/nonexistent")
        assert resp.code == 404

    @patch("xorq_web.handlers.get_session_entries", return_value=[])
    @patch("xorq_web.handlers.load_lineage_html", return_value={})
    @patch("xorq_web.handlers.load_build_metadata", return_value={})
    @patch("xorq_web.handlers.ensure_buckaroo_session", return_value={"session": "abc"})
    @patch("xorq_web.handlers.get_catalog_entries", return_value=[])
    @patch("xorq.ibis_yaml.compiler.load_expr")
    @patch("xorq.common.utils.caching_utils.get_xorq_cache_dir", return_value="/tmp/cache")
    def test_valid_session_renders(
        self,
        mock_cache,
        mock_load_expr,
        mock_entries,
        mock_bk_session,
        mock_meta,
        mock_lineage,
        mock_session,
    ):
        import tempfile

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
    def test_catalog_index_ignores_query_param(self, mock_entries, mock_all, mock_session):
        resp = self.fetch("/?s=abc123")
        assert resp.code == 200
