"""Tests for the xorq_web server package."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tornado.testing
import tornado.web

from xorq_web.app import make_app
from xorq_web.metadata import (
    _render_node_html,
    get_catalog_entries,
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

    @patch("xorq_web.handlers.get_catalog_entries", return_value=[])
    def test_empty_catalog_renders(self, mock_entries):
        resp = self.fetch("/")
        assert resp.code == 200
        body = resp.body.decode()
        assert "xorq" in body
        assert "No catalog entries" in body

    @patch("xorq_web.handlers.get_catalog_entries")
    def test_catalog_with_entries(self, mock_entries):
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
        resp = self.fetch("/")
        assert resp.code == 200
        body = resp.body.decode()
        assert "test_expr" in body
        assert "/entry/test_expr" in body


# -----------------------------------------------------------------------
# handlers.py — ExpressionDetailHandler (via Tornado test client)
# -----------------------------------------------------------------------
class TestExpressionDetailHandler(tornado.testing.AsyncHTTPTestCase):
    def get_app(self):
        return make_app(buckaroo_port=8455)

    @patch("xorq_web.handlers.get_catalog_entries", return_value=[])
    @patch("xorq.catalog.resolve_build_dir", return_value=None)
    @patch("xorq.catalog.load_catalog")
    def test_missing_target_returns_404(self, mock_cat, mock_resolve, mock_entries):
        resp = self.fetch("/entry/nonexistent")
        assert resp.code == 404

    @patch("xorq_web.handlers.load_lineage_html", return_value={})
    @patch("xorq_web.handlers.load_build_schema", return_value=[("id", "int64")])
    @patch("xorq_web.handlers.load_build_metadata", return_value={"current_library_version": "0.3.7"})
    @patch("xorq_web.handlers.ensure_buckaroo_session", return_value={"session": "abc"})
    @patch("xorq_web.handlers.get_catalog_entries")
    @patch("xorq.catalog.resolve_build_dir")
    @patch("xorq.catalog.load_catalog")
    @patch("xorq.ibis_yaml.compiler.load_expr")
    @patch("xorq.common.utils.caching_utils.get_xorq_cache_dir", return_value="/tmp/cache")
    def test_valid_target_renders_detail(
        self,
        mock_cache,
        mock_load_expr,
        mock_cat,
        mock_resolve,
        mock_entries,
        mock_bk_session,
        mock_meta,
        mock_schema,
        mock_lineage,
    ):
        import tempfile

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
            assert "int64" in body


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
