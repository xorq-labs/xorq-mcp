"""Basic tests for xorq MCP tools."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def results_dir(tmp_path):
    """Provide a temporary results directory."""
    return tmp_path / "results"


class TestXorqRun:
    """Tests for the xorq_run tool."""

    def test_missing_script(self):
        from xorq_mcp_tool import xorq_run

        result = xorq_run("/nonexistent/script.py")
        assert "Error: file not found" in result

    def test_missing_expr_variable(self, tmp_path):
        from xorq_mcp_tool import xorq_run

        script = tmp_path / "empty_script.py"
        script.write_text("x = 42\n")
        result = xorq_run(str(script), expr_name="expr")
        assert "variable 'expr' not found" in result
        assert "x" in result  # should list available names


class TestViewData:
    """Tests for the view_data tool."""

    @patch("xorq_mcp_tool._view_impl")
    def test_delegates_to_view_impl(self, mock_view):
        from xorq_mcp_tool import view_data

        mock_view.return_value = "summary"
        result = view_data("/some/file.parquet")
        mock_view.assert_called_once_with("/some/file.parquet")
        assert result == "summary"


class TestCatalogLs:
    """Tests for the xorq_catalog_ls tool."""

    def test_empty_catalog(self):
        from xorq.catalog import XorqCatalog

        with patch("xorq.catalog.load_catalog", return_value=XorqCatalog()):
            from xorq_mcp_tool import xorq_catalog_ls

            result = xorq_catalog_ls()
            assert result == "Catalog is empty."


class TestDiffBuilds:
    """Tests for the xorq_diff_builds tool."""

    def test_missing_left_target(self):
        from xorq.catalog import XorqCatalog

        with patch("xorq.catalog.load_catalog", return_value=XorqCatalog()):
            from xorq_mcp_tool import xorq_diff_builds

            result = xorq_diff_builds("/nonexistent/left", "/nonexistent/right")
            assert "not found" in result


class TestLineage:
    """Tests for the xorq_lineage tool."""

    def test_missing_build_target(self):
        from xorq.catalog import XorqCatalog

        with patch("xorq.catalog.load_catalog", return_value=XorqCatalog()):
            from xorq_mcp_tool import xorq_lineage

            result = xorq_lineage("/nonexistent/build")
            assert "not found" in result


class TestFormatLineageText:
    """Tests for the _format_lineage_text helper."""

    def test_simple_node(self):
        from xorq_mcp_tool import _format_lineage_text

        # Create a mock node
        mock_op = MagicMock()
        mock_op.__class__.__name__ = "Field"
        mock_op.name = "col_a"

        mock_node = MagicMock()
        mock_node.op = mock_op
        mock_node.children = ()

        result = _format_lineage_text(mock_node)
        assert "Field: col_a" in result

    def test_nested_nodes(self):
        from xorq_mcp_tool import _format_lineage_text

        child_op = MagicMock()
        child_op.__class__.__name__ = "DatabaseTable"
        child_op.name = "source_table"
        child_node = MagicMock()
        child_node.op = child_op
        child_node.children = ()

        parent_op = MagicMock()
        parent_op.__class__.__name__ = "Field"
        parent_op.name = "col_a"
        parent_node = MagicMock()
        parent_node.op = parent_op
        parent_node.children = (child_node,)

        result = _format_lineage_text(parent_node)
        assert "Field: col_a" in result
        assert "DatabaseTable: source_table" in result
        # Child should be indented more than parent
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert lines[1].startswith("  ")
