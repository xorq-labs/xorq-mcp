# Plan: xorq-mcp — Standalone MCP Server

## What & Why

New repo `xorq-mcp`. An MCP server for xorq that uses buckaroo for visualization. When an LLM writes a xorq expression and runs it, the result opens in Buckaroo's interactive table viewer.

## Repo Layout

```
xorq_mcp/
├── pyproject.toml          # hatch, deps: xorq + buckaroo[mcp]
├── xorq_mcp_tool.py        # Single-file MCP server
└── tests/
    └── test_tools.py
```

Entry point: `xorq-mcp = "xorq_mcp_tool:main"` — runnable via `uvx xorq-mcp`.

Dependencies: `xorq>=0.3.8`, `buckaroo[mcp]>=0.12.8` (provides mcp SDK, tornado server, visualization).

## xorq_mcp_tool.py

### Infrastructure (copied from buckaroo_mcp_tool.py, ~200 lines)

Buckaroo's `buckaroo_mcp_tool.py` has battle-tested infrastructure for managing a background Tornado visualization server. Copy these unchanged (adjusting log paths to `~/.xorq/logs/`):

- **Logging** — file-based, `~/.xorq/logs/mcp_tool.log`
- **`ensure_server()`** — lazily spawns `python -m buckaroo.server` on port 8700, health-checks, version-mismatch detection
- **`_view_impl(path)`** — POSTs file path to `/load`, opens browser, returns text summary (rows, columns, dtypes)
- **Process lifecycle** — `_cleanup_server()`, pipe-based watchdog, signal handlers, parent-death watcher

### Tools (new code, ~200 lines)

**7 MCP tools:**

| Tool | What it does |
|------|-------------|
| `xorq_run(script_path, expr_name="expr")` | Import script → `build_expr()` → `load_expr()` → `to_parquet()` (cache-aware) → `_view_impl()` |
| `xorq_view_build(target)` | Resolve build (path/alias/entry) → `load_expr()` → `to_parquet()` → `_view_impl()` |
| `view_data(path)` | Raw file (CSV/Parquet/etc) → `_view_impl()` |
| `xorq_catalog_ls()` | `load_catalog()` → format entries + aliases as text |
| `xorq_catalog_info(target)` | Resolve target → show revision history + schema |
| `xorq_lineage(target, column="")` | `build_column_trees()` → format as indented text |
| `xorq_diff_builds(left, right)` | `git diff --no-index` on expr.yaml files |

**`xorq_run` flow** (the primary tool):

1. `import_from_path(script_path)` — import the user's script, extract `expr` variable
2. `build_expr(expr, builds_dir)` — serialize DAG to `builds/{hash}/`
3. `load_expr(build_path, cache_dir)` — load expression back from build
4. `expr.to_parquet(output_path)` — execute (no-op if cached)
5. `_view_impl(output_path)` — display in Buckaroo, return summary

Output parquet goes to `/tmp/xorq_mcp_results/{build_hash}.parquet`.

### xorq APIs used

All from xorq's existing codebase — no new xorq code needed:

- `xorq.common.utils.import_utils.import_from_path` — script import
- `xorq.ibis_yaml.compiler.build_expr` — serialize expression DAG
- `xorq.ibis_yaml.compiler.load_expr` — load expression from build dir
- `xorq.common.utils.caching_utils.get_xorq_cache_dir` — default cache path
- `xorq.catalog.load_catalog` / `resolve_build_dir` — catalog navigation
- `xorq.common.utils.lineage_utils.build_column_trees` — column lineage

## Implementation Steps

1. Create repo + `pyproject.toml`
2. Write `xorq_mcp_tool.py`: copy buckaroo infrastructure, add 7 tools
3. Test `xorq_run` with `xorq/examples/simple_example.py`
4. Test catalog tools with a populated catalog

## Verification

- `xorq-mcp` starts, `xorq_run` builds + executes + opens Buckaroo
- Second run of same expression is fast (cache hit)
- `view_data` works for raw files
- Server auto-starts if not running
