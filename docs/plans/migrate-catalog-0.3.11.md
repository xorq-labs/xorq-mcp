# Migrate xorq-mcp to xorq 0.3.11 Catalog API

**Issue:** https://github.com/buckaroo-data/xorq-mcp/issues/2

## Problem

xorq 0.3.11 replaced the flat-file catalog (`load_catalog()`, `resolve_build_dir()`, `Target`, `XorqCatalog`, `CatalogPaths`, etc.) with a git-backed `Catalog` class. The MCP server imports the old functions in 6+ locations and will crash on startup once upgraded.

Additionally, the `ExpressionDetailHandler` takes 44-64s to load complex expressions like `good_deal_by_position` because expression execution and lineage computation block the page render.

## API Migration Map

| Old (0.3.10) | New (0.3.11) |
|---|---|
| `load_catalog()` | `Catalog.from_default()` |
| `catalog.entries` | `catalog.catalog_entries` (tuple of `CatalogEntry`) |
| `catalog.aliases` | `catalog.catalog_aliases` (tuple of `CatalogAlias`) |
| `resolve_build_dir(target, catalog)` → Path | `catalog.get_catalog_entry(name)` → `CatalogEntry`, then `extract_build_tgz_context(entry.catalog_path)` |
| `Target.from_str(target, catalog)` | Manual resolution: `catalog.contains(name)` / iterate `catalog_aliases` |
| `XorqCatalog()` (empty mock) | `MagicMock` with empty `.catalog_entries` |
| `CatalogPaths.create()`, `do_save_catalog()` | `catalog.add(build_dir_path, aliases=(alias,))` |
| `entry.entry_id` | `entry.name` |
| `entry.current_revision`, `entry.history` | `CatalogAlias.list_revisions()` for git-based revision history |
| `rev.build.build_id`, `rev.metadata` | Metadata in `entry.metadata_path` (.metadata.yaml); build_id = `entry.name` |

## Performance Profile (0.3.11 catalog)

| Operation | Time | Cached? |
|---|---|---|
| `Catalog.from_default()` | 49ms | N/A |
| `catalog_entries` (11 entries) | 6ms | **No** — re-scans each access |
| `catalog_aliases` (11 aliases) | 7ms | **No** — re-scans each access |
| `contains()` / `get_catalog_entry()` | 1ms | N/A |
| `list_revisions()` | 19ms | N/A |
| `extract_build_tgz_context()` | 8ms | N/A |
| `load_expr_from_tgz()` | 19ms | N/A |

Catalog ops are fast individually. The 44-64s page loads on `good_deal_by_position` come from `to_parquet()` (expression execution) and `build_column_trees()` (lineage), not catalog git overhead.

---

## Design Decisions

### 1. Request-scoped `CatalogSnapshot`

`catalog_entries` and `catalog_aliases` are not cached — every property access re-scans the git repo. `CatalogSnapshot` reads both once at the start of a request and exposes the same interface. All functions in the request path accept an optional `snapshot` param. Created once per handler invocation, never stored globally, so it always reflects current catalog state.

### 2. Context-manager lifetime safety for extracted build dirs

`extract_build_tgz_context(entry.catalog_path)` yields a temp directory that is deleted when the context exits. In `ExpressionDetailHandler`, both `_load_execute_and_register()` and `load_lineage_html()` must run **inside** the `with` block.

For `xorq_diff_builds()` which needs two build dirs simultaneously, `_resolve_to_build_dir()` manually enters the context and returns a cleanup callback. Cleanup runs in `finally`.

For simpler tools (`xorq_view_build`, `xorq_lineage`, `xorq_catalog_info`), prefer `load_expr_from_tgz()` which handles extraction internally.

### 3. Alias-resolution determinism

When multiple aliases point to the same entry, `display_name` uses `sorted(aliases)[0]` (alphabetically first). `resolve_target()` checks entry names first, then aliases in snapshot-frozen order. If no match, returns `None` — callers surface a clear "target not found" error (no silent fallback to a wrong entry).

### 4. Metadata resilience

`.metadata.yaml` becomes the source of truth for per-entry metadata. `_read_entry_metadata()` returns `{}` on missing or malformed files (try/except with fallback). `_write_entry_metadata()` does read-merge-write with `yaml.safe_load` / `yaml.dump`. No data loss on partial writes since the file is small and atomic enough for single-process access.

### 5. Phased rollout — 3 PRs instead of 1

Each PR passes all tests independently so regressions are bisectable.

---

## Implementation Plan

### PR 1: Core helpers + catalog reading (`catalog_utils.py`, `metadata.py`)

**Step 1a — `CatalogSnapshot` class** (`catalog_utils.py`)

```python
class CatalogSnapshot:
    def __init__(self, catalog=None):
        if catalog is None:
            catalog = Catalog.from_default()
        self._entries = catalog.catalog_entries      # read once
        self._aliases = catalog.catalog_aliases      # read once
        self._alias_lookup = {}                      # entry_name -> sorted list of alias strings
        for ca in self._aliases:
            self._alias_lookup.setdefault(ca.catalog_entry.name, []).append(ca.alias)
        for k in self._alias_lookup:
            self._alias_lookup[k].sort()             # deterministic ordering

    def contains(name) -> bool: ...
    def get_catalog_entry(name) -> CatalogEntry | None: ...
    def aliases_for(entry_name) -> list[str]: ...
```

**Step 1b — `resolve_target()` helper** (`catalog_utils.py`)

Replaces `Target.from_str()`. Accepts `Catalog | CatalogSnapshot`. Checks entry names first, then aliases. Returns `None` on no match (not a silent wrong match).

**Step 1c — `_read_entry_metadata()` / `_write_entry_metadata()`** (`catalog_utils.py`)

Read/write `.metadata.yaml` on a `CatalogEntry`. `_read_entry_metadata` returns `{}` on missing/malformed files. `_write_entry_metadata` does read-merge-write.

**Step 1d — Rewrite `register_in_catalog()`** (`catalog_utils.py`)

`Catalog.from_default()` → `catalog.add(Path(build_path), aliases=(alias,))` → write `.metadata.yaml`.

**Step 1e — Rewrite `update_revision_metadata()`** (`catalog_utils.py`)

`Catalog.from_default()` → `get_catalog_entry()` → read/merge/write `.metadata.yaml`.

**Step 1f — Update `get_all_runs(snapshot=None)`** (`metadata.py`)

Accept optional snapshot. Use `snapshot.aliases_for()`. Read metadata from `_read_entry_metadata()`.

**Step 1g — Update `get_entry_revisions(target, snapshot=None)`** (`metadata.py`)

Replace `Target.from_str()` with `resolve_target()`. Use `CatalogAlias.list_revisions()` for history. Fallback to single-item list.

**Step 1h — Update `get_catalog_entries(snapshot=None)`** (`metadata.py`)

Same pattern: snapshot, `aliases_for()`, `_read_entry_metadata()`.

**PR 1 acceptance criteria:**
- [ ] `CatalogSnapshot`, `resolve_target`, metadata read/write helpers implemented
- [ ] `register_in_catalog()` and `update_revision_metadata()` migrated
- [ ] `get_all_runs`, `get_entry_revisions`, `get_catalog_entries` accept optional snapshot and pass tests
- [ ] `resolve_target` tested for: entry name match, alias match, no match returns `None`, ambiguous alias returns first match

### PR 2: Handlers + MCP tools (`handlers.py`, `xorq_mcp_tool.py`)

**Step 2a — `ExpressionDetailHandler.get()`** (`handlers.py`)

- Create `CatalogSnapshot()` once, pass to all catalog functions
- `resolve_target(base_name, snapshot)` replaces old trio
- `extract_build_tgz_context(entry.catalog_path)` wraps both executor calls:
  ```python
  with extract_build_tgz_context(entry.catalog_path) as build_dir:
      metadata = load_build_metadata(build_dir)
      await loop.run_in_executor(None, partial(_load_execute_and_register, build_dir, ...))
      lineage = await loop.run_in_executor(None, partial(load_lineage_html, build_dir))
  ```

**Step 2b — `CatalogIndexHandler`, `RunsHandler`** (`handlers.py`)

Create snapshot once, pass through to `get_catalog_entries()` / `get_all_runs_merged()`.

**Step 2c-g — MCP tools** (`xorq_mcp_tool.py`)

- `xorq_view_build`: `resolve_target()` + `load_expr_from_tgz()` (no manual context)
- `xorq_catalog_ls`: `Catalog.from_default()` → iterate `catalog_entries`
- `xorq_catalog_info`: `resolve_target()` + `_read_entry_metadata()` + `load_expr_from_tgz()` + `list_revisions()`
- `xorq_lineage`: `resolve_target()` + `load_expr_from_tgz()`
- `xorq_diff_builds`: `_resolve_to_build_dir()` with manual context + `finally` cleanup

**PR 2 acceptance criteria:**
- [ ] Handlers instantiate one snapshot per request and thread it through
- [ ] `ExpressionDetailHandler` runs all build-dir work inside extraction context
- [ ] All MCP tools migrated with proper cleanup for diff flow
- [ ] No imports of removed 0.3.10 APIs remain in handlers/tools

### PR 3: Test updates (`tests/test_tools.py`, `tests/test_web.py`)

**Step 3a — Test helpers**

`_mock_empty_catalog()` and `_mock_catalog_entry(name, aliases=None)` for consistent test setup.

**Step 3b — `test_tools.py`**

Replace `XorqCatalog()` / `load_catalog` with `Catalog.from_default()` mocks.

**Step 3c — `test_web.py`**

- Metadata tests: unchanged
- Catalog reading tests: mock `Catalog.from_default` + `_read_entry_metadata`
- Handler tests: mock `resolve_target`, `_read_entry_metadata`, `extract_build_tgz_context`, `Catalog.from_default`
- Add context-manager lifetime test: assert `build_dir` exists during executor calls inside `with` block
- Add `resolve_target` parity tests: verify same resolution behavior as old `Target.from_str` for entry names, aliases, and `name@revision` syntax

**PR 3 acceptance criteria:**
- [ ] No legacy catalog mocks (`XorqCatalog`, `load_catalog`, `Target.from_str`, `resolve_build_dir`) remain
- [ ] Context-manager lifetime tested (build dir valid during executor work)
- [ ] `resolve_target` parity with `Target.from_str` covered (entry name, alias, not-found)
- [ ] Metadata resilience tested (missing file → `{}`, malformed file → `{}`)
- [ ] CI green: lint, unit, integration

---

## Verification (each PR)

```bash
python -c "from xorq_mcp_tool import mcp"                                              # smoke
.venv/bin/ruff check . && .venv/bin/ruff format --check .                               # lint
.venv/bin/pytest tests/test_web.py tests/test_tools.py -v                               # unit
.venv/bin/pytest tests/test_buckaroo_reliability.py tests/test_playwright.py -v          # integration
```

## Definition of Done

- [ ] No imports/usages of removed 0.3.10 APIs remain
- [ ] All catalog paths operate correctly on xorq 0.3.11
- [ ] Expression detail page correct and avoids avoidable catalog rescans
- [ ] Full test suite passes (lint, unit, integration)
- [ ] `resolve_target` matching semantics documented and tested for parity with old `Target.from_str`
- [ ] `.metadata.yaml` read/write is resilient to missing/malformed files

## Prior Art

Phases 1-6 were prototyped in a single pass — all 92 tests passed (61 unit + 31 integration). Code was reverted because `CatalogSnapshot` threading was incomplete. The phased PR approach lands each piece with full test coverage before moving on.
