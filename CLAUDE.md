# xorq-mcp Development Guide

## Package Manager

Always use **uv** (not pip) for dependency management.

```bash
# Install all deps including test extras
uv pip install -e ".[test]"

# Add a new dependency
uv pip install <package>

# Sync from lockfile
uv sync
```

## Running Tests

Tests are split into two groups that **must be run separately** due to Tornado AsyncHTTPTestCase and Playwright event loop conflicts.

### Unit tests (fast, no servers needed)
```bash
.venv/bin/pytest tests/test_web.py tests/test_tools.py -v
```

### Integration + Playwright tests (starts real Buckaroo + web servers)
```bash
.venv/bin/pytest tests/test_buckaroo_reliability.py tests/test_playwright.py -v
```

### Run everything sequentially
```bash
.venv/bin/pytest tests/test_web.py tests/test_tools.py -v && \
.venv/bin/pytest tests/test_buckaroo_reliability.py tests/test_playwright.py -v
```

**Do NOT run `pytest tests/` with all test files together** — Tornado's `AsyncHTTPTestCase` conflicts with Playwright's event loop when collected in the same session.

### Test ports
- Unit tests: use Tornado's internal test ports (no real servers)
- Integration tests: Buckaroo on **8655**, xorq-web on **8656**
- Dev servers: Buckaroo on **8455**, xorq-web on **8456**

## Server Architecture

Two servers run side-by-side:
1. **Buckaroo** (port 8455) — data viewer, serves parquet via WebSocket
2. **xorq-web** (port 8456) — catalog UI, embeds Buckaroo in an iframe

The MCP tool (`xorq_mcp_tool.py`) manages both server lifecycles.

## Linting

```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
```

## Key directories

- `xorq_web/` — Tornado web server (handlers, templates, static assets)
- `tests/` — Unit tests (`test_web.py`, `test_tools.py`), integration tests (`test_buckaroo_reliability.py`), Playwright tests (`test_playwright.py`)
- `examples/` — xorq expression scripts (excluded from linting)
- `data/` — sample datasets (gitignored)
- `builds/` — build artifacts (gitignored)
