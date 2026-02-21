# Known Bugs

## Session entries lost on MCP reconnect

**Severity:** Medium
**Status:** Unresolved

Session-local expressions (built with `catalog=False`) are stored in `~/.xorq/sessions/{pid}.json`, keyed by the MCP process PID. `get_session_entries()` treats any manifest whose PID is no longer alive as stale and deletes it.

When the MCP server restarts (e.g. `/mcp` reconnect in Claude Code), the old PID dies and all session entries are immediately garbage-collected. The web UI silently shows nothing in the Session section.

**Workaround:** Re-run all the relevant scripts after every MCP reconnect.

**Resolution ideas (not yet pursued):**
- Make session manifests PID-independent (e.g. a single `sessions.json` not keyed by PID)
- Survive reconnects by tracking a "session token" that outlives individual MCP processes
- Auto-promote long-lived sessions to the catalog after some TTL
