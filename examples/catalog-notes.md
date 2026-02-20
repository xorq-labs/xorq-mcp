# xorq Catalog — Working Understanding

This is an intermediate work product. The goal is to capture my current understanding of the catalog, verify it against the codebase, and surface questions for Hussain and Dan.

## Core Concept: A Lockfile for Data Pipelines

The catalog (`catalog.yaml`) pins the resolved state of data pipeline expressions — schemas, transforms, and their hashes — the way `uv.lock` pins package versions. It gives human-readable **aliases** to specific runs (revisions) of xorq expressions, so you can refer to `"LTV"` instead of a UUID + revision hash.

In the code, the `XorqCatalog` stores `Entry` objects (identified by UUID), each with a `history` of `Revision` objects (sequentially numbered `r1`, `r2`, ...). Each `Revision` records `node_hashes`, `expr_hashes`, `schema_fingerprint`, a `meta_digest`, and a `Build` reference pointing to the serialized YAML manifest on disk. Aliases are a `name -> (entry_id, revision_id)` mapping.

## How It Deals with Changing Data

Three time points illustrate how input-addressed caching interacts with the catalog:

- At **t1**: expression is run, cache is written, hash is `h1`
- At **t2**: source data unchanged — cache hit, same hash `h1`, no recompute
- At **t3**: source data changed — hash becomes `h2`, cache miss, recompute

**What gets hashed by default:**
- The **schema** of each table/input (column names + types)
- The **expression nodes** themselves (the transform graph)
- **Optionally**, the last modification time of the source

There are two cache strategies in the codebase:
- `SnapshotStrategy` — hashes schema + expression nodes only (name, schema, source, namespace). This is what the YAML compiler uses.
- `ModificationTimeStrategy` — also includes modification timestamps. For file reads, it uses `mtime`/`Last-Modified`/`e_tag`. For Snowflake, it calls `get_snowflake_last_modification_time`. For Postgres, it hashes schema + source + namespace (no mtime by default).

**Question:** Which strategy does the catalog default to? `SnapshotStrategy` is used for YAML builds, but `ModificationTimeStrategy` captures the t1/t2/t3 story better. Is the catalog meant to record which strategy was used?

## Incremental Processing

xorq does not do incremental processing out of the box. Other engines might; xorq isn't solving that problem.

You *could* write expressions that give different answers when run at different times (e.g., "last month's sales"), grab the hash for a particular run, and reference that hash. But this is a bad idea because you can't reproduce it — "last month's sales" gives you January in February, but there's no way to get back to that result after February. This is particularly bad when combined with aliases, because the alias would point to a result that can't be reproduced.

The catalog isn't meant for storing expressions and caches like that. Those are poor designs of your data architecture.

**Question:** Is there a recommended pattern? E.g., should expressions that are time-windowed always be parameterized with explicit date ranges rather than relative ones?

## Comparing Results After a Package Upgrade

If you run expressions with package-set-1, then again with package-set-2 (e.g., bump pandas), the **build name** (build_id — the directory name) changes between runs, but the **expression entry** stays the same. So you get two revisions under the same entry and can diff them.

The CLI supports this: `xorq catalog diff-builds` lets you diff two builds by their catalog targets (alias, entry_id, or `entry@revision` syntax). It compares the serialized YAML expressions, SQL, profiles, and metadata using `git diff --no-index`.

## Sharing: Catalog as Distribution Mechanism

Two modes for a data scientist sharing results with a business analyst:

1. **Ship the cache** — export the catalog + cached data (`xorq catalog export` copies `catalog.yaml` + `catalog-builds/` to a target directory)
2. **Serve via Flight** — reference a cached node/hash via an Arrow Flight endpoint (the `ServerRecord` class tracks running Flight servers and their associated `node_hash` / target)

## Aliases

Aliases live in the `aliases` section of `catalog.yaml`. They map a human name to `(entry_id, revision_id)`. The `Target.from_str` resolver handles `"LTV"`, `"LTV@r2"`, or raw `entry_id@revision` formats.

## Swapping a Pipeline Step

Use case: a pipeline is `step1 -> step2 (random forest) -> step3 (join)`. You want to swap step2 to linear regression without recomputing step1 or step3.

**Question:** How does this work mechanically? I see how the catalog can reference specific node hashes, but is the workflow: (1) run the original pipeline, (2) note the cache hash for step1's output, (3) build a new expression that reads from that cached hash as its input, then runs linear regression + step3? Or is there a more first-class way to do this?

## Git-Based Collaboration

The catalog is sequentially versioned and stored in a shared git repo between collaborators. Facts (hashes) are append-only — revisions are numbered `r1`, `r2`, ... and appended to `entry.history`. The catalog is a single YAML file with no deletion of history, just appending.

## CLI Surface

- `xorq catalog ls` — list entries and aliases
- `xorq catalog add <build_path> [--alias NAME]` — add a build to the catalog
- `xorq catalog info` — show catalog path, entry count, alias count
- `xorq catalog rm <entry>` — remove entry or alias
- `xorq catalog export <output_path>` — export catalog + builds to a directory
- `xorq catalog diff-builds <left> <right>` — diff two builds
- `xorq lineage <target>` — print per-column lineage trees for a build

## Replay

**Question:** Why would you want to replay a computation if you'll get the same result? Possible reasons:
- **Verification/audit** — prove you get the same result from the same inputs
- **Reproducing on a different machine** — the manifest captures the full expression graph
- **After a cache eviction** — data is gone from cache but the catalog still knows what the expression was
- **Package upgrade scenario** — replay with different deps to compare

Is replay just `xorq run <manifest>`? Or is there a distinction between "re-execute" and "replay" (e.g., replaying from cached intermediates)?

---

## Deep Dive: Alias Resolution and Git Merge Safety

`Target.from_str` resolves aliases **at the moment you invoke it** — it reads the current `catalog.yaml`, looks up the alias, and returns the `(entry_id, revision_id)` it points to right then. This happens in CLI commands like `xorq catalog diff-builds`, `xorq lineage`, and `xorq run`.

But **at build time** (`xorq build`), the expression graph is serialized into a YAML manifest with concrete node hashes. The alias isn't baked into the manifest — the resolved hashes are. So the manifest is self-contained.

This means there are two different moments where "what does LTV mean?" matters:

### Build time (early-bound, safe)

When you `xorq build` a pipeline that uses the output of LTV as an input, the expression graph already contains the concrete cached node hash. The alias isn't in the manifest. So user 1 can update the LTV alias after user 2 built PE, and PE's manifest is unaffected — it locked in the specific hash at build time.

### CLI / catalog operations (late-bound, potentially surprising)

If user 2 runs `xorq run PE` and PE is an alias, that resolves to whatever `PE` currently points at. That's fine. But if user 2 runs something like `xorq catalog diff-builds LTV@r1 LTV@r2` after user 1 moved the LTV alias to a completely different entry_id, they'd be comparing apples to oranges without realizing it.

### The git merge question

In the YAML, aliases is a flat map:

```yaml
aliases:
  LTV:
    entry_id: abc-123
    revision_id: r3
  PE:
    entry_id: def-456
    revision_id: r2
```

If user 1 changes `LTV.revision_id` from `r2` to `r3`, and user 2 changes `PE.revision_id` from `r1` to `r2`, git will auto-merge that cleanly — they touched different keys. No textual conflict.

But the **semantic** problem is real: PE was built against LTV@r2. Now LTV points at r3. The catalog doesn't encode that dependency. PE's alias still points at its own entry, blissfully unaware that the LTV it was built from has moved.

### Where does dependency tracking live?

The catalog tracks entries and aliases independently — it's a flat namespace, not a dependency graph. The dependency information lives in the serialized expression (the manifest YAML), not in the catalog. The manifest contains the concrete hashes that were resolved at build time.

**Key question for Hussain/Dan:** Is the catalog meant to track inter-expression dependencies, or is that the manifest's job? If the catalog is purely a flat registry of builds + aliases, then "no merge conflicts on hashes" is true, and alias coherence is the user's responsibility. But if the catalog is supposed to provide a consistent view of a pipeline's dependency graph, then there's a gap — the catalog doesn't know that PE depends on LTV.

A related question: **when does alias substitution happen in practice?** If it's always at build time (early-bound into the manifest), then the alias is just a convenience for humans and the real contract is the hash. If it ever happens at run time (late-bound), then alias consistency across collaborators becomes a correctness concern, not just a UX concern.
