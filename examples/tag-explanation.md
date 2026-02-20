## From Static Graphs to Parameterized Artifacts

### The Problem

Traditional ML workflows are scripts. You run a Python file, it trains a model, pickles it, and you hope you remembered which data, which code version, and which dependencies produced it. Reproducing or redeploying means re-running code and managing serialized model files by hand.

### The Static Graph: Useful but Limited

xorq represents your pipeline as a lazy expression graph. Every node — data sources, transformations, the fitted model, test evaluation — gets a deterministic, input-addressed hash. You run `xorq build` and the graph is compiled into a YAML manifest: a portable, versioned artifact that captures the full pipeline logic, dependencies, and lineage.

This is already an improvement. You can trace how any output was produced by walking the graph. You can reproduce a result exactly by re-running the manifest. But it's a **snapshot in time** — the training data, the model, everything is frozen.

### Tags Make the Graph Parameterizable

A **Tag node** is a metadata annotation you attach to any point in the expression graph:

```python
train_data = raw_data.filter(...).tag("train")
live_input = incoming_data.tag("live_data")
```

Tags don't affect the data or the hash. They're preserved in the YAML manifest as named markers — **substitution points** where new data can be bound at runtime, without editing source code.

This turns the manifest from a **record** of a computation into a **reusable function** with named parameters.

### Retraining Without Changing Code

Tag the training data node. Later, retrain on new data:

```bash
xorq run-unbound builds/<manifest-hash> \
  --to-unbind-tag "train" < new_training_data.arrow
```

Under the hood:

1. The manifest is loaded — full expression graph with the fitted pipeline
2. `expr_to_unbound()` finds the node tagged `"train"` and replaces it with a placeholder (`UnboundTable`)
3. Downstream cache nodes are stripped (their keys depended on the old data)
4. New data is read from stdin (Arrow IPC) and bound into the placeholder
5. The entire graph re-executes — `fit` runs again, producing a new model

Same manifest. No code changes. Different data in, different model out.

### Serving Predictions on Live Data

Tag the inference input node. In production, either run as a batch:

```bash
xorq run-unbound builds/<manifest-hash> \
  --to-unbind-tag "live_data" < todays_batch.arrow
```

Or expose it as a persistent service:

```bash
xorq serve-unbound builds/<manifest-hash> \
  --to-unbind-tag "live_data" --port 8080
```

This starts an Arrow Flight server. Clients connect, push new data via Flight RPC, and get predictions back. The manifest's expression graph runs with whatever data the client sends.

### What Changes

| Without tags | With tags |
|---|---|
| Frozen artifact — reproduces one specific run | Parameterized artifact — a template for runs |
| "Here's what happened" | "Here's what can happen, given new inputs" |
| Lineage tool | Deployment artifact |

The expression graph is the function body. The tagged nodes are the arguments.
