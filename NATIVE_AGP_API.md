# Persistent native AGP API

## Why this interface exists

The readable Python propagation implementation performs deterministic power
iteration and is useful as a baseline. The original paper backend calls a C++
executable for each question, which repeatedly exports and reloads the graph.
`NativeAGPBackend` instead keeps one authors' `Graph` object alive inside the
Python process and calls the paper algorithm through a small C-compatible API.

```text
AGPPipeline
    -> NativeAGPBackend.propagate()
        -> Python ctypes
            -> libagp_api
                -> authors' Graph::query()
```

## Build

The authors' checkout remains external and unchanged:

```bash
bash scripts/build_agp_library.sh \
  /Users/feiyuzhang/Desktop/COMP90055/AGP-dynamic
```

The build creates `build/paper_backend/libagp_api.dylib` on macOS or
`libagp_api.so` on Linux. It compiles a temporary copy of `Graph.cpp` with two
documented compatibility changes:

1. convert five 1-based neighbor IDs to 0-based score-array indexes;
2. move AGP-Static++ sampling-structure construction out of each query so the
   persistent handle prepares it only once.

## C API lifecycle

`paper_backend/agp_api.h` exports four lifecycle operations:

1. `agp_create` builds the graph and optionally prepares AGP-Static++;
2. `agp_query` accepts seed IDs, seed masses, weights, epsilon, and query type;
3. `agp_vertex_count` reports the output-vector size;
4. `agp_destroy` releases the persistent handle.

Errors are returned as status codes and explained by `agp_last_error`. C++
exceptions do not cross the C boundary.

## Python lifecycle

`NativeAGPBackend` loads the shared library with `ctypes`. The first call to
`propagate` maps application node IDs to contiguous 1-based C++ IDs, removes
self-loops and duplicate edges, and creates the native handle. Later calls reuse
that handle. `close()` releases it; the backend also supports a `with` block.

One backend instance is deliberately tied to one `KnowledgeGraph`. Attempting to
reuse it with another graph raises an error instead of silently querying stale
native state.

## Run

```bash
python3 -m agp_research \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --backend native \
  --paper-query-type S \
  ask "Which pages directly neighbor Census Australia" \
  --strategy rules \
  --no-answer
```

Use `--paper-query-type N` for the authors' exact query and `S` for
AGP-Static++. The existing `--paper-a`, `--paper-b`, `--paper-delta`, and
`--paper-relative-error` options configure both paper backends.

## Verification completed

- native exact scores match the independent one-shot bridge on a small graph;
- exact and approximate modes both reuse the same handle across two queries;
- all 49 unit tests pass with the library built;
- an end-to-end Facebook CLI query returns valid ranked nodes;
- three Facebook approximate queries reused one handle. Setup plus the first
  query took about 0.183 seconds; subsequent queries took about 0.0015 and
  0.0014 seconds on the development machine.

The timings only confirm graph reuse. They are not a controlled benchmark and
should not be presented as a general speedup measurement.

## Current limitations

- The authors' implementation supports unweighted, undirected graphs here.
- AGP-Static++ rejects isolated vertices because its degree buckets require
  positive degree.
- Approximate queries are randomized by the upstream implementation.
- The wrapper currently exposes static queries, not dynamic insert/delete calls.
- The upstream container types do not completely release all internal allocation
  buffers when a handle is destroyed. This is not a per-query leak, but repeated
  create/destroy cycles in one long process should be avoided until upstream
  ownership is repaired.
- A controlled scaling experiment must compare load time, repeated-query time,
  memory, and approximation error before performance conclusions are made.
