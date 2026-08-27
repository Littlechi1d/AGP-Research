# Facebook Large Page-Page dataset guide

## Provenance

- Dataset: Facebook Large Page-Page Network
- Repository: UCI Machine Learning Repository, dataset 527
- DOI: `10.24432/C50900`
- License: CC BY 4.0
- Download page: <https://archive.ics.uci.edu/dataset/527/facebook+large+page+page+network>
- Archive SHA-256: `fa2f5c4bcb806695873acaba2769c318aa41085417074ae0406fe7c450a4e0d8`

The SNAP information page no longer provides the files. The project therefore
uses the official downloadable UCI copy and records this fact for reproducibility.

## Local layout

Original files are preserved outside the project:

```text
/Users/feiyuzhang/Desktop/COMP90055/datasets/facebook_large/
├── facebook_large.zip
└── raw/
    └── facebook_large/
        ├── README.txt
        ├── citing.txt
        ├── musae_facebook_edges.csv
        ├── musae_facebook_features.json
        └── musae_facebook_target.csv
```

Project-ready files are stored at:

```text
/Users/feiyuzhang/Desktop/COMP90055/agp_research/data/facebook_large/
├── nodes.csv
├── edges.csv
└── metadata.json
```

## Source schema

`musae_facebook_target.csv` contains:

| Column | Meaning |
|---|---|
| `id` | Contiguous dataset node ID from 0 to 22469 |
| `facebook_id` | Original Facebook page ID |
| `page_name` | Display name used for entity matching |
| `page_type` | `company`, `government`, `politician`, or `tvshow` |

`musae_facebook_edges.csv` contains `id_1,id_2`. Each row represents a mutual
like relationship and is treated as an undirected, unweighted edge.

`musae_facebook_features.json` maps node IDs to feature-index lists extracted
from page descriptions. The archive does not contain a vocabulary that maps
these indices back to words, so the converter preserves the raw JSON but does
not invent textual meanings for the indices.

## Conversion decisions

The dependency-free converter is `scripts/prepare_facebook_large.py`.

It performs these transformations:

1. Prefix source node IDs with `fb_`.
2. Convert every usable relationship to weight `1.0`.
3. Remove 179 self-loops.
4. Validate that every edge endpoint exists.
5. Preserve all 22,470 nodes; the graph has no isolated nodes.
6. Resolve duplicate normalized page names. In each duplicate group, the
   highest-degree page keeps the original name (smallest ID breaks a tie); other
   titles receive ` [page ID]`.
7. Write checksums and transformation counts to `metadata.json`.

The converted graph contains 22,470 nodes and 170,823 non-self edges. There are
367 duplicate-name groups affecting 2,305 nodes, so title disambiguation is
necessary for deterministic exact matching.

## Reproduce conversion

```bash
cd /Users/feiyuzhang/Desktop/COMP90055/agp_research

python3 scripts/prepare_facebook_large.py \
  --source-dir /Users/feiyuzhang/Desktop/COMP90055/datasets/facebook_large/raw/facebook_large \
  --output-dir data/facebook_large
```

The command prints the metadata record and replaces the generated CSV outputs.
The original UCI files are never modified.

## Run one Python-backend query

```bash
cd /Users/feiyuzhang/Desktop/COMP90055/agp_research

python3 -m agp_research \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --backend python \
  ask "Which pages are connected to NASA Student Launch through the network" \
  --strategy fixed --depth 2 --decay 0.6 --top-k 5 --no-answer
```

The verified result maps `NASA Student Launch` to `fb_10` and retrieves related
NASA pages. This demonstrates software operation, not retrieval accuracy.

## Run one paper-backend query

The converted graph is unweighted and compatible with the paper adapter:

```bash
python3 -m agp_research \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --backend paper --paper-a 0 --paper-b 1 --paper-query-type S \
  ask "Which pages are connected to NASA Student Launch through the network" \
  --strategy fixed --depth 2 --decay 0.6 --top-k 5 --no-answer
```

The C++ bridge must already be compiled. On this graph, each invocation reloads
and preprocesses the graph, so it may take substantially longer than the Python
backend for a single query.

## Use the Facebook benchmark, not the original sample questions

`data/questions.json` contains labels for the original eight-node demonstration
graph. Those node IDs do not belong to this Facebook graph.

The prepared Facebook development and test sets have independently computed
topology labels. See `FACEBOOK_BENCHMARK.md` for their definitions, regeneration
command, baseline result, and experimental protocol.
