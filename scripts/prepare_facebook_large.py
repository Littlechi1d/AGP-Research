#!/usr/bin/env python3
"""Convert the UCI Facebook Large Page-Page dataset for agp_research.

The converter is deliberately dependency-free and preserves the downloaded raw
files. It removes self-loops, assigns unit edge weights, produces unique titles,
and writes a metadata record describing every transformation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_TARGET_COLUMNS = {"id", "facebook_id", "page_name", "page_type"}
EXPECTED_EDGE_COLUMNS = {"id_1", "id_2"}


def normalize(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = required - columns
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def unique_titles(
    node_rows: list[dict[str, str]], degrees: Counter[str]
) -> tuple[dict[str, str], dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in node_rows:
        groups[normalize(row["page_name"])].append(row)

    titles: dict[str, str] = {}
    duplicate_groups = 0
    duplicated_nodes = 0
    for rows in groups.values():
        ranked = sorted(rows, key=lambda row: (-degrees[row["id"]], int(row["id"])))
        canonical = ranked[0]
        titles[canonical["id"]] = canonical["page_name"].strip()
        if len(ranked) > 1:
            duplicate_groups += 1
            duplicated_nodes += len(ranked)
            for row in ranked[1:]:
                titles[row["id"]] = f'{row["page_name"].strip()} [page {row["id"]}]'

    normalized_titles = [normalize(title) for title in titles.values()]
    if len(normalized_titles) != len(set(normalized_titles)):
        raise ValueError("title disambiguation did not produce unique titles")
    return titles, {
        "duplicate_normalized_name_groups": duplicate_groups,
        "nodes_in_duplicate_name_groups": duplicated_nodes,
        "duplicate_title_policy": (
            "Keep the highest-degree page's original name (smallest numeric ID breaks "
            "ties); append ' [page ID]' to other pages with the same normalized name."
        ),
    }


def convert(source_dir: Path, output_dir: Path) -> dict[str, object]:
    target_path = source_dir / "musae_facebook_target.csv"
    edge_path = source_dir / "musae_facebook_edges.csv"
    feature_path = source_dir / "musae_facebook_features.json"
    for path in (target_path, edge_path, feature_path):
        if not path.is_file():
            raise FileNotFoundError(f"required source file not found: {path}")

    node_rows = read_csv(target_path, EXPECTED_TARGET_COLUMNS)
    edge_rows = read_csv(edge_path, EXPECTED_EDGE_COLUMNS)
    node_ids = {row["id"] for row in node_rows}
    if len(node_ids) != len(node_rows):
        raise ValueError("node IDs are not unique")
    if any(not row["page_name"].strip() for row in node_rows):
        raise ValueError("one or more page names are empty")

    unknown = {
        endpoint
        for row in edge_rows
        for endpoint in (row["id_1"], row["id_2"])
        if endpoint not in node_ids
    }
    if unknown:
        raise ValueError(f"edges contain unknown node IDs: {sorted(unknown)[:10]}")

    self_loops = 0
    converted_edges: set[tuple[str, str]] = set()
    for row in edge_rows:
        source, target = row["id_1"], row["id_2"]
        if source == target:
            self_loops += 1
            continue
        edge = tuple(sorted((source, target), key=int))
        converted_edges.add(edge)

    degrees: Counter[str] = Counter(
        endpoint for edge in converted_edges for endpoint in edge
    )
    isolated = sorted(node_ids - set(degrees), key=int)
    titles, title_metadata = unique_titles(node_rows, degrees)

    with feature_path.open(encoding="utf-8") as stream:
        features = json.load(stream)
    if set(features) != node_ids:
        raise ValueError("feature node IDs do not exactly match target node IDs")

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_output = output_dir / "nodes.csv"
    edges_output = output_dir / "edges.csv"
    with nodes_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "title", "description"])
        writer.writeheader()
        for row in sorted(node_rows, key=lambda item: int(item["id"])):
            node_id = row["id"]
            category = row["page_type"].strip()
            description = (
                f'Verified Facebook page "{row["page_name"].strip()}"; '
                f"dataset category: {category}; source Facebook ID: "
                f'{row["facebook_id"]}; encoded description features: '
                f"{len(features[node_id])}."
            )
            writer.writerow(
                {"id": f"fb_{node_id}", "title": titles[node_id], "description": description}
            )

    with edges_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["source", "target", "weight", "description"]
        )
        writer.writeheader()
        for source, target in sorted(converted_edges, key=lambda edge: (int(edge[0]), int(edge[1]))):
            writer.writerow(
                {
                    "source": f"fb_{source}",
                    "target": f"fb_{target}",
                    "weight": "1.0",
                    "description": "Mutual like relationship between verified Facebook pages.",
                }
            )

    categories = Counter(row["page_type"].strip() for row in node_rows)
    metadata: dict[str, object] = {
        "dataset": "Facebook Large Page-Page Network",
        "source_repository": "UCI Machine Learning Repository",
        "source_dataset_id": 527,
        "doi": "10.24432/C50900",
        "license": "CC BY 4.0",
        "source_url": "https://archive.ics.uci.edu/dataset/527/facebook+large+page+page+network",
        "node_count": len(node_rows),
        "raw_edge_rows": len(edge_rows),
        "converted_edge_count": len(converted_edges),
        "removed_self_loops": self_loops,
        "removed_duplicate_undirected_edges": len(edge_rows) - self_loops - len(converted_edges),
        "isolated_node_count": len(isolated),
        "isolated_node_ids": [f"fb_{node_id}" for node_id in isolated],
        "categories": dict(sorted(categories.items())),
        "node_id_policy": "Prefix the source integer ID with 'fb_'.",
        "edge_policy": "Treat mutual likes as undirected, unweighted edges with weight 1.0.",
        "feature_policy": (
            "Preserve the raw feature JSON. Record only each node's feature count in its "
            "description because no feature-index vocabulary is included in the archive."
        ),
        "source_file_sha256": {
            target_path.name: sha256(target_path),
            edge_path.name: sha256(edge_path),
            feature_path.name: sha256(feature_path),
        },
        "output_file_sha256": {
            nodes_output.name: sha256(nodes_output),
            edges_output.name: sha256(edges_output),
        },
        **title_metadata,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = convert(args.source_dir, args.output_dir)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
