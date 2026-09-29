#!/usr/bin/env python3
"""Convert GitHub MUSAE or Deezer Europe into the project graph schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    name: str
    node_file: str
    edge_file: str
    feature_file: str
    node_columns: frozenset[str]
    edge_columns: tuple[str, str]
    prefix: str
    repository: str
    source_url: str
    doi: str | None
    license: str | None
    edge_description: str


CONFIGS = {
    "github": DatasetConfig(
        key="github",
        name="GitHub MUSAE Social Network",
        node_file="musae_git_target.csv",
        edge_file="musae_git_edges.csv",
        feature_file="musae_git_features.json",
        node_columns=frozenset({"id", "name", "ml_target"}),
        edge_columns=("id_1", "id_2"),
        prefix="gh_",
        repository="UCI Machine Learning Repository",
        source_url="https://archive.ics.uci.edu/dataset/588/github+musae",
        doi="10.24432/C5Z02B",
        license="CC BY 4.0",
        edge_description="Mutual follower relationship between GitHub developers.",
    ),
    "deezer": DatasetConfig(
        key="deezer",
        name="Deezer Europe Social Network",
        node_file="deezer_europe_target.csv",
        edge_file="deezer_europe_edges.csv",
        feature_file="deezer_europe_features.json",
        node_columns=frozenset({"id", "target"}),
        edge_columns=("node_1", "node_2"),
        prefix="dz_",
        repository="Stanford Network Analysis Project (SNAP)",
        source_url="https://snap.stanford.edu/data/feather-deezer-social.html",
        doi=None,
        license=None,
        edge_description="Mutual follower relationship between Deezer users.",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, required: set[str] | frozenset[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def _title_and_description(
    config: DatasetConfig, row: dict[str, str], feature_count: int
) -> tuple[str, str]:
    node_id = row["id"]
    if config.key == "github":
        name = row["name"].strip()
        if not name:
            raise ValueError(f"GitHub developer {node_id} has an empty name")
        return name, (
            f'GitHub developer "{name}"; binary dataset class: {row["ml_target"]}; '
            f"encoded profile features: {feature_count}."
        )
    return f"Deezer user {node_id}", (
        f"Anonymized Deezer Europe user {node_id}; binary dataset class: "
        f'{row["target"]}; encoded liked-artist features: {feature_count}.'
    )


def convert(dataset: str, source_dir: Path, output_dir: Path) -> dict[str, object]:
    config = CONFIGS[dataset]
    node_path = source_dir / config.node_file
    edge_path = source_dir / config.edge_file
    feature_path = source_dir / config.feature_file
    for path in (node_path, edge_path, feature_path):
        if not path.is_file():
            raise FileNotFoundError(f"required source file not found: {path}")

    node_rows = read_csv(node_path, config.node_columns)
    edge_rows = read_csv(edge_path, set(config.edge_columns))
    node_ids = {row["id"] for row in node_rows}
    if len(node_ids) != len(node_rows):
        raise ValueError("node IDs are not unique")
    if any(not node_id.isdigit() for node_id in node_ids):
        raise ValueError("node IDs must be nonnegative integers")

    source_col, target_col = config.edge_columns
    unknown = {
        endpoint
        for row in edge_rows
        for endpoint in (row[source_col], row[target_col])
        if endpoint not in node_ids
    }
    if unknown:
        raise ValueError(f"edges contain unknown node IDs: {sorted(unknown)[:10]}")

    self_loops = 0
    converted_edges: set[tuple[str, str]] = set()
    for row in edge_rows:
        source, target = row[source_col], row[target_col]
        if source == target:
            self_loops += 1
            continue
        converted_edges.add(tuple(sorted((source, target), key=int)))

    degrees: Counter[str] = Counter(
        endpoint for edge in converted_edges for endpoint in edge
    )
    isolated = sorted(node_ids - set(degrees), key=int)
    with feature_path.open(encoding="utf-8") as stream:
        features = json.load(stream)
    if set(features) != node_ids:
        raise ValueError("feature node IDs do not exactly match target node IDs")

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_output = output_dir / "nodes.csv"
    edges_output = output_dir / "edges.csv"
    normalized_titles: set[str] = set()
    with nodes_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "title", "description"])
        writer.writeheader()
        for row in sorted(node_rows, key=lambda item: int(item["id"])):
            title, description = _title_and_description(
                config, row, len(features[row["id"]])
            )
            normalized = " ".join(title.casefold().split())
            if normalized in normalized_titles:
                raise ValueError(f"duplicate normalized title: {title}")
            normalized_titles.add(normalized)
            writer.writerow({
                "id": f'{config.prefix}{row["id"]}',
                "title": title,
                "description": description,
            })

    with edges_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["source", "target", "weight", "description"]
        )
        writer.writeheader()
        for source, target in sorted(
            converted_edges, key=lambda edge: (int(edge[0]), int(edge[1]))
        ):
            writer.writerow({
                "source": f"{config.prefix}{source}",
                "target": f"{config.prefix}{target}",
                "weight": "1.0",
                "description": config.edge_description,
            })

    class_column = "ml_target" if dataset == "github" else "target"
    metadata: dict[str, object] = {
        "dataset": config.name,
        "source_repository": config.repository,
        "source_url": config.source_url,
        "doi": config.doi,
        "license": config.license,
        "node_count": len(node_rows),
        "raw_edge_rows": len(edge_rows),
        "converted_edge_count": len(converted_edges),
        "removed_self_loops": self_loops,
        "removed_duplicate_undirected_edges": (
            len(edge_rows) - self_loops - len(converted_edges)
        ),
        "isolated_node_count": len(isolated),
        "isolated_node_ids": [f"{config.prefix}{node_id}" for node_id in isolated],
        "binary_classes": dict(sorted(Counter(
            row[class_column] for row in node_rows
        ).items())),
        "node_id_policy": f"Prefix the source integer ID with '{config.prefix}'.",
        "title_policy": (
            "Preserve the supplied GitHub username."
            if dataset == "github"
            else "Use 'Deezer user <source ID>' because source users are anonymized."
        ),
        "edge_policy": "Treat mutual follows as undirected edges with weight 1.0.",
        "feature_policy": (
            "Preserve the raw feature JSON and record only each node's feature count "
            "because the archives do not include a feature-index vocabulary."
        ),
        "source_file_sha256": {
            path.name: sha256(path) for path in (node_path, edge_path, feature_path)
        },
        "output_file_sha256": {
            nodes_output.name: sha256(nodes_output),
            edges_output.name: sha256(edges_output),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        convert(args.dataset, args.source_dir, args.output_dir),
        indent=2, ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
