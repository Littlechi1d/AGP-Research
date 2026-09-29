import csv
import json
from pathlib import Path

import pytest

from scripts.prepare_additional_social_graph import convert


def _write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


@pytest.mark.parametrize("dataset", ["github", "deezer"])
def test_converter_writes_project_schema_and_removes_duplicate_edges(
    tmp_path: Path, dataset: str
):
    source = tmp_path / "raw"
    output = tmp_path / "output"
    source.mkdir()
    if dataset == "github":
        _write_csv(source / "musae_git_target.csv", ["id", "name", "ml_target"],
                   [[0, "Alice", 0], [1, "Bob", 1], [2, "Carol", 0]])
        _write_csv(source / "musae_git_edges.csv", ["id_1", "id_2"],
                   [[0, 1], [1, 0], [2, 2]])
        feature = source / "musae_git_features.json"
    else:
        _write_csv(source / "deezer_europe_target.csv", ["id", "target"],
                   [[0, 0], [1, 1], [2, 0]])
        _write_csv(source / "deezer_europe_edges.csv", ["node_1", "node_2"],
                   [[0, 1], [1, 0], [2, 2]])
        feature = source / "deezer_europe_features.json"
    feature.write_text(json.dumps({"0": [1], "1": [2, 3], "2": []}))

    metadata = convert(dataset, source, output)
    nodes = list(csv.DictReader((output / "nodes.csv").open()))
    edges = list(csv.DictReader((output / "edges.csv").open()))
    assert len(nodes) == 3
    assert len(edges) == 1
    assert edges[0]["weight"] == "1.0"
    assert metadata["removed_self_loops"] == 1
    assert metadata["removed_duplicate_undirected_edges"] == 1
    assert metadata["isolated_node_count"] == 1
    assert json.loads((output / "metadata.json").read_text())["node_count"] == 3


def test_converter_rejects_unknown_edge_endpoint(tmp_path: Path):
    _write_csv(tmp_path / "musae_git_target.csv", ["id", "name", "ml_target"],
               [[0, "Alice", 0]])
    _write_csv(tmp_path / "musae_git_edges.csv", ["id_1", "id_2"], [[0, 9]])
    (tmp_path / "musae_git_features.json").write_text(json.dumps({"0": []}))
    with pytest.raises(ValueError, match="unknown node IDs"):
        convert("github", tmp_path, tmp_path / "output")
