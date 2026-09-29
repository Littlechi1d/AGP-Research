#!/usr/bin/env python3
"""Generate topology-labelled development and evaluation questions for new graphs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


QUESTION_TYPES = ("direct", "comparison", "path", "similarity")
DATASETS = {
    "github": {
        "prefix": "gh_", "target_id": "id", "target_class": "ml_target",
        "entity_plural": "developers", "link": "mutual-follow connections",
    },
    "deezer": {
        "prefix": "dz_", "target_id": "id", "target_class": "target",
        "entity_plural": "users", "link": "mutual-follow connections",
    },
}


def normalize(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def numeric_id(node_id: str) -> int:
    return int(node_id.rsplit("_", 1)[1])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_graph(nodes_path: Path, edges_path: Path):
    with nodes_path.open(encoding="utf-8", newline="") as stream:
        titles = {row["id"]: row["title"] for row in csv.DictReader(stream)}
    adjacency = {node_id: set() for node_id in titles}
    with edges_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source, target = row["source"], row["target"]
            if source not in adjacency or target not in adjacency:
                raise ValueError("edge endpoint is absent from nodes.csv")
            if source != target:
                adjacency[source].add(target)
                adjacency[target].add(source)
    return titles, adjacency


def read_categories(path: Path, dataset: str) -> dict[str, str]:
    config = DATASETS[dataset]
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {config["target_id"], config["target_class"]}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"targets file must contain {sorted(required)}")
    return {
        f'{config["prefix"]}{row[config["target_id"]]}': row[config["target_class"]]
        for row in rows
    }


def extracted_ids(question: str, titles: dict[str, str]) -> list[str]:
    padded = f" {normalize(question)} "
    matches = [
        (node_id, title) for node_id, title in titles.items()
        if f" {normalize(title)} " in padded
    ]
    matches.sort(key=lambda item: (-len(item[1]), item[1]))
    return [node_id for node_id, _ in matches]


def valid_question(question: str, seeds: list[str], titles: dict[str, str]) -> bool:
    padded = f" {normalize(question)} "
    return all(f" {normalize(titles[seed])} " in padded for seed in seeds)


def shuffled_nodes(adjacency, rng):
    nodes = sorted(adjacency, key=numeric_id)
    rng.shuffle(nodes)
    return nodes


def example(dataset, split, number, kind, question, seeds, relevant, rule):
    ordered = list(relevant) if kind == "path" else sorted(relevant, key=numeric_id)
    return {
        "id": f"{dataset}_{split}_{kind}_{number:02d}",
        "question": question,
        "question_type": kind,
        "seed_node_ids": seeds,
        "relevant_node_ids": ordered,
        "generation_rule": rule,
    }


def unique_three_hop_path(start: str, target: str, adjacency):
    if target in adjacency[start] or adjacency[start] & adjacency[target]:
        return None
    pairs = [
        (left, right)
        for left in adjacency[start]
        for right in (adjacency[left] & adjacency[target])
        if right != start and left != target
    ]
    return pairs[0] if len(pairs) == 1 else None


def generate_split(dataset, split, count, titles, adjacency, categories, used, rng):
    config = DATASETS[dataset]
    entities = config["entity_plural"]
    link = config["link"]
    results = []

    direct = []
    for seed in shuffled_nodes(adjacency, rng):
        relevant = adjacency[seed]
        if seed in used or not 3 <= len(relevant) <= 10:
            continue
        question = f"Which {entities} have a direct {link[:-1]} with {titles[seed]}"
        if not valid_question(question, [seed], titles):
            continue
        direct.append(example(dataset, split, len(direct) + 1, "direct", question,
                              [seed], relevant, "All nodes adjacent to the seed."))
        used.add(seed)
        if len(direct) == count:
            break
    if len(direct) != count:
        raise RuntimeError(f"created only {len(direct)} of {count} direct questions")
    results.extend(direct)

    comparison = []
    for first in shuffled_nodes(adjacency, rng):
        if first in used or not 2 <= len(adjacency[first]) <= 30:
            continue
        candidates = set().union(*(adjacency[n] for n in adjacency[first]))
        candidates.discard(first)
        candidates.difference_update(adjacency[first])
        ordered = sorted(candidates, key=numeric_id)
        rng.shuffle(ordered)
        for second in ordered:
            relevant = adjacency[first] & adjacency[second]
            if (second in used or not 2 <= len(adjacency[second]) <= 30
                    or not 2 <= len(relevant) <= 10):
                continue
            seeds = [first, second]
            question = (
                f"Which {entities} have direct {link} with both "
                f"{titles[first]} and {titles[second]}"
            )
            if not valid_question(question, seeds, titles):
                continue
            comparison.append(example(
                dataset, split, len(comparison) + 1, "comparison", question,
                seeds, relevant, "Intersection of the two seeds' direct-neighbour sets.",
            ))
            used.update(seeds)
            break
        if len(comparison) == count:
            break
    if len(comparison) != count:
        raise RuntimeError(f"created only {len(comparison)} of {count} comparison questions")
    results.extend(comparison)

    paths = []
    for start in shuffled_nodes(adjacency, rng):
        if start in used or not 2 <= len(adjacency[start]) <= 12:
            continue
        candidates, seen = [], set()
        left_nodes = sorted(adjacency[start], key=numeric_id)
        rng.shuffle(left_nodes)
        for left in left_nodes:
            right_nodes = sorted(adjacency[left] - {start}, key=numeric_id)
            rng.shuffle(right_nodes)
            for right in right_nodes:
                target_nodes = sorted(adjacency[right] - {start, left}, key=numeric_id)
                rng.shuffle(target_nodes)
                for target in target_nodes:
                    if target not in seen:
                        seen.add(target)
                        candidates.append(target)
                    if len(candidates) >= 250:
                        break
                if len(candidates) >= 250:
                    break
            if len(candidates) >= 250:
                break
        for target in candidates:
            if target in used or not 2 <= len(adjacency[target]) <= 12:
                continue
            middle = unique_three_hop_path(start, target, adjacency)
            if middle is None:
                continue
            seeds = [start, target]
            question = (
                f"To get from {titles[start]} to {titles[target]} using the fewest "
                f"{link}, which two {entities} do you pass through, in order"
            )
            if not valid_question(question, seeds, titles):
                continue
            paths.append(example(
                dataset, split, len(paths) + 1, "path", question, seeds, middle,
                "Two internal nodes, in order, on a unique shortest path of three edges.",
            ))
            used.update(seeds)
            break
        if len(paths) == count:
            break
    if len(paths) != count:
        raise RuntimeError(f"created only {len(paths)} of {count} path questions")
    results.extend(paths)

    similarity = []
    for seed in shuffled_nodes(adjacency, rng):
        if seed in used or not 2 <= len(adjacency[seed]) <= 30:
            continue
        distance_two = set().union(*(adjacency[n] for n in adjacency[seed]))
        distance_two.discard(seed)
        distance_two.difference_update(adjacency[seed])
        relevant = {node for node in distance_two if categories[node] == categories[seed]}
        if not 3 <= len(relevant) <= 10:
            continue
        question = (
            f"Which {entities} are in the same dataset class as {titles[seed]} and "
            f"can be reached through exactly one intermediate {entities[:-1]}, but "
            f"are not directly connected to {titles[seed]}"
        )
        if not valid_question(question, [seed], titles):
            continue
        similarity.append(example(
            dataset, split, len(similarity) + 1, "similarity", question, [seed], relevant,
            "Same-class nodes at shortest-path distance exactly two from the seed.",
        ))
        used.add(seed)
        if len(similarity) == count:
            break
    if len(similarity) != count:
        raise RuntimeError(f"created only {len(similarity)} of {count} similarity questions")
    results.extend(similarity)
    return results


def validate(examples, titles, adjacency, categories, expected_per_type):
    counts = Counter(item["question_type"] for item in examples)
    if counts != Counter({kind: expected_per_type for kind in QUESTION_TYPES}):
        raise AssertionError(f"unbalanced question types: {counts}")
    for item in examples:
        seeds = item["seed_node_ids"]
        relevant = set(item["relevant_node_ids"])
        if not relevant or relevant & set(seeds) or not relevant <= titles.keys():
            raise AssertionError(f"invalid labels in {item['id']}")
        if not valid_question(item["question"], seeds, titles):
            raise AssertionError(f"seed title is absent from {item['id']}")
        kind = item["question_type"]
        if kind == "direct":
            expected = adjacency[seeds[0]]
        elif kind == "comparison":
            expected = adjacency[seeds[0]] & adjacency[seeds[1]]
        elif kind == "path":
            middle = unique_three_hop_path(seeds[0], seeds[1], adjacency)
            expected = set(middle or ())
            if middle and item["relevant_node_ids"] != list(middle):
                raise AssertionError(f"path order mismatch in {item['id']}")
        else:
            seed = seeds[0]
            distance_two = set().union(*(adjacency[n] for n in adjacency[seed]))
            distance_two.discard(seed)
            distance_two.difference_update(adjacency[seed])
            expected = {node for node in distance_two if categories[node] == categories[seed]}
        if relevant != expected:
            raise AssertionError(f"topology label mismatch in {item['id']}")


def generate(dataset, nodes, edges, targets, output_dir, *, dev_per_type=5,
             eval_per_type=10, random_seed=90055):
    if dev_per_type < 1 or eval_per_type < 1:
        raise ValueError("question counts must be positive")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    nodes, edges, targets = map(Path, (nodes, edges, targets))
    titles, adjacency = read_graph(nodes, edges)
    categories = read_categories(targets, dataset)
    if set(categories) != set(titles):
        raise ValueError("target categories and graph node IDs differ")
    rng, used = random.Random(random_seed), set()
    development = generate_split(
        dataset, "dev", dev_per_type, titles, adjacency, categories, used, rng
    )
    dev_seeds = {seed for item in development for seed in item["seed_node_ids"]}
    evaluation = generate_split(
        dataset, "eval", eval_per_type, titles, adjacency, categories, used, rng
    )
    eval_seeds = {seed for item in evaluation for seed in item["seed_node_ids"]}
    validate(development, titles, adjacency, categories, dev_per_type)
    validate(evaluation, titles, adjacency, categories, eval_per_type)
    if dev_seeds & eval_seeds:
        raise AssertionError("development and evaluation seeds overlap")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"agp-{dataset}-questions-", dir=output.parent) as temp:
        staged = Path(temp)
        dev_file = staged / "questions_dev.json"
        eval_file = staged / "questions_eval.json"
        dev_file.write_text(json.dumps(development, indent=2) + "\n", encoding="utf-8")
        eval_file.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "random_seed": random_seed,
            "development_questions": len(development),
            "evaluation_questions": len(evaluation),
            "questions_per_type": {"development": dev_per_type, "evaluation": eval_per_type},
            "development_unique_seeds": len(dev_seeds),
            "evaluation_unique_seeds": len(eval_seeds),
            "seed_overlap_between_splits": 0,
            "label_source": "deterministic graph topology; no AGP or LLM output used",
            "keyword_mapping_note": (
                "Questions contain every intended seed title. GitHub usernames can be "
                "ordinary words, so the Facebook-only all-title substring extractor is "
                "not used to define seeds; experiment retrieval must use the frozen LLM "
                "keyword extractor and approximate title mapper."
                if dataset == "github" else
                "Questions contain every intended stable 'Deezer user <ID>' seed title; "
                "experiment retrieval must use the same frozen keyword-extraction and "
                "title-mapping procedure as the other datasets."
            ),
            "source_sha256": {
                "nodes": sha256(nodes), "edges": sha256(edges), "targets": sha256(targets),
            },
            "question_sha256": {
                "development": sha256(dev_file), "evaluation": sha256(eval_file),
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staged.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-per-type", type=int, default=5)
    parser.add_argument("--eval-per-type", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=90055)
    args = parser.parse_args()
    print(json.dumps(generate(
        args.dataset, args.nodes, args.edges, args.targets, args.output,
        dev_per_type=args.dev_per_type, eval_per_type=args.eval_per_type,
        random_seed=args.random_seed,
    ), indent=2))


if __name__ == "__main__":
    main()
