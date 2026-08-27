#!/usr/bin/env python3
"""Generate deterministic topology-grounded questions for Facebook Large.

The labels are computed from graph structure before any AGP retrieval is run.
This prevents the evaluated method from defining its own ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


QUESTION_TYPES = ("direct", "comparison", "path", "similarity")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create development and test questions for Facebook Large"
    )
    parser.add_argument("--nodes", type=Path, required=True, help="Converted nodes.csv")
    parser.add_argument("--edges", type=Path, required=True, help="Converted edges.csv")
    parser.add_argument(
        "--targets", type=Path, required=True,
        help="Original musae_facebook_target.csv (provides page categories)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-per-type", type=int, default=5)
    parser.add_argument("--test-per-type", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=90055)
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def read_graph(nodes_path: Path, edges_path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    with nodes_path.open(encoding="utf-8", newline="") as stream:
        titles = {row["id"]: row["title"] for row in csv.DictReader(stream)}
    adjacency = {node_id: set() for node_id in titles}
    with edges_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source, target = row["source"], row["target"]
            if source == target:
                continue
            adjacency[source].add(target)
            adjacency[target].add(source)
    return titles, adjacency


def read_categories(targets_path: Path) -> dict[str, str]:
    with targets_path.open(encoding="utf-8", newline="") as stream:
        return {f"fb_{row['id']}": row["page_type"] for row in csv.DictReader(stream)}


def extracted_ids(question: str, titles: dict[str, str]) -> list[str]:
    """Mirror agp_research.planner.local_keyword_extraction exactly."""
    padded_question = f" {normalize(question)} "
    matches = [
        (node_id, title) for node_id, title in titles.items()
        if f" {normalize(title)} " in padded_question
    ]
    matches.sort(key=lambda item: (-len(item[1]), item[1]))
    return [node_id for node_id, _ in matches]


def valid_question(question: str, seeds: list[str], titles: dict[str, str]) -> bool:
    return set(extracted_ids(question, titles)) == set(seeds)


def example(
    number: int,
    kind: str,
    question: str,
    seeds: list[str],
    relevant: set[str],
    rule: str,
) -> dict:
    return {
        "id": f"facebook_{kind}_{number:02d}",
        "question": question,
        "question_type": kind,
        "seed_node_ids": seeds,
        "relevant_node_ids": sorted(relevant, key=numeric_id),
        "generation_rule": rule,
    }


def numeric_id(node_id: str) -> int:
    return int(node_id.removeprefix("fb_"))


def shuffled_nodes(adjacency: dict[str, set[str]], rng: random.Random) -> list[str]:
    nodes = sorted(adjacency, key=numeric_id)
    rng.shuffle(nodes)
    return nodes


def make_direct(
    count: int, titles: dict[str, str], adjacency: dict[str, set[str]],
    used_seeds: set[str], rng: random.Random,
) -> list[dict]:
    results = []
    for seed in shuffled_nodes(adjacency, rng):
        relevant = adjacency[seed]
        if seed in used_seeds or not 3 <= len(relevant) <= 10:
            continue
        question = f"Which pages directly neighbor {titles[seed]}"
        if not valid_question(question, [seed], titles):
            continue
        results.append(example(len(results) + 1, "direct", question, [seed], relevant,
            "All graph nodes adjacent to the seed page."))
        used_seeds.add(seed)
        if len(results) == count:
            return results
    raise RuntimeError(f"Could create only {len(results)} of {count} direct questions")


def make_comparison(
    count: int, titles: dict[str, str], adjacency: dict[str, set[str]],
    used_seeds: set[str], rng: random.Random,
) -> list[dict]:
    results = []
    for first in shuffled_nodes(adjacency, rng):
        if first in used_seeds or not 2 <= len(adjacency[first]) <= 30:
            continue
        candidates = set()
        for neighbor in adjacency[first]:
            candidates.update(adjacency[neighbor])
        candidates.discard(first)
        candidates.difference_update(adjacency[first])
        ordered = sorted(candidates, key=numeric_id)
        rng.shuffle(ordered)
        for second in ordered:
            if second in used_seeds or not 2 <= len(adjacency[second]) <= 30:
                continue
            relevant = adjacency[first] & adjacency[second]
            if not 2 <= len(relevant) <= 10:
                continue
            question = f"Which pages are liked by both {titles[first]} and {titles[second]}"
            seeds = [first, second]
            if not valid_question(question, seeds, titles):
                continue
            results.append(example(len(results) + 1, "comparison", question, seeds, relevant,
                "The intersection of the two seed pages' direct-neighbor sets."))
            used_seeds.update(seeds)
            break
        if len(results) == count:
            return results
    raise RuntimeError(f"Could create only {len(results)} of {count} comparison questions")


def unique_three_hop_path(
    start: str, target: str, adjacency: dict[str, set[str]]
) -> tuple[str, str] | None:
    if target in adjacency[start] or adjacency[start] & adjacency[target]:
        return None
    middle_pairs = [
        (left, right)
        for left in adjacency[start]
        for right in (adjacency[left] & adjacency[target])
        if right != start and left != target
    ]
    return middle_pairs[0] if len(middle_pairs) == 1 else None


def make_paths(
    count: int, titles: dict[str, str], adjacency: dict[str, set[str]],
    used_seeds: set[str], rng: random.Random,
) -> list[dict]:
    results = []
    for start in shuffled_nodes(adjacency, rng):
        if start in used_seeds or not 2 <= len(adjacency[start]) <= 12:
            continue
        # Examine concrete length-three walks instead of materializing a huge
        # three-hop union around hub nodes. The set and cap keep this bounded.
        candidates = []
        seen_targets = set()
        left_nodes = sorted(adjacency[start], key=numeric_id)
        rng.shuffle(left_nodes)
        for left in left_nodes:
            right_nodes = sorted(adjacency[left] - {start}, key=numeric_id)
            rng.shuffle(right_nodes)
            for right in right_nodes:
                target_nodes = sorted(adjacency[right] - {start, left}, key=numeric_id)
                rng.shuffle(target_nodes)
                for target in target_nodes:
                    if target not in seen_targets:
                        seen_targets.add(target)
                        candidates.append(target)
                    if len(candidates) >= 250:
                        break
                if len(candidates) >= 250:
                    break
            if len(candidates) >= 250:
                break
        for target in candidates:
            if target in used_seeds or not 2 <= len(adjacency[target]) <= 12:
                continue
            middle = unique_three_hop_path(start, target, adjacency)
            if middle is None:
                continue
            question = f"Which pages form the unique shortest path between {titles[start]} and {titles[target]}"
            seeds = [start, target]
            if not valid_question(question, seeds, titles):
                continue
            results.append(example(len(results) + 1, "path", question, seeds, set(middle),
                "The two internal nodes on the unique shortest path of exactly three edges."))
            used_seeds.update(seeds)
            break
        if len(results) == count:
            return results
    raise RuntimeError(f"Could create only {len(results)} of {count} path questions")


def make_similarity(
    count: int, titles: dict[str, str], adjacency: dict[str, set[str]],
    categories: dict[str, str], used_seeds: set[str], rng: random.Random,
) -> list[dict]:
    results = []
    for seed in shuffled_nodes(adjacency, rng):
        if seed in used_seeds or not 2 <= len(adjacency[seed]) <= 30:
            continue
        distance_two = set()
        for neighbor in adjacency[seed]:
            distance_two.update(adjacency[neighbor])
        distance_two.discard(seed)
        distance_two.difference_update(adjacency[seed])
        relevant = {node for node in distance_two if categories[node] == categories[seed]}
        if not 3 <= len(relevant) <= 10:
            continue
        question = f"Which pages similar to {titles[seed]} share its category and are two hops away"
        if not valid_question(question, [seed], titles):
            continue
        results.append(example(len(results) + 1, "similarity", question, [seed], relevant,
            "Same-category nodes at shortest-path distance exactly two from the seed."))
        used_seeds.add(seed)
        if len(results) == count:
            return results
    raise RuntimeError(f"Could create only {len(results)} of {count} similarity questions")


def generate_split(
    per_type: int, titles: dict[str, str], adjacency: dict[str, set[str]],
    categories: dict[str, str], used_seeds: set[str], rng: random.Random,
) -> list[dict]:
    result = []
    result.extend(make_direct(per_type, titles, adjacency, used_seeds, rng))
    result.extend(make_comparison(per_type, titles, adjacency, used_seeds, rng))
    result.extend(make_paths(per_type, titles, adjacency, used_seeds, rng))
    result.extend(make_similarity(per_type, titles, adjacency, categories, used_seeds, rng))
    return result


def validate(
    examples: list[dict], titles: dict[str, str], adjacency: dict[str, set[str]],
    categories: dict[str, str], expected_per_type: int,
) -> None:
    counts = Counter(item["question_type"] for item in examples)
    if counts != Counter({kind: expected_per_type for kind in QUESTION_TYPES}):
        raise AssertionError(f"Unbalanced question types: {counts}")
    all_ids = set(titles)
    for item in examples:
        seeds = item["seed_node_ids"]
        relevant = set(item["relevant_node_ids"])
        if not relevant or not relevant <= all_ids or relevant & set(seeds):
            raise AssertionError(f"Invalid label IDs in {item['id']}")
        if set(extracted_ids(item["question"], titles)) != set(seeds):
            raise AssertionError(f"Seed extraction failed for {item['id']}")
        kind = item["question_type"]
        if kind == "direct":
            expected = adjacency[seeds[0]]
        elif kind == "comparison":
            expected = adjacency[seeds[0]] & adjacency[seeds[1]]
        elif kind == "path":
            middle = unique_three_hop_path(seeds[0], seeds[1], adjacency)
            expected = set(middle or ())
        else:
            seed = seeds[0]
            distance_two = set().union(*(adjacency[n] for n in adjacency[seed]))
            distance_two.discard(seed)
            distance_two.difference_update(adjacency[seed])
            expected = {n for n in distance_two if categories[n] == categories[seed]}
        if relevant != expected:
            raise AssertionError(f"Topology label mismatch in {item['id']}")


def main() -> None:
    args = arguments()
    if args.dev_per_type < 1 or args.test_per_type < 1:
        raise SystemExit("--dev-per-type and --test-per-type must be positive")
    titles, adjacency = read_graph(args.nodes, args.edges)
    categories = read_categories(args.targets)
    if set(categories) != set(titles):
        raise SystemExit("Target categories and converted node IDs do not match")

    rng = random.Random(args.random_seed)
    used_seeds: set[str] = set()
    development = generate_split(
        args.dev_per_type, titles, adjacency, categories, used_seeds, rng
    )
    dev_seeds = {seed for item in development for seed in item["seed_node_ids"]}
    test = generate_split(
        args.test_per_type, titles, adjacency, categories, used_seeds, rng
    )
    test_seeds = {seed for item in test for seed in item["seed_node_ids"]}

    validate(development, titles, adjacency, categories, args.dev_per_type)
    validate(test, titles, adjacency, categories, args.test_per_type)
    if dev_seeds & test_seeds:
        raise AssertionError("Development and test seed pages overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "facebook_questions_dev.json").write_text(
        json.dumps(development, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "facebook_questions_test.json").write_text(
        json.dumps(test, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "random_seed": args.random_seed,
        "development_questions": len(development),
        "test_questions": len(test),
        "questions_per_type": {
            "development": args.dev_per_type, "test": args.test_per_type
        },
        "development_unique_seeds": len(dev_seeds),
        "test_unique_seeds": len(test_seeds),
        "seed_overlap_between_splits": 0,
        "label_source": "deterministic graph topology; no AGP output used",
    }
    (args.output_dir / "facebook_questions_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
