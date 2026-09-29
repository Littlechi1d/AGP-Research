from scripts.generate_additional_questions import extracted_ids, validate


def test_validator_checks_all_four_topology_rules_and_path_order():
    ids = [f"gh_{i}" for i in range(15)]
    titles = {node: f"Developer {i}" for i, node in enumerate(ids)}
    adjacency = {node: set() for node in ids}

    def edge(a, b):
        adjacency[f"gh_{a}"].add(f"gh_{b}")
        adjacency[f"gh_{b}"].add(f"gh_{a}")

    for other in (1, 2, 3):
        edge(0, other)
    for common in (6, 7):
        edge(4, common)
        edge(5, common)
    edge(8, 9)
    edge(9, 10)
    edge(10, 11)
    for target in (12, 13, 14):
        edge(2, target)
    categories = {node: "A" for node in ids}
    categories["gh_1"] = categories["gh_2"] = categories["gh_3"] = "B"
    questions = [
        {"id": "d", "question_type": "direct", "question": "Developer 0",
         "seed_node_ids": ["gh_0"], "relevant_node_ids": ["gh_1", "gh_2", "gh_3"]},
        {"id": "c", "question_type": "comparison", "question": "Developer 4 Developer 5",
         "seed_node_ids": ["gh_4", "gh_5"], "relevant_node_ids": ["gh_6", "gh_7"]},
        {"id": "p", "question_type": "path", "question": "Developer 8 Developer 11",
         "seed_node_ids": ["gh_8", "gh_11"], "relevant_node_ids": ["gh_9", "gh_10"]},
        {"id": "s", "question_type": "similarity", "question": "Developer 1",
         "seed_node_ids": ["gh_1"], "relevant_node_ids": ["gh_2", "gh_3"]},
    ]
    validate(questions, titles, adjacency, categories, 1)
    assert extracted_ids("Developer 1 and Developer 10", titles) == ["gh_10", "gh_1"]
