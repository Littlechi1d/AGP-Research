# Additional dataset preparation

The project adds GitHub MUSAE and Deezer Europe as two independent graph
domains alongside Facebook. Original downloads are preserved outside the Git
repository under `/Users/feiyuzhang/Desktop/COMP90055/datasets/`.

## Convert GitHub MUSAE

```sh
python3 scripts/prepare_additional_social_graph.py \
  --dataset github \
  --source-dir /Users/feiyuzhang/Desktop/COMP90055/datasets/github_musae/raw/git_web_ml \
  --output-dir data/github_musae
```

GitHub usernames become node titles, source IDs receive the `gh_` prefix, and
mutual follower relationships become undirected unit-weight edges.

## Convert Deezer Europe

```sh
python3 scripts/prepare_additional_social_graph.py \
  --dataset deezer \
  --source-dir /Users/feiyuzhang/Desktop/COMP90055/datasets/deezer_europe/raw/deezer_europe \
  --output-dir data/deezer_europe
```

The Deezer archive anonymizes its users. Titles therefore use the reproducible
form `Deezer user <source ID>` rather than invented names. Source IDs receive
the `dz_` prefix. Both converters preserve the original feature JSON outside
the repository and record only the feature count in each node description.

Each output directory contains:

- `nodes.csv`: `id,title,description`
- `edges.csv`: `source,target,weight,description`
- `metadata.json`: provenance, transformation counts, policies, and SHA-256
  checksums for source and output files

## Generate topology-labelled questions

The generator creates 20 development questions and 40 seed-disjoint evaluation
questions per dataset. Each split is balanced across direct, comparison,
three-edge connection-chain, and same-class distance-two questions. Labels are
calculated from graph topology without using AGP rankings or LLM answers.

```sh
python3 scripts/generate_additional_questions.py \
  --dataset github \
  --nodes data/github_musae/nodes.csv \
  --edges data/github_musae/edges.csv \
  --targets /Users/feiyuzhang/Desktop/COMP90055/datasets/github_musae/raw/git_web_ml/musae_git_target.csv \
  --output data/github_musae/benchmark

python3 scripts/generate_additional_questions.py \
  --dataset deezer \
  --nodes data/deezer_europe/nodes.csv \
  --edges data/deezer_europe/edges.csv \
  --targets /Users/feiyuzhang/Desktop/COMP90055/datasets/deezer_europe/raw/deezer_europe/deezer_europe_target.csv \
  --output data/deezer_europe/benchmark
```

Each benchmark directory contains `questions_dev.json`,
`questions_eval.json`, and `manifest.json`. Evaluation questions must not be
used to tune prompts, mapping thresholds, AGP parameters, or selectors.
