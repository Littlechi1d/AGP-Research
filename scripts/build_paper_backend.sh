#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/AGP-dynamic" >&2
  exit 2
fi

source_dir="$(cd "$1" && pwd)"
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
output_dir="$project_dir/build/paper_backend"
mkdir -p "$output_dir"

# Five published query assignments store 1-based IDs into 0-based score arrays.
# Patch only a build copy; never modify the external checkout.
cp "$source_dir/graph/Graph.cpp" "$output_dir/Graph.cpp"
patch -s "$output_dir/Graph.cpp" \
  "$project_dir/paper_backend/patches/graph_query_zero_based.patch"

compiler="${CXX:-c++}"
"$compiler" -std=c++17 -O3 \
  -I"$source_dir" \
  -I"$source_dir/graph" \
  "$project_dir/paper_backend/agp_query_bridge.cpp" \
  "$output_dir/Graph.cpp" \
  "$source_dir/graph/Vertex.cpp" \
  "$source_dir/DS/DS.cpp" \
  "$source_dir/MyLib/MyTimer.cpp" \
  -o "$output_dir/agp_query_bridge"

echo "$output_dir/agp_query_bridge"
