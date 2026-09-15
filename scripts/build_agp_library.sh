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

cp "$source_dir/graph/Graph.cpp" "$output_dir/Graph.native.cpp"
patch -s "$output_dir/Graph.native.cpp" \
  "$project_dir/paper_backend/patches/graph_query_zero_based.patch"
patch -s "$output_dir/Graph.native.cpp" \
  "$project_dir/paper_backend/patches/graph_query_persistent.patch"
perl -i -pe \
  's/^    printf\("query time: %.9lf\\n", end - start\);$/    (void) end;  \/\/ Native API callers measure latency./' \
  "$output_dir/Graph.native.cpp"

compiler="${CXX:-c++}"
case "$(uname -s)" in
  Darwin)
    library="$output_dir/libagp_api.dylib"
    shared_flags=(-dynamiclib)
    ;;
  *)
    library="$output_dir/libagp_api.so"
    shared_flags=(-shared)
    ;;
esac

"$compiler" -std=c++17 -O3 -fPIC "${shared_flags[@]}" \
  -I"$source_dir" \
  -I"$source_dir/graph" \
  "$project_dir/paper_backend/agp_api.cpp" \
  "$output_dir/Graph.native.cpp" \
  "$source_dir/graph/Vertex.cpp" \
  "$source_dir/DS/DS.cpp" \
  "$source_dir/MyLib/MyTimer.cpp" \
  -o "$library"

echo "$library"
