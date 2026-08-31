#!/usr/bin/env bash
# Turn an exported .pte into the context binaries the runtime loads.
#
#   ./extract_context.sh <artifact_dir> <out_dir> <n_shards> [params.json]
#
# One `forward_i.bin` per shard plus its JSON. The JSON is not optional -- the runner reads tensor
# names, shapes and quantisation encodings out of it, and the KV write-back needs the cache scales.
set -euo pipefail
ART=${1:?artifact dir}
OUT=${2:?output dir}
N=${3:?shard count (target 6, draft 1)}
PARAMS=${4:-}
: "${EXECUTORCH_ROOT:?set EXECUTORCH_ROOT}"
: "${QNN_SDK_ROOT:?set QNN_SDK_ROOT}"
Q="$QNN_SDK_ROOT/bin/x86_64-linux-clang"

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cp "$ART"/hybrid_llama_qnn.pte "$TMP"/
cd "$EXECUTORCH_ROOT"
python -c "
from executorch.backends.qualcomm.utils.utils import dump_context_from_pte
print('binaries', len(dump_context_from_pte('$TMP/hybrid_llama_qnn.pte')))"

rm -rf "$OUT"; mkdir -p "$OUT"
for i in $(seq 0 $((N-1))); do
  # a sharded target dumps kv_forward_<i>.bin; a single-graph draft dumps prefill_forward_0.bin
  src="$TMP/kv_forward_$i.bin"; [ -f "$src" ] || src="$TMP/prefill_forward_$i.bin"
  cp "$src" "$OUT/forward_$i.bin"
  LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang" \
    "$Q/qnn-context-binary-utility" --context_binary "$OUT/forward_$i.bin" \
      --json_file "$OUT/forward_${i}_json.json" >/dev/null
done
[ -n "$PARAMS" ] && cp "$PARAMS" "$OUT/params.json"
echo "$OUT: $(du -sh "$OUT" | cut -f1)"
