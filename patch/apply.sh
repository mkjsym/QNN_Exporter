#!/bin/bash
# Apply the QNN export patch to an ExecuTorch checkout.
#
#   ./apply.sh [path-to-executorch]      # default: $EXECUTORCH_ROOT, else ../../../executorch
#   ./apply.sh -R [path]                 # undo
#
# Exports run out of the ExecuTorch source tree, not out of this repo, so this is the one place
# where we reach into somebody else's checkout. Keeping it as a patch rather than a fork means
# rebasing onto a newer ExecuTorch is a merge conflict, not an archaeology exercise.
set -euo pipefail

BASE=88a5f60543adb2e0baebf451a8d066c6649fd4b7
HERE=$(cd "$(dirname "$0")" && pwd)
PATCH=$HERE/executorch-qnn-export.patch

REVERSE=""
if [ "${1:-}" = "-R" ] || [ "${1:-}" = "--reverse" ]; then REVERSE="-R"; shift; fi
ET=${1:-${EXECUTORCH_ROOT:-$HERE/../../executorch}}

[ -d "$ET/.git" ] || { echo "not an ExecuTorch git checkout: $ET" >&2; exit 1; }
ET=$(cd "$ET" && pwd)

head=$(git -C "$ET" rev-parse HEAD)
if [ "$head" != "$BASE" ]; then
  echo "note: developed against $BASE, this checkout is at $head"
  echo "      a clean apply here means the touched files did not move; a reject means they did."
fi

if git -C "$ET" apply $REVERSE --check "$PATCH" 2>/dev/null; then
  git -C "$ET" apply $REVERSE "$PATCH"
elif git -C "$ET" apply $REVERSE --3way --check "$PATCH" 2>/dev/null; then
  echo "exact apply failed; falling back to a 3-way merge"
  git -C "$ET" apply $REVERSE --3way "$PATCH"
else
  echo "cannot apply. already applied? try: $0 -R $ET" >&2
  git -C "$ET" apply $REVERSE --check "$PATCH"      # let git print why
  exit 1
fi

[ -n "$REVERSE" ] && { echo "reverted in $ET"; exit 0; }

echo "applied to $ET"
echo
echo "check that the registrations took, from an env where executorch imports:"
echo "  python -c \"from executorch.examples.qualcomm.oss_scripts.llama import SUPPORTED_LLM_MODELS as S; \\"
echo "             c=S['qwen3-8b']; print(c.quant_recipe.__name__, c.shard_layers, c.num_sharding)\""
echo "  expected: Qwen3_8BQuantRecipe [2, 10, 18, 26, 34] 6"
echo
echo "then follow docs/export-guide.md from '4. Export 실행'."
