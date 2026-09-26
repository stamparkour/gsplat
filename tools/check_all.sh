#!/usr/bin/env bash
# Runs every backward-pass check (docs/backward.md). Exits nonzero if any fails.
cd "$(dirname "$0")"
status=0
for f in blend alpha conic cov2d mean rotscale color e2e; do
    echo "== check_$f.py"
    out=$(python3 "check_$f.py") || status=1
    echo "$out"
    grep -q -e MISMATCH -e FAILED <<<"$out" && status=1
done
echo
[ $status -eq 0 ] && echo "all checks ok" || echo "SOME CHECKS FAILED"
exit $status
