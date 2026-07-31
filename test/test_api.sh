#!/usr/bin/env bash
# Smoke-test the API. Start the server first:
#   uvicorn api:app --reload --app-dir src
#
# Checks the happy paths and, just as importantly, the error paths — an API
# that only works on valid input is half-finished.

set -u
BASE="${BASE:-http://localhost:8000}"

pass=0; fail=0

check () {
  local name="$1" expected="$2" actual="$3"
  if [ "$actual" = "$expected" ]; then
    printf '  ok    %-42s %s\n' "$name" "$actual"; pass=$((pass+1))
  else
    printf '  FAIL  %-42s got %s, want %s\n' "$name" "$actual" "$expected"; fail=$((fail+1))
  fi
}

status () { curl -s -o /dev/null -w '%{http_code}' "$@"; }

echo "testing $BASE"
echo
echo "health"
check "GET /health" 200 "$(status "$BASE/health")"
curl -s "$BASE/health" | python3 -m json.tool | sed 's/^/    /'

echo
echo "query"
check "valid question" 200 "$(status -X POST "$BASE/query" \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which AI practices are prohibited?"}')"

check "custom k" 200 "$(status -X POST "$BASE/query" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the penalties?","k":3}')"

echo
echo "validation — these must be rejected"
check "question too short" 422 "$(status -X POST "$BASE/query" \
  -H 'Content-Type: application/json' -d '{"question":"a"}')"

check "k above bound" 422 "$(status -X POST "$BASE/query" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the penalties?","k":500}')"

check "k below bound" 422 "$(status -X POST "$BASE/query" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the penalties?","k":0}')"

check "missing field" 422 "$(status -X POST "$BASE/query" \
  -H 'Content-Type: application/json' -d '{}')"

check "malformed json" 422 "$(status -X POST "$BASE/query" \
  -H 'Content-Type: application/json' -d 'not json')"

echo
echo "streaming (first few events)"
curl -sN -X POST "$BASE/query/stream" \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which AI practices are prohibited?","k":3}' \
  | head -c 600 | sed 's/^/    /'

echo
echo
echo "sample answer"
curl -s -X POST "$BASE/query" \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which AI practices are prohibited?","k":5}' \
  | python3 -c "
import json,sys
r = json.load(sys.stdin)
print('   ', r['answer'][:400].replace('\n', '\n    '))
print()
print(f\"    {r['latency_ms']}ms  \${r['cost_usd']:.5f}\")
for s in r['sources']:
    print(f\"    [{s['n']}] {s['source']} p.{s['page']} ({s['distance']:.3f})\")
"

echo
echo "----"
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]