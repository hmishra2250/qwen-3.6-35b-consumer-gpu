#!/bin/bash
# Quality comparison test: sends hard prompts to a running server
# Tests multi-step reasoning, code, logic, and physics
# Run with server started as: ./run.sh (Q4_0) or KV_TYPE=q8_0 CTX=65536 ./run.sh (q8_0)
set -euo pipefail

PORT=${PORT:-8080}
HOST=${HOST:-127.0.0.1}
URL="http://$HOST:$PORT/v1/chat/completions"
LABEL=${1:-"test"}
RESULTS_DIR="$(dirname "$0")/results"

mkdir -p "$RESULTS_DIR"

if ! curl -s "http://$HOST:$PORT/health" | grep -q "ok"; then
    echo "ERROR: Server not running. Start with ./run.sh first."
    exit 1
fi

echo "=== Quality Test Suite: $LABEL ==="
echo "Server: $URL"
echo ""

run_test() {
    local name="$1"
    local prompt="$2"
    local max_tokens="${3:-16384}"

    echo "[Running] $name..."
    local start=$(date +%s%N)

    local result=$(curl -s "$URL" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "
import json
print(json.dumps({
    'messages': [{'role': 'user', 'content': '''$prompt'''}],
    'temperature': 0.0,
    'max_tokens': $max_tokens
}))
")")

    local end=$(date +%s%N)
    local elapsed=$(( (end - start) / 1000000 ))

    local tokens=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo "0")
    local content=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message'].get('content',''))" 2>/dev/null || echo "")

    echo "  Tokens: $tokens | Time: ${elapsed}ms"

    echo "$result" > "$RESULTS_DIR/quality_${LABEL}_$(echo "$name" | tr ' ' '_').json"

    if [ -n "$content" ]; then
        echo "  Answer preview: $(echo "$content" | head -5)"
    else
        echo "  [Answer still in thinking tokens]"
    fi
    echo ""
}

# Test 1: Multi-step calculus
run_test "calculus_optimization" \
    "A farmer has 100m of fencing for 3 sides of a rectangle against a river. Find dimensions that maximize area. Then find new optimal dimensions if he splits into 2 equal pens with a fence parallel to the river. Show calculus." \
    16384

# Test 2: Einstein's riddle
run_test "einsteins_riddle" \
    "5 houses, different colors. Different nationalities, beverages, smokes, pets. 1.Brit=red 2.Swede=dogs 3.Dane=tea 4.Green left of white 5.Green=coffee 6.PallMall=birds 7.Yellow=Dunhill 8.Center=milk 9.Norwegian=house1 10.Blend next to cats 11.Horses next to Dunhill 12.BlueMaster=beer 13.German=Prince 14.Norwegian next to blue 15.Blend neighbor drinks water. Who owns fish?" \
    16384

# Test 3: Code with concurrency
run_test "threadsafe_lru_cache" \
    "Write a thread-safe LRU cache in Python with O(1) get/put, threading locks, configurable max size, evict LRU when full. Include 10-thread stress test." \
    16384

# Test 4: Bug finding
run_test "async_bug_detection" \
    "Find ALL bugs (at least 3) in this async Python rate limiter. Explain each precisely: class RateLimiter with is_allowed(client_id) using asyncio.get_event_loop().time(), cleanup() iterating self.requests dict while modifying, and main() that creates cleanup_task but never cancels it." \
    8192

# Test 5: Physics derivation
run_test "schwarzschild_radius" \
    "Derive the Schwarzschild radius from Newtons escape velocity. Show GR gives same answer. Calculate R_s for Earth (M=5.972e24 kg) with units. What happens physically if compressed to that size?" \
    16384

echo "=== All tests complete ==="
echo "Results saved to $RESULTS_DIR/quality_${LABEL}_*.json"
echo ""
echo "To compare Q4_0 vs q8_0:"
echo "  1. ./run.sh  # start Q4_0 server"
echo "  2. ./tests/quality.sh q4_0"
echo "  3. KV_TYPE=q8_0 CTX=65536 ./run.sh  # restart with q8_0"
echo "  4. ./tests/quality.sh q8_0"
echo "  5. diff results"
