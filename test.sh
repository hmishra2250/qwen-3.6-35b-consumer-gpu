#!/bin/bash
# Quick smoke test: send a prompt to the running server and measure speed

PORT=${PORT:-8080}
HOST=${HOST:-127.0.0.1}
URL="http://$HOST:$PORT/v1/chat/completions"

echo "=== Qwen3.6-35B-A3B Smoke Test ==="
echo "Server: $URL"
echo ""

# Check server is running
if ! curl -s "http://$HOST:$PORT/health" | grep -q "ok"; then
    echo "ERROR: Server not running at $URL"
    echo "Start it with: ./run.sh"
    exit 1
fi

echo "[1/3] Short generation test..."
START=$(date +%s%N)
RESULT=$(curl -s "$URL" \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Write hello world in Python. Be brief."}],
        "temperature": 0.6,
        "max_tokens": 128
    }')
END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))

TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "?")
echo "  Response tokens: $TOKENS"
echo "  Wall time: ${ELAPSED}ms"
if [ "$TOKENS" != "?" ] && [ "$TOKENS" -gt 0 ]; then
    RATE=$(python3 -c "print(f'{$TOKENS / ($ELAPSED / 1000):.1f}')" 2>/dev/null)
    echo "  Effective rate: ${RATE} tok/s"
fi
echo ""

echo "[2/3] Longer generation test..."
START=$(date +%s%N)
RESULT=$(curl -s "$URL" \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Explain how a B-tree works, step by step. Include insertion and deletion. Be thorough but concise."}],
        "temperature": 0.6,
        "max_tokens": 512
    }')
END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))

TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "?")
echo "  Response tokens: $TOKENS"
echo "  Wall time: ${ELAPSED}ms"
if [ "$TOKENS" != "?" ] && [ "$TOKENS" -gt 0 ]; then
    RATE=$(python3 -c "print(f'{$TOKENS / ($ELAPSED / 1000):.1f}')" 2>/dev/null)
    echo "  Effective rate: ${RATE} tok/s"
fi
echo ""

echo "[3/3] Quality check..."
ANSWER=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:200])" 2>/dev/null)
echo "  Preview: $ANSWER..."
echo ""

echo "[VRAM] $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
echo ""
echo "=== Test Complete ==="
