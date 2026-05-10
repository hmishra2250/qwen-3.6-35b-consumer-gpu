#!/bin/bash
# Speed benchmark: measures tok/s at different generation lengths
# Run against a running server
set -euo pipefail

PORT=${PORT:-8080}
HOST=${HOST:-127.0.0.1}
URL="http://$HOST:$PORT/v1/chat/completions"

if ! curl -s "http://$HOST:$PORT/health" | grep -q "ok"; then
    echo "ERROR: Server not running. Start with ./run.sh first."
    exit 1
fi

echo "=== Speed Benchmark ==="
echo "Server: $URL"
echo "GPU temp: $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null || echo '?')C"
echo ""

python3 -c "
import json, subprocess, time

URL = 'http://$HOST:$PORT/v1/chat/completions'

tests = [
    ('128 tokens', 'Write hello world in Python. Be brief.', 128),
    ('512 tokens', 'Explain how hash tables work with collision handling.', 512),
    ('1024 tokens', 'Explain how a B-tree works step by step including insertion, deletion, and rebalancing.', 1024),
    ('2048 tokens', 'Write a comprehensive guide to RESTful API design covering versioning, auth, pagination, error handling, rate limiting, and caching with examples.', 2048),
]

print(f\"{'Test':<15} {'Tokens':>7} {'Time':>8} {'Rate':>10}\")
print('-' * 45)

for name, prompt, max_tok in tests:
    payload = json.dumps({
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.6,
        'max_tokens': max_tok
    })
    start = time.time()
    r = subprocess.run(
        ['curl', '-s', URL, '-H', 'Content-Type: application/json', '-d', payload],
        capture_output=True, text=True, timeout=300
    )
    elapsed = time.time() - start
    d = json.loads(r.stdout)
    tokens = d['usage']['completion_tokens']
    rate = tokens / elapsed
    print(f'{name:<15} {tokens:>7} {elapsed:>7.1f}s {rate:>8.1f} t/s')
"

echo ""
echo "VRAM: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo '?')"
echo "GPU temp: $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null || echo '?')C"
