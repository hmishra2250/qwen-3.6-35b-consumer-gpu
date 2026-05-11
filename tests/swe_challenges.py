#!/usr/bin/env python3
"""
Hard SWE/Coding challenges for KV cache quantization quality comparison.
Sends prompts to a running llama-server or external API, extracts code, runs test cases, scores results.

Usage:
    python3 tests/swe_challenges.py [label]
    # label is e.g. "q4_0" or "q8_0" — used for result filenames

    # External API (e.g. Gemini via OpenAI-compatible endpoint):
    API_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions \
    API_KEY=your-key API_MODEL=gemma-4-31b-it python3 tests/swe_challenges.py gemma-31b
"""

import json, subprocess, sys, time, textwrap, re, os, tempfile, requests

API_URL = os.environ.get("API_URL",
    f"http://127.0.0.1:{os.environ.get('PORT', '8080')}/v1/chat/completions")
API_KEY = os.environ.get("API_KEY", "")
API_MODEL = os.environ.get("API_MODEL", "")
API_TEMP = float(os.environ.get("API_TEMP", "0.0"))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CHALLENGES = [
    {
        "id": "C01_range_sum",
        "name": "Count of Range Sum (Merge Sort)",
        "prompt": textwrap.dedent("""\
            Implement `count_range_sum(nums: list[int], lower: int, upper: int) -> int` that returns the number of range sums S(i,j) = nums[i]+...+nums[j] (0<=i<=j<len(nums)) in [lower, upper] inclusive.

            Must run in O(n log n) using prefix sums + modified merge sort.

            Return ONLY the Python function, no explanation. Include all necessary imports."""),
        "test_code": textwrap.dedent("""\
            # Brute force reference
            def _bf(nums, lo, hi):
                c = 0
                for i in range(len(nums)):
                    s = 0
                    for j in range(i, len(nums)):
                        s += nums[j]
                        if lo <= s <= hi: c += 1
                return c

            assert count_range_sum([-2, 5, -1], -2, 2) == 3
            assert count_range_sum([0], 0, 0) == 1
            assert count_range_sum([1, -1, 0], 0, 0) == 3
            assert count_range_sum([-2147483647, 0, -2147483647, 2147483647], -564, 3864) == 3

            import random
            random.seed(42)
            for _ in range(50):
                n = random.randint(1, 100)
                nums = [random.randint(-50, 50) for _ in range(n)]
                lo, hi = sorted(random.sample(range(-200, 200), 2))
                assert count_range_sum(nums, lo, hi) == _bf(nums, lo, hi), f"Failed on {nums}, {lo}, {hi}"
            print("PASS")
        """),
    },
    {
        "id": "C02_burst_balloons",
        "name": "Burst Balloons (Interval DP)",
        "prompt": textwrap.dedent("""\
            Implement `max_coins(nums: list[int]) -> int`.

            You have n balloons with values nums[i]. Bursting balloon i gives nums[i-1]*nums[i]*nums[i+1] coins (out-of-bounds = 1). Return maximum coins from bursting all.

            Use bottom-up O(n^3) interval DP. Key insight: think about which balloon to burst LAST in each interval.

            Return ONLY the Python function, no explanation."""),
        "test_code": textwrap.dedent("""\
            assert max_coins([3, 1, 5, 8]) == 167
            assert max_coins([1, 5]) == 10
            assert max_coins([7]) == 7
            assert max_coins([3, 1, 5]) == 35
            assert max_coins([1, 1, 1, 1, 1]) == 5
            assert max_coins([9, 76, 64, 21, 97, 60]) == 1086136
            assert max_coins([]) == 0
            assert max_coins([1]) == 1
            assert max_coins([2, 3]) == 9
            print("PASS")
        """),
    },
    {
        "id": "C03_matrix_fib_pisano",
        "name": "Matrix Fibonacci + Pisano Period",
        "prompt": textwrap.dedent("""\
            Implement these four functions where each builds on the previous:

            1. `matrix_mult(A, B)` — multiply two 2x2 integer matrices. No numpy.
            2. `matrix_power(M, n)` — compute M^n using fast exponentiation O(log n). Must call matrix_mult.
            3. `fibonacci(n)` — return n-th Fibonacci (fib(0)=0, fib(1)=1). Must use matrix_power with [[1,1],[1,0]]. Handle n=0.
            4. `pisano_period(m)` — find the Pisano period pi(m): smallest k>0 where fib(k)%m==0 and fib(k+1)%m==1. Must work efficiently using modular matrix exponentiation internally (don't compute huge fib numbers then mod).

            Return ONLY the Python functions, no explanation."""),
        "test_code": textwrap.dedent("""\
            assert fibonacci(0) == 0
            assert fibonacci(1) == 1
            assert fibonacci(10) == 55
            assert fibonacci(50) == 12586269025
            assert fibonacci(100) == 354224848179261915075

            assert pisano_period(2) == 3
            assert pisano_period(3) == 8
            assert pisano_period(10) == 60
            assert pisano_period(100) == 300

            # Cross-check fibonacci
            def fib_iter(n):
                a, b = 0, 1
                for _ in range(n): a, b = b, a + b
                return a
            for i in range(80):
                assert fibonacci(i) == fib_iter(i), f"fibonacci({i}) wrong"

            # Verify pisano property
            for m in [2, 3, 5, 7, 10, 100]:
                p = pisano_period(m)
                assert fib_iter(p) % m == 0, f"fib({p}) % {m} != 0"
                assert fib_iter(p + 1) % m == 1, f"fib({p}+1) % {m} != 1"
            print("PASS")
        """),
    },
    {
        "id": "C04_tree_serialize",
        "name": "Serialize/Deserialize Binary Tree",
        "prompt": textwrap.dedent("""\
            Implement serialization and deserialization of a binary tree.

            class TreeNode:
                def __init__(self, val="", left=None, right=None):
                    self.val = val
                    self.left = left
                    self.right = right

            def serialize(root) -> str:
                \"\"\"Encode tree to string.\"\"\"

            def deserialize(data) -> 'TreeNode | None':
                \"\"\"Decode string to tree.\"\"\"

            Requirements:
            - Node values can contain ANY characters: commas, brackets, quotes, backslashes, newlines, null bytes
            - Must handle trees with 5000+ nodes without hitting recursion limit (use iterative approach)
            - Must handle None (empty tree)
            - Round-trip: deserialize(serialize(tree)) must produce identical tree

            Return ONLY the class and functions, no explanation. Include the TreeNode class."""),
        "test_code": textwrap.dedent("""\
            def trees_equal(a, b):
                if a is None and b is None: return True
                if a is None or b is None: return False
                return a.val == b.val and trees_equal(a.left, b.left) and trees_equal(a.right, b.right)

            # Empty tree
            assert deserialize(serialize(None)) is None

            # Single node
            t = TreeNode("hello")
            assert trees_equal(deserialize(serialize(t)), t)

            # Special characters
            import sys
            sys.setrecursionlimit(10000)
            for val in ["", "hello", "a,b", "a\\\\b", 'a"b', "a\\nb", "a\\x00b", "[[]]", "null"]:
                node = TreeNode(val)
                restored = deserialize(serialize(node))
                assert restored is not None and restored.val == val, f"Failed on val={repr(val)}, got {repr(restored.val if restored else None)}"

            # Full tree depth 4
            def make_full(depth, prefix=""):
                if depth == 0: return None
                n = TreeNode(f"{prefix}d{depth}")
                n.left = make_full(depth - 1, prefix + "L")
                n.right = make_full(depth - 1, prefix + "R")
                return n
            t4 = make_full(4)
            assert trees_equal(deserialize(serialize(t4)), t4)

            # Deep skewed tree (test iterative approach)
            root = TreeNode("0")
            curr = root
            for i in range(1, 2000):
                curr.left = TreeNode(str(i))
                curr = curr.left
            s = serialize(root)
            restored = deserialize(s)
            curr1, curr2 = root, restored
            for i in range(2000):
                assert curr1.val == curr2.val, f"Mismatch at depth {i}"
                curr1, curr2 = curr1.left, curr2.left
            print("PASS")
        """),
    },
    {
        "id": "C05_topo_sort",
        "name": "All Topological Sorts + Longest Path",
        "prompt": textwrap.dedent("""\
            Implement a directed graph class:

            class DirectedGraph:
                def __init__(self, n: int): ...
                def add_edge(self, u: int, v: int) -> None: ...
                def has_cycle(self) -> bool: ...
                def topological_sort(self) -> list[int] | None:
                    \"\"\"One valid ordering, or None if cycle.\"\"\"
                def all_topological_sorts(self) -> list[list[int]]:
                    \"\"\"ALL valid orderings in lexicographic order. [] if cycle.\"\"\"
                def longest_path(self) -> int:
                    \"\"\"Longest path length (edges). -1 if cycle.\"\"\"

            Return ONLY the Python class, no explanation."""),
        "test_code": textwrap.dedent("""\
            # Diamond DAG
            g = DirectedGraph(4)
            g.add_edge(0, 1); g.add_edge(0, 2); g.add_edge(1, 3); g.add_edge(2, 3)
            assert not g.has_cycle()
            ts = g.topological_sort()
            assert ts is not None and ts[0] == 0 and ts[-1] == 3
            ats = g.all_topological_sorts()
            assert ats == [[0, 1, 2, 3], [0, 2, 1, 3]], f"Got {ats}"
            assert g.longest_path() == 2

            # Cycle
            g2 = DirectedGraph(3)
            g2.add_edge(0, 1); g2.add_edge(1, 2); g2.add_edge(2, 0)
            assert g2.has_cycle()
            assert g2.topological_sort() is None
            assert g2.all_topological_sorts() == []
            assert g2.longest_path() == -1

            # Self-loop
            g3 = DirectedGraph(2)
            g3.add_edge(0, 0)
            assert g3.has_cycle()

            # Linear chain
            g5 = DirectedGraph(5)
            for i in range(4): g5.add_edge(i, i + 1)
            assert g5.all_topological_sorts() == [[0, 1, 2, 3, 4]]
            assert g5.longest_path() == 4

            # Empty graph
            g6 = DirectedGraph(3)
            assert len(g6.all_topological_sorts()) == 6  # 3!
            assert g6.longest_path() == 0
            print("PASS")
        """),
    },
    {
        "id": "C06_calendar",
        "name": "Calendar Interval Merging with Priority",
        "prompt": textwrap.dedent("""\
            Implement a calendar with priority-based conflict resolution:

            from dataclasses import dataclass
            from typing import Any, Optional

            @dataclass
            class Event:
                start: float
                end: float
                title: str
                priority: int  # higher = more important

            class Calendar:
                def __init__(self): ...

                def add_event(self, event: Event) -> list[Event]:
                    \"\"\"Add event. If conflicts:
                    - Higher priority new event: trim/split existing events, return modified/removed.
                    - Equal/lower priority: trim new event around existing, return fragments added (may be empty).
                    Raise ValueError if start >= end.\"\"\"

                def get_events(self, start: float, end: float) -> list[Event]:
                    \"\"\"Events overlapping [start, end), sorted by start.\"\"\"

                def free_slots(self, start: float, end: float, min_duration: float = 0) -> list[tuple[float, float]]:
                    \"\"\"Free slots in [start, end) at least min_duration long.\"\"\"

                def utilization(self, start: float, end: float) -> float:
                    \"\"\"Fraction of [start, end) occupied (0.0-1.0).\"\"\"

            Return ONLY the Python code (dataclass + class), no explanation."""),
        "test_code": textwrap.dedent("""\
            cal = Calendar()
            cal.add_event(Event(100, 200, "A", 1))
            cal.add_event(Event(300, 400, "B", 1))
            assert len(cal.get_events(0, 500)) == 2

            # Higher priority overwrites
            cal.add_event(Event(150, 350, "C", 5))
            events = cal.get_events(0, 500)
            assert any(e.title == "A" and e.end == 150 for e in events), f"A not trimmed: {events}"
            assert any(e.title == "C" and e.start == 150 and e.end == 350 for e in events)
            assert any(e.title == "B" and e.start == 350 for e in events)

            # Lower priority gets trimmed to nothing
            fragments = cal.add_event(Event(140, 360, "D", 1))
            assert fragments == [] or all(f.end - f.start <= 0 for f in fragments), f"D should be empty: {fragments}"

            # Free slots
            cal2 = Calendar()
            cal2.add_event(Event(100, 200, "X", 1))
            cal2.add_event(Event(300, 400, "Y", 1))
            free = cal2.free_slots(0, 500)
            assert free == [(0, 100), (200, 300), (400, 500)], f"Got {free}"
            assert cal2.free_slots(0, 500, min_duration=150) == [(200, 300)]

            # Utilization
            assert abs(cal2.utilization(0, 500) - 0.4) < 0.001

            # Higher priority splits existing
            cal4 = Calendar()
            cal4.add_event(Event(100, 400, "Wide", 1))
            cal4.add_event(Event(200, 300, "Hi", 5))
            events = cal4.get_events(0, 500)
            wide_parts = [e for e in events if e.title == "Wide"]
            assert len(wide_parts) == 2, f"Wide should be split: {events}"

            # Invalid
            try:
                cal.add_event(Event(200, 100, "Bad", 1))
                assert False
            except ValueError:
                pass
            print("PASS")
        """),
    },
    {
        "id": "C07_regex",
        "name": "Mini Regex Engine",
        "prompt": textwrap.dedent("""\
            Implement a regex engine:

            def regex_match(pattern: str, text: str) -> bool:
                \"\"\"Full match (anchored both ends). Supports:
                . (any char), * (0+), + (1+), ? (0 or 1),
                [abc] char classes, [^abc] negated, [a-z] ranges,
                (...) grouping, | alternation, \\\\ escaping.\"\"\"

            Build an NFA or use recursive descent. Must handle groups with quantifiers like (ab)+ correctly.

            Return ONLY the Python function(s), no explanation."""),
        "test_code": textwrap.dedent("""\
            # Basic
            assert regex_match("abc", "abc") == True
            assert regex_match("abc", "abcd") == False
            assert regex_match("a.c", "axc") == True
            assert regex_match("a.c", "ac") == False

            # Quantifiers
            assert regex_match("ab*c", "ac") == True
            assert regex_match("ab*c", "abbc") == True
            assert regex_match("ab+c", "ac") == False
            assert regex_match("ab+c", "abc") == True
            assert regex_match("ab?c", "ac") == True
            assert regex_match("ab?c", "abc") == True
            assert regex_match("ab?c", "abbc") == False

            # Char classes
            assert regex_match("[abc]", "b") == True
            assert regex_match("[abc]", "d") == False
            assert regex_match("[^abc]", "d") == True
            assert regex_match("[^abc]", "a") == False
            assert regex_match("[a-z]", "m") == True
            assert regex_match("[a-z]", "M") == False

            # Groups
            assert regex_match("(ab)+", "ababab") == True
            assert regex_match("(ab)+", "aba") == False

            # Alternation
            assert regex_match("a|b", "a") == True
            assert regex_match("a|b", "c") == False
            assert regex_match("(cat|dog)+", "catdog") == True
            assert regex_match("(cat|dog)+", "catdogx") == False

            # Escaping
            assert regex_match(r"a\\.b", "a.b") == True
            assert regex_match(r"a\\.b", "axb") == False

            # Edge cases
            assert regex_match("a*", "") == True
            assert regex_match(".*", "anything") == True
            print("PASS")
        """),
    },
    {
        "id": "C08_consistent_hash",
        "name": "Consistent Hash Ring",
        "prompt": textwrap.dedent("""\
            Implement a consistent hashing ring:

            class ConsistentHashRing:
                def __init__(self, num_virtual_nodes: int = 150): ...

                def add_node(self, node_id: str) -> None:
                    \"\"\"Add a physical node with virtual nodes.\"\"\"

                def remove_node(self, node_id: str) -> None:
                    \"\"\"Remove a physical node. Raise KeyError if not found.\"\"\"

                def get_node(self, key: str) -> str | None:
                    \"\"\"Node responsible for key (clockwise), or None if empty.\"\"\"

                def get_nodes_for_key(self, key: str, num_replicas: int = 3) -> list[str]:
                    \"\"\"num_replicas DISTINCT physical nodes clockwise. If fewer exist, return all.\"\"\"

            Use MD5 hashing, bisect for O(log n) lookup. Virtual nodes named f"{node_id}#v{i}".

            Return ONLY the Python class, no explanation. Include necessary imports."""),
        "test_code": textwrap.dedent("""\
            ring = ConsistentHashRing(num_virtual_nodes=100)

            assert ring.get_node("key1") is None
            assert ring.get_nodes_for_key("key1", 3) == []

            ring.add_node("server-1")
            assert ring.get_node("key1") == "server-1"

            ring.add_node("server-2")
            for key in [f"key-{i}" for i in range(100)]:
                assert ring.get_node(key) in ("server-1", "server-2")

            # Distribution should be roughly even
            counts = {"server-1": 0, "server-2": 0}
            for i in range(10000):
                counts[ring.get_node(f"item-{i}")] += 1
            ratio = min(counts.values()) / max(counts.values())
            assert ratio > 0.6, f"Distribution too skewed: {counts}, ratio={ratio}"

            ring.add_node("server-3")
            replicas = ring.get_nodes_for_key("important-key", 3)
            assert len(replicas) == 3
            assert len(set(replicas)) == 3

            # Fewer nodes than replicas
            ring2 = ConsistentHashRing(100)
            ring2.add_node("only")
            assert ring2.get_nodes_for_key("k", 3) == ["only"]

            # Consistent hashing property: adding node only moves keys TO new node
            ring3 = ConsistentHashRing(100)
            ring3.add_node("A"); ring3.add_node("B")
            keys = [f"k{i}" for i in range(2000)]
            before = {k: ring3.get_node(k) for k in keys}
            ring3.add_node("C")
            after = {k: ring3.get_node(k) for k in keys}
            for k in keys:
                if before[k] != after[k]:
                    assert after[k] == "C", f"Key {k} moved {before[k]}->{after[k]}, should only go to C"

            ring.remove_node("server-2")
            for key in [f"item-{i}" for i in range(100)]:
                assert ring.get_node(key) in ("server-1", "server-3")

            try:
                ring.remove_node("nonexistent")
                assert False
            except KeyError:
                pass
            print("PASS")
        """),
    },
    {
        "id": "C09_async_bugs",
        "name": "Find & Fix Async Queue Bugs",
        "prompt": textwrap.dedent("""\
            The following async code has AT LEAST 4 bugs. Find ALL bugs, explain each, and provide the COMPLETE corrected code.

            ```python
            import asyncio
            from collections import deque

            class AsyncBoundedQueue:
                def __init__(self, maxsize):
                    self.maxsize = maxsize
                    self.queue = deque()
                    self.not_full = asyncio.Event()
                    self.not_empty = asyncio.Event()
                    self.not_full.set()
                    self.closed = False

                async def put(self, item):
                    while len(self.queue) >= self.maxsize:
                        self.not_full.clear()
                        await self.not_full.wait()
                    self.queue.append(item)
                    self.not_empty.set()

                async def get(self):
                    while len(self.queue) == 0:
                        if self.closed: return None
                        self.not_empty.clear()
                        await self.not_empty.wait()
                    item = self.queue.popleft()
                    self.not_full.set()
                    return item

                def close(self):
                    self.closed = True
                    self.not_empty.set()

            async def producer(queue, items):
                for item in items:
                    await queue.put(item)
                queue.close()

            async def consumer(queue, results, consumer_id):
                while True:
                    item = await queue.get()
                    if item is None: break
                    await asyncio.sleep(0.001)
                    results.append((consumer_id, item * 2))

            async def main():
                queue = AsyncBoundedQueue(maxsize=3)
                results = []
                items = list(range(20))
                prod = asyncio.create_task(producer(queue, items))
                consumers = [asyncio.create_task(consumer(queue, results, i)) for i in range(3)]
                await prod
                await asyncio.gather(*consumers)
                assert len(results) == 20
                print(f"OK: {len(results)} items processed")

            asyncio.run(main())
            ```

            Return the bug list AND the complete corrected code that actually runs and passes the assertion."""),
        "test_code": textwrap.dedent("""\
            # The corrected code should have been provided by the model.
            # We just verify it runs without hanging (5 second timeout)
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("Code hung - bugs not fixed")
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)
            try:
                asyncio.run(main())
                signal.alarm(0)
                print("PASS")
            except TimeoutError:
                print("FAIL: Code hung")
                sys.exit(1)
            except Exception as e:
                print(f"FAIL: {e}")
                sys.exit(1)
        """),
    },
    {
        "id": "C10_lru_ttl",
        "name": "LRU Cache with TTL",
        "prompt": textwrap.dedent("""\
            Implement a thread-safe LRU cache with time-based expiration:

            class TTLCache:
                def __init__(self, max_size: int, ttl_seconds: float): ...
                def get(self, key: str): \"\"\"Return value or None if expired/missing. Updates access order.\"\"\"
                def put(self, key: str, value) -> None: \"\"\"Insert/update. Evict LRU if full. Reset TTL on update.\"\"\"
                def cleanup(self) -> int: \"\"\"Remove expired entries, return count removed.\"\"\"
                def size(self) -> int: \"\"\"Count of non-expired entries.\"\"\"

            Requirements:
            - O(1) get/put (OrderedDict or DLL+dict)
            - Thread-safe with threading.Lock
            - Use time.monotonic() not time.time()
            - Lazy expiry on access + explicit cleanup()
            - size() must not count expired entries

            Return ONLY the Python class, no explanation. Include imports."""),
        "test_code": textwrap.dedent("""\
            from unittest.mock import patch
            import threading

            # Basic LRU
            cache = TTLCache(max_size=2, ttl_seconds=60)
            cache.put("a", 1)
            cache.put("b", 2)
            cache.get("a")       # touch a
            cache.put("c", 3)    # evicts b (LRU)
            assert cache.get("a") == 1
            assert cache.get("b") is None
            assert cache.get("c") == 3

            # TTL expiration
            with patch('time.monotonic') as mt:
                mt.return_value = 1000.0
                c2 = TTLCache(max_size=10, ttl_seconds=5.0)
                c2.put("x", 42)
                mt.return_value = 1003.0
                assert c2.get("x") == 42
                mt.return_value = 1006.0
                assert c2.get("x") is None
                assert c2.size() == 0

            # Cleanup
            with patch('time.monotonic') as mt:
                mt.return_value = 2000.0
                c3 = TTLCache(max_size=10, ttl_seconds=5.0)
                c3.put("a", 1); c3.put("b", 2); c3.put("c", 3)
                mt.return_value = 2006.0
                removed = c3.cleanup()
                assert removed == 3
                assert c3.size() == 0

            # Thread safety
            c4 = TTLCache(max_size=100, ttl_seconds=10)
            errors = []
            def worker(tid):
                try:
                    for i in range(500):
                        c4.put(f"{tid}-{i}", i)
                        c4.get(f"{tid}-{i % 10}")
                except Exception as e:
                    errors.append(e)
            threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
            for t in threads: t.start()
            for t in threads: t.join()
            assert not errors, f"Thread errors: {errors}"
            print("PASS")
        """),
    },
]


RETRYABLE_CODES = {429, 500, 502, 503, 504}
RETRYABLE_STATUSES = {"UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL", "DEADLINE_EXCEEDED"}


def _is_retryable_error(d):
    if isinstance(d, list):
        d = d[0] if d else {}
    if not isinstance(d, dict):
        return True
    err = d.get("error", {})
    if isinstance(err, str):
        return True
    return err.get("code", 0) in RETRYABLE_CODES or \
           str(err.get("status", "")).upper() in RETRYABLE_STATUSES


def _err_msg(d):
    err = d.get("error", d)
    if isinstance(err, str):
        return err
    if isinstance(err, dict):
        return err.get("message", str(err))
    return str(err)


def send_prompt(prompt, max_tokens=16384, retries=5):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": API_TEMP,
        "max_tokens": max_tokens,
    }
    if API_MODEL:
        body["model"] = API_MODEL

    hdrs = {"Content-Type": "application/json"}
    if API_KEY:
        hdrs["Authorization"] = f"Bearer {API_KEY}"

    last_err = None
    for attempt in range(retries + 1):
        start = time.time()
        try:
            resp = requests.post(API_URL, json=body, headers=hdrs, timeout=900)
            elapsed = time.time() - start
        except (requests.Timeout, requests.ConnectionError) as e:
            elapsed = time.time() - start
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                wait = min(30 * (2 ** attempt), 300)
                print(f"    [retry {attempt+1}/{retries}] {last_err[:80]}, wait {wait}s...", flush=True)
                time.sleep(wait)
                continue
            return {"content": "", "reasoning": "", "tokens": 0,
                    "elapsed": elapsed, "error": last_err}

        if resp.status_code in RETRYABLE_CODES:
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if attempt < retries:
                wait = min(30 * (2 ** attempt), 300)
                print(f"    [retry {attempt+1}/{retries}] HTTP {resp.status_code}, wait {wait}s...", flush=True)
                time.sleep(wait)
                continue
            return {"content": "", "reasoning": "", "tokens": 0,
                    "elapsed": elapsed, "error": last_err}

        try:
            d = resp.json()
        except (json.JSONDecodeError, ValueError):
            last_err = f"Bad JSON: {resp.text[:200]}"
            if attempt < retries:
                wait = min(15 * (2 ** attempt), 300)
                print(f"    [retry {attempt+1}/{retries}] {last_err[:80]}, wait {wait}s...", flush=True)
                time.sleep(wait)
                continue
            return {"content": "", "reasoning": "", "tokens": 0,
                    "elapsed": elapsed, "error": last_err}

        if isinstance(d, list):
            d = d[0] if d else {}

        if "error" in d:
            if _is_retryable_error(d) and attempt < retries:
                last_err = _err_msg(d)
                wait = min(30 * (2 ** attempt), 300)
                print(f"    [retry {attempt+1}/{retries}] {last_err[:80]}, wait {wait}s...", flush=True)
                time.sleep(wait)
                continue
            return {"content": "", "reasoning": "", "tokens": 0,
                    "elapsed": elapsed, "error": _err_msg(d)}

        try:
            msg = d["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            last_err = f"Bad response shape: {str(d)[:200]}"
            if attempt < retries:
                wait = min(15 * (2 ** attempt), 300)
                print(f"    [retry {attempt+1}/{retries}] {last_err[:80]}, wait {wait}s...", flush=True)
                time.sleep(wait)
                continue
            return {"content": "", "reasoning": "", "tokens": 0,
                    "elapsed": elapsed, "error": last_err}

        usage = d.get("usage", {})
        return {
            "content": msg.get("content", ""),
            "reasoning": msg.get("reasoning_content", ""),
            "tokens": usage.get("completion_tokens", 0),
            "elapsed": elapsed,
        }

    return {"content": "", "reasoning": "", "tokens": 0, "elapsed": 0,
            "error": f"All {retries} retries exhausted: {last_err}"}


def extract_code(response_text):
    """Extract Python code from markdown code blocks or raw text."""
    text = re.sub(r"<thought>.*?</thought>", "", response_text, flags=re.DOTALL).strip()
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n\n".join(blocks)
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n\n".join(blocks)
    return text


def run_test(code, test_code, timeout=30):
    """Run extracted code + test cases in a subprocess. Return (passed, output)."""
    full_code = code + "\n\n" + test_code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        try:
            r = subprocess.run(
                ["python3", f.name],
                capture_output=True, text=True, timeout=timeout,
            )
            output = r.stdout + r.stderr
            passed = r.returncode == 0 and "PASS" in r.stdout
            return passed, output[-2000:]
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        finally:
            os.unlink(f.name)


def log(msg=""):
    print(msg, flush=True)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "test"
    results = []

    log(f"=== SWE Challenge Suite: {label} ===")
    log(f"Running {len(CHALLENGES)} challenges\n")

    for i, ch in enumerate(CHALLENGES):
        log(f"[{i+1}/{len(CHALLENGES)}] {ch['name']}...")

        resp = send_prompt(ch["prompt"])

        if "error" in resp:
            log(f"  API ERROR: {str(resp['error'])[:200]}")
            results.append({
                "id": ch["id"], "name": ch["name"], "tokens": 0,
                "elapsed": resp["elapsed"], "has_answer": False,
                "passed": False, "output": f"API error: {resp['error']}",
                "content": "",
            })
            continue

        content = resp["content"]
        visible = re.sub(r"<thought>.*?</thought>", "", content, flags=re.DOTALL).strip() if content else ""
        has_answer = bool(visible)

        if not has_answer:
            log(f"  {resp['tokens']} tok, {resp['elapsed']:.0f}s — NO ANSWER (all thinking)")
            results.append({
                "id": ch["id"],
                "name": ch["name"],
                "tokens": resp["tokens"],
                "elapsed": resp["elapsed"],
                "has_answer": False,
                "passed": False,
                "output": "All tokens consumed by thinking",
                "content": "",
                "reasoning_preview": resp.get("reasoning", "")[:500],
            })
            continue

        code = extract_code(content)
        passed, output = run_test(code, ch["test_code"])

        status = "PASS" if passed else "FAIL"
        log(f"  {resp['tokens']} tok, {resp['elapsed']:.0f}s — {status}")
        if not passed:
            log(f"  Output: {output[:300]}")

        results.append({
            "id": ch["id"],
            "name": ch["name"],
            "tokens": resp["tokens"],
            "elapsed": resp["elapsed"],
            "has_answer": True,
            "passed": passed,
            "output": output,
            "content": content,
        })

    # Summary
    total = len(results)
    answered = sum(1 for r in results if r["has_answer"])
    passed = sum(1 for r in results if r["passed"])

    log(f"\n{'='*50}")
    log(f"RESULTS: {passed}/{total} passed ({answered}/{total} answered)")
    log(f"{'='*50}")
    for r in results:
        status = "PASS" if r["passed"] else ("NO ANSWER" if not r["has_answer"] else "FAIL")
        log(f"  [{status:>9}] {r['name']}")

    outfile = os.path.join(RESULTS_DIR, f"swe_{label}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nFull results: {outfile}")

    return passed, total


if __name__ == "__main__":
    main()
