"""A tiny load test: N requests, C at a time, reporting latency percentiles.

    python scripts/load_test.py --requests 500 --concurrency 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

ROW = [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0]


def one(url: str) -> float:
    body = json.dumps({"features": ROW}).encode()
    request = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()
    return (time.perf_counter() - start) * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8000/predict")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()

    latencies: list[float] = []
    lock = threading.Lock()
    per_thread = max(1, args.requests // args.concurrency)

    def worker() -> None:
        local = [one(args.url) for _ in range(per_thread)]
        with lock:
            latencies.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - start

    latencies.sort()
    pct = lambda p: latencies[min(len(latencies) - 1, int(len(latencies) * p / 100))]  # noqa: E731
    print(f"{len(latencies)} requests, {args.concurrency} concurrent, {total:.2f}s "
          f"({len(latencies) / total:.0f} req/s)")
    print(f"mean {statistics.mean(latencies):.1f} ms   p50 {pct(50):.1f}   p95 {pct(95):.1f}   "
          f"p99 {pct(99):.1f}   max {latencies[-1]:.1f}")


if __name__ == "__main__":
    main()
