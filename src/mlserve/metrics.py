"""In-process metrics, exported in Prometheus text format.

Deliberately dependency-free: counters, an error tally by type, and a latency
histogram with the buckets an SLO is written against.
"""

from __future__ import annotations

import threading
from collections import Counter

BUCKETS_MS = (1, 5, 10, 25, 50, 100, 250, 500, 1000)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = Counter()
        self.errors = Counter()
        self.predictions = Counter()
        self.latency_buckets = Counter()
        self.latency_sum_ms = 0.0
        self.latency_count = 0

    def observe(self, endpoint: str, latency_ms: float, *, count: int = 1, label: str | None = None) -> None:
        with self._lock:
            self.requests[endpoint] += 1
            self.latency_count += count
            self.latency_sum_ms += latency_ms
            if label:
                self.predictions[label] += count
            for bucket in BUCKETS_MS:
                if latency_ms <= bucket:
                    self.latency_buckets[bucket] += 1

    def record_error(self, kind: str) -> None:
        with self._lock:
            self.errors[kind] += 1

    @property
    def average_latency_ms(self) -> float:
        return self.latency_sum_ms / self.latency_count if self.latency_count else 0.0

    def snapshot(self) -> dict:
        return {
            "requests": dict(self.requests),
            "errors": dict(self.errors),
            "predictions_by_label": dict(self.predictions),
            "average_latency_ms": round(self.average_latency_ms, 3),
            "total_predictions": self.latency_count,
        }

    def prometheus(self) -> str:
        lines = [
            "# HELP mlserve_requests_total Requests handled, by endpoint.",
            "# TYPE mlserve_requests_total counter",
        ]
        for endpoint, n in sorted(self.requests.items()):
            lines.append(f'mlserve_requests_total{{endpoint="{endpoint}"}} {n}')
        lines += ["# HELP mlserve_errors_total Failed requests, by reason.",
                  "# TYPE mlserve_errors_total counter"]
        for kind, n in sorted(self.errors.items()):
            lines.append(f'mlserve_errors_total{{reason="{kind}"}} {n}')
        lines += ["# HELP mlserve_predictions_total Predictions, by predicted label.",
                  "# TYPE mlserve_predictions_total counter"]
        for label, n in sorted(self.predictions.items()):
            lines.append(f'mlserve_predictions_total{{label="{label}"}} {n}')
        lines += ["# HELP mlserve_latency_ms Request latency.", "# TYPE mlserve_latency_ms histogram"]
        cumulative = 0
        for bucket in BUCKETS_MS:
            cumulative = self.latency_buckets.get(bucket, 0)
            lines.append(f'mlserve_latency_ms_bucket{{le="{bucket}"}} {cumulative}')
        lines.append(f'mlserve_latency_ms_bucket{{le="+Inf"}} {sum(self.requests.values())}')
        lines.append(f"mlserve_latency_ms_sum {self.latency_sum_ms:.3f}")
        lines.append(f"mlserve_latency_ms_count {sum(self.requests.values())}")
        return "\n".join(lines) + "\n"


metrics = Metrics()
