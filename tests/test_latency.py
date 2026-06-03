"""
Latency tests for the Iris Classification model.
Measures prediction speed untuk single sample, batch, dan average across runs.
Run with: pytest tests/test_latency.py -v -s
"""

import pickle
import time

import numpy as np
import pytest


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    """Load the trained model from disk."""
    with open("model/model.pkl", "rb") as f:
        return pickle.load(f)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def measure_latency_ms(func, *args, runs=1):
    """Measure execution time in milliseconds."""
    start = time.perf_counter()
    for _ in range(runs):
        func(*args)
    end = time.perf_counter()
    return (end - start) * 1000 / runs


# ─── Latency Tests ───────────────────────────────────────────────────────────

class TestSinglePredictionLatency:
    def test_single_prediction_under_100ms(self, model):
        """Single prediction harus selesai < 100ms"""
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        latency = measure_latency_ms(model.predict, sample)
        print(f"\n  Single prediction latency : {latency:.3f}ms")
        assert latency < 100, f"Terlalu lambat: {latency:.2f}ms (limit: 100ms)"

    def test_single_prediction_under_50ms(self, model):
        """Single prediction idealnya < 50ms"""
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        latency = measure_latency_ms(model.predict, sample)
        print(f"\n  Single prediction latency : {latency:.3f}ms")
        assert latency < 50, f"Latency {latency:.2f}ms melebihi 50ms"


class TestBatchPredictionLatency:
    def test_batch_100_under_200ms(self, model):
        """Batch 100 samples harus < 200ms"""
        batch = np.random.rand(100, 4)
        latency = measure_latency_ms(model.predict, batch)
        print(f"\n  Batch (100)  latency      : {latency:.3f}ms")
        assert latency < 200

    def test_batch_1000_under_500ms(self, model):
        """Batch 1000 samples harus < 500ms"""
        batch = np.random.rand(1000, 4)
        latency = measure_latency_ms(model.predict, batch)
        print(f"\n  Batch (1000) latency      : {latency:.3f}ms")
        assert latency < 500

    def test_batch_10000_under_2000ms(self, model):
        """Batch 10.000 samples harus < 2000ms"""
        batch = np.random.rand(10000, 4)
        latency = measure_latency_ms(model.predict, batch)
        print(f"\n  Batch (10k)  latency      : {latency:.3f}ms")
        assert latency < 2000


class TestAverageLatency:
    def test_average_latency_100_runs(self, model):
        """Average latency dari 100 runs harus < 10ms"""
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        latencies = []

        for _ in range(100):
            start = time.perf_counter()
            model.predict(sample)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        print(f"\n  ── Latency Stats (100 runs) ──")
        print(f"  Avg : {avg:.3f}ms")
        print(f"  P50 : {p50:.3f}ms")
        print(f"  P95 : {p95:.3f}ms")
        print(f"  P99 : {p99:.3f}ms")
        print(f"  Min : {min(latencies):.3f}ms")
        print(f"  Max : {max(latencies):.3f}ms")

        assert avg < 10, f"Average latency terlalu tinggi: {avg:.2f}ms"

    def test_p95_latency_under_20ms(self, model):
        """P95 latency harus < 20ms (95% request selesai dalam 20ms)"""
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        latencies = []

        for _ in range(100):
            start = time.perf_counter()
            model.predict(sample)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        p95 = np.percentile(latencies, 95)
        print(f"\n  P95 latency: {p95:.3f}ms")
        assert p95 < 20, f"P95 {p95:.2f}ms melebihi 20ms"
