"""
Benchmark and measure model inference latency with statistical rigor.
Uses 100 warm-up runs followed by 1,000 single-record prediction iterations.
Reports Mean, Std, P50, P90, P95, and P99 latency percentiles in milliseconds.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

def measure_latency():
    model_path = "models/best_model.joblib"
    scaler_path = "models/scaler.joblib"
    features_path = "models/feature_names.json"

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        sys.exit(1)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path, "r") as f:
        feature_names = json.load(f)

    # Use a DataFrame with proper column names to prevent UserWarning
    sample_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    sample_scaled = scaler.transform(sample_df)

    # 1. Warm-up runs (100 iterations)
    for _ in range(100):
        _ = model.predict_proba(sample_scaled)

    # 2. Timed single-sample predictions (1,000 iterations)
    latencies_single_raw = []
    for _ in range(1000):
        t0 = time.perf_counter_ns()
        _ = model.predict_proba(sample_scaled)
        t1 = time.perf_counter_ns()
        latencies_single_raw.append((t1 - t0) / 1e6)  # Convert ns to ms

    latencies_single = np.array(latencies_single_raw)

    # 3. Timed end-to-end single-sample inference (preprocessing + prediction)
    latencies_e2e_raw = []
    for _ in range(1000):
        t0 = time.perf_counter_ns()
        scaled = scaler.transform(sample_df)
        _ = model.predict_proba(scaled)
        t1 = time.perf_counter_ns()
        latencies_e2e_raw.append((t1 - t0) / 1e6)

    latencies_e2e = np.array(latencies_e2e_raw)

    # 4. Timed 100-sample batch predictions (500 iterations)
    batch_df = pd.DataFrame(np.zeros((100, len(feature_names))), columns=feature_names)
    batch_scaled = scaler.transform(batch_df)

    latencies_batch_raw = []
    for _ in range(500):
        t0 = time.perf_counter_ns()
        _ = model.predict_proba(batch_scaled)
        t1 = time.perf_counter_ns()
        latencies_batch_raw.append((t1 - t0) / 1e6)

    latencies_batch = np.array(latencies_batch_raw)

    latency_report = {
        "timing_methodology": "100 warm-up runs followed by 1,000 repeated iterations using time.perf_counter_ns() on single-record scaled feature vector",
        "single_prediction_ms": {
            "mean": round(float(np.mean(latencies_single)), 4),
            "std": round(float(np.std(latencies_single)), 4),
            "p50": round(float(np.percentile(latencies_single, 50)), 4),
            "p90": round(float(np.percentile(latencies_single, 90)), 4),
            "p95": round(float(np.percentile(latencies_single, 95)), 4),
            "p99": round(float(np.percentile(latencies_single, 99)), 4),
            "min": round(float(np.min(latencies_single)), 4),
            "max": round(float(np.max(latencies_single)), 4),
            "samples_tested": 1000
        },
        "end_to_end_single_ms": {
            "mean": round(float(np.mean(latencies_e2e)), 4),
            "p50": round(float(np.percentile(latencies_e2e, 50)), 4),
            "p95": round(float(np.percentile(latencies_e2e, 95)), 4),
            "p99": round(float(np.percentile(latencies_e2e, 99)), 4)
        },
        "batch_100_ms": {
            "mean_total": round(float(np.mean(latencies_batch)), 4),
            "mean_per_sample": round(float(np.mean(latencies_batch) / 100), 5),
            "p50_total": round(float(np.percentile(latencies_batch, 50)), 4),
            "p95_total": round(float(np.percentile(latencies_batch, 95)), 4),
            "p99_total": round(float(np.percentile(latencies_batch, 99)), 4)
        }
    }

    print(json.dumps(latency_report, indent=2))
    return latency_report

if __name__ == "__main__":
    measure_latency()
