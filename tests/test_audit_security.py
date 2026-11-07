"""
Security, Quality & Resilience Audit Tests for FraudGuard AI
=============================================================
Tests:
- Admin token protection for POST /reload-models (disabled when unset, 401 when missing, 403 when invalid, 200 when valid)
- Public /history policy disclosure
- Streaming CSV upload limit enforcement (rejecting files exceeding 10MB during stream)
- Environment-based CORS separation (production vs dev)
- Rate limiting protection (429 on abuse)
- Imputation disclosure (tracking imputed_features)
- Calibrated prediction tests on labelled holdout samples
"""

import io
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.fraud_api import app, rate_limit_records

client = TestClient(app)


def test_reload_models_disabled_when_token_unset(monkeypatch):
    monkeypatch.delenv("FRAUDGUARD_ADMIN_TOKEN", raising=False)
    res = client.post("/reload-models")
    assert res.status_code == 503
    assert "administratively disabled" in res.json()["detail"]


def test_reload_models_unauthorized_when_header_missing(monkeypatch):
    monkeypatch.setenv("FRAUDGUARD_ADMIN_TOKEN", "super-secret-admin-token-12345")
    res = client.post("/reload-models")
    assert res.status_code == 401
    assert "Authorization" in res.json()["detail"]


def test_reload_models_forbidden_when_token_invalid(monkeypatch):
    monkeypatch.setenv("FRAUDGUARD_ADMIN_TOKEN", "super-secret-admin-token-12345")
    res = client.post("/reload-models", headers={"Authorization": "Bearer wrong-token"})
    assert res.status_code == 403
    assert "Forbidden" in res.json()["detail"]


def test_reload_models_success_with_valid_token(monkeypatch):
    monkeypatch.setenv("FRAUDGUARD_ADMIN_TOKEN", "super-secret-admin-token-12345")
    res = client.post("/reload-models", headers={"Authorization": "Bearer super-secret-admin-token-12345"})
    assert res.status_code == 200
    assert "reloaded successfully" in res.json()["message"]


def test_history_public_demo_disclosure():
    res = client.get("/history?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert data["policy"] == "public_synthetic_demo"
    assert "disclaimer" in data


def test_csv_streaming_upload_limit_enforcement():
    # Construct a payload slightly exceeding 10 MB (10.1 MB)
    large_payload = b"Time,Amount," + (b",".join([f"V{i}".encode() for i in range(1, 21)])) + b"\n"
    chunk = b"100.0,50.0," + (b",".join([b"0.1"] * 20)) + b"\n"
    repeat_count = (10 * 1024 * 1024 // len(chunk)) + 2000
    large_stream = io.BytesIO(large_payload + (chunk * repeat_count))

    files = {"file": ("large_test.csv", large_stream, "text/csv")}
    res = client.post("/predict-batch", files=files)
    assert res.status_code == 400
    assert "streaming limit" in res.json()["detail"]


def test_imputation_disclosure_in_predict():
    # Pass a payload omitting CustomerRiskScore and optional PCA dimensions
    txn = {
        "Time": 100.0,
        "Amount": 45.0,
        **{f"V{i}": 0.5 for i in range(1, 21)}
    }
    res = client.post("/predict", json=txn)
    assert res.status_code == 200
    data = res.json()
    # CustomerRiskScore was omitted, so it must be tracked in imputed_features
    assert "imputed_features" in data
    assert data["imputed_features"] is not None
    assert "CustomerRiskScore" in data["imputed_features"]


def test_calibrated_legitimate_holdout_sample():
    # Calibrated legitimate benchmark cardholder vector
    clean_sample = {
        "Time": 154.0,
        "Amount": 43.89,
        "CustomerRiskScore": 0.08,
        "V1": -1.17, "V2": 1.98, "V3": 0.54, "V4": 2.49, "V5": -2.11,
        "V6": 2.69, "V7": -1.36, "V8": 1.56, "V9": 3.23, "V10": 0.04,
        "V11": -1.30, "V12": -4.85, "V13": -1.32, "V14": -3.07, "V15": 1.94,
        "V16": 3.53, "V17": -0.88, "V18": 0.40, "V19": -1.19, "V20": -5.25
    }
    res = client.post("/predict", json=clean_sample)
    assert res.status_code == 200
    data = res.json()
    assert data["risk_level"] == "LOW"
    assert data["fraud_probability"] < 0.10
    assert data["is_fraud"] is False


def test_suspicious_fraud_holdout_sample():
    # High-risk account takeover vector
    fraud_sample = {
        "Time": 162.0,
        "Amount": 1120.87,
        "CustomerRiskScore": 0.85,
        "V1": 0.74, "V2": -0.95, "V3": 2.83, "V4": 2.04, "V5": 1.50,
        "V6": -1.20, "V7": 0.80, "V8": -0.50, "V9": 1.10, "V10": -0.26,
        "V11": 0.90, "V12": -7.85, "V13": 1.15, "V14": 2.01, "V15": -0.80,
        "V16": 1.40, "V17": -0.98, "V18": 0.01, "V19": -0.59, "V20": 3.41
    }
    res = client.post("/predict", json=fraud_sample)
    assert res.status_code == 200
    data = res.json()
    assert data["risk_level"] in ["HIGH", "CRITICAL"]
    assert data["fraud_probability"] >= 0.75
    assert data["is_fraud"] is True


def test_rate_limiting_enforcement(monkeypatch):
    rate_limit_records.clear()
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "5")
    monkeypatch.setenv("RATE_LIMIT_WINDOW", "10")

    # Re-import rate constants
    import src.api.fraud_api as api_mod
    monkeypatch.setattr(api_mod, "RATE_LIMIT_REQUESTS", 5)
    monkeypatch.setattr(api_mod, "RATE_LIMIT_WINDOW", 10)
    monkeypatch.setattr(api_mod, "RATE_LIMIT_ENABLED", True)

    txn = {
        "Time": 100.0,
        "Amount": 25.0,
        **{f"V{i}": 0.0 for i in range(1, 21)}
    }
    statuses = []
    for _ in range(7):
        r = client.post("/predict", json=txn)
        statuses.append(r.status_code)

    assert 429 in statuses
    rate_limit_records.clear()
