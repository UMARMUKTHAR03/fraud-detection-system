"""
Expanded API test cases for FraudGuard AI
=========================================
Tests:
- GET /model-info
- GET /analytics
- GET /history (with filter & limit)
- POST /predict-batch (CSV upload multipart)
"""

import io
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.fraud_api import app

client = TestClient(app)


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["product"] == "FraudGuard AI"
    assert data["production_model"] == "XGBoost"
    assert "cross_validation" in data
    assert "holdout_evaluation" in data
    assert data["decision_threshold"] == 0.50
    assert "features" in data
    assert len(data["features"]) > 0


def test_analytics_endpoint():
    response = client.get("/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "model_comparison" in data
    assert "holdout_evaluation" in data
    assert "top_features" in data
    assert "roc_curve" in data
    assert "pr_curve" in data
    assert len(data["top_features"]) > 0


def test_history_endpoint():
    # Make a prediction first
    tx = {
        "Time": 420.0,
        "Amount": 1250.0,
        **{f"V{i}": -1.2 * (i % 3) for i in range(1, 21)},
        "transaction_id": "TEST_HIST_01",
    }
    client.post("/predict", json=tx)

    # Query history
    response = client.get("/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "transactions" in data
    assert isinstance(data["transactions"], list)


def test_predict_batch_csv_upload():
    # Create sample CSV
    csv_content = "Time,Amount," + ",".join([f"V{i}" for i in range(1, 21)]) + "\n"
    csv_content += "100.0,50.0," + ",".join(["0.1"] * 20) + "\n"
    csv_content += "200.0,4500.0," + ",".join(["-2.5"] * 20) + "\n"

    files = {"file": ("test_batch.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/predict-batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert "batch_summary" in data
    assert data["batch_summary"]["total_transactions"] == 2


def test_predict_batch_csv_download():
    csv_content = "Time,Amount," + ",".join([f"V{i}" for i in range(1, 21)]) + "\n"
    csv_content += "100.0,50.0," + ",".join(["0.1"] * 20) + "\n"
    files = {"file": ("test_batch.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/predict-batch?download=true", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert "attachment" in response.headers["content-disposition"]
    text = response.text
    assert "Transaction_ID" in text
    assert "Fraud_Probability" in text
    assert "Risk_Level" in text

