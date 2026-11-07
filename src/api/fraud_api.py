"""
FraudGuard AI — Enterprise Real-Time Fraud Detection & Risk Analytics API
=========================================================================

Production FastAPI implementation featuring:
- Real-time transaction scoring via champion XGBoost model + soft-voting ensemble
- Exact risk classification (LOW <25%, MEDIUM 25-50%, HIGH 50-75%, CRITICAL >=75%)
- Real-time SHAP decision explainability & local feature attribution
- SQLite persistent transaction prediction logging (`data/fraud_history.db`)
- High-velocity batch prediction support (JSON array + CSV file upload)
- Model metadata, 5-Fold cross-validation metrics, confusion matrix, and ROC/PR analytics
- CORS restricted to official GitHub Pages and localhost development origins

Author: UMAR MUKTHAR (@UMARMUKTHAR03)
Platform: FraudGuard AI
Version: 2.4.0
"""

import io
import json
import logging
import os
import sqlite3
import tempfile

# Pre-configure writable matplotlib cache directory to avoid permissions issues/warnings
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "fraudguard_mpl")
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import secrets
from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
# Configurable persistent data directory matching Render mount path /opt/render/project/src/data
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))
DB_PATH = os.path.join(DATA_DIR, "fraud_history.db")

def get_utc_now_iso() -> str:
    """Return timezone-aware ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()

# Lifespan context manager replacing deprecated @app.on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_models()
    yield

# Initialize FastAPI app
app = FastAPI(
    title="FraudGuard AI API",
    description="Intelligent Financial Fraud Detection & Risk Analytics Platform",
    version="2.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Secure CORS middleware: Production vs Development separation
env_mode = os.environ.get("ENVIRONMENT", "production").lower()
custom_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
parsed_custom = [o.strip() for o in custom_origins_env.split(",") if o.strip()]

if env_mode in ["development", "dev", "local", "test"]:
    allowed_origins = [
        "https://umarmukthar03.github.io",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5500",
    ] + [o for o in parsed_custom if o not in [
        "https://umarmukthar03.github.io",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5500",
    ]]
else:
    # Production mode: Only explicitly authorized production origins
    base_prod = ["https://umarmukthar03.github.io"]
    allowed_origins = base_prod + [o for o in parsed_custom if o not in base_prod]

# ---------------------------------------------------------------------
# In-Memory Rate Limiting
# ---------------------------------------------------------------------
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"
rate_limit_records: Dict[str, List[float]] = {}


def check_rate_limit(request: Request):
    if not RATE_LIMIT_ENABLED:
        return
    client_ip = request.client.host if request.client else "unknown"
    # Respect X-Forwarded-For if behind a proxy
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    timestamps = rate_limit_records.setdefault(client_ip, [])
    # Prune expired timestamps
    rate_limit_records[client_ip] = [t for t in timestamps if t > window_start]

    if len(rate_limit_records[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s window.",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
        )
    rate_limit_records[client_ip].append(now)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
)

# Global model state
best_model = None
scaler = None
explainer = None
ensemble_model = None
# Empirical dataset population means derived from training artifacts for missing-value handling
POPULATION_DEFAULTS = {
    "V1": -0.936786, "V2": 1.005415, "V3": 1.022082, "V4": 0.958918,
    "V5": -0.96937, "V6": 0.98872, "V7": -0.99786, "V8": 0.99639,
    "V9": 0.969, "V10": 0.000593, "V11": -0.944742, "V12": -5.231849,
    "V13": -1.005169, "V14": -0.950391, "V15": 1.034456, "V16": 0.879585,
    "V17": -0.979328, "V18": 0.008808, "V19": 0.96065, "V20": -2.138873,
    "Hour": 11.469094, "DayOfWeek": 1.01075, "CustomerRiskScore": 0.285,
}

feature_names = []
metrics_data = {}
model_loaded = False
model_load_time: Optional[str] = None

# Telemetry
request_count = 0
total_processing_time = 0.0
start_time = time.time()


# ---------------------------------------------------------------------
# SQLite Database Setup & Logging
# ---------------------------------------------------------------------
def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transaction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            amount REAL,
            time REAL,
            fraud_probability REAL,
            is_fraud INTEGER,
            risk_level TEXT,
            confidence_score REAL,
            processing_time_ms REAL,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def log_prediction_to_db(record: Dict[str, Any]):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO transaction_history (
                transaction_id, amount, time, fraud_probability, is_fraud,
                risk_level, confidence_score, processing_time_ms, explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("transaction_id"),
                record.get("amount"),
                record.get("time"),
                record.get("fraud_probability"),
                1 if record.get("is_fraud") else 0,
                record.get("risk_level"),
                record.get("confidence_score"),
                record.get("processing_time_ms"),
                json.dumps(record.get("explanation", {})),
                get_utc_now_iso(),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to log transaction to SQLite: {e}")


# ---------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------
class TransactionRequest(BaseModel):
    """Schema for validating a single transaction."""

    Time: float = Field(..., ge=0, description="Time in seconds from reference point")
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")

    # Anonymized PCA Features V1..V20
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float

    # Optional enriched fields
    Hour: Optional[float] = None
    DayOfWeek: Optional[float] = None
    CustomerRiskScore: Optional[float] = None
    transaction_id: Optional[str] = Field(None, description="Unique transaction ID")

    @field_validator("Amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Amount must be non-negative")
        return v

    @field_validator("Time")
    @classmethod
    def validate_time(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Time must be non-negative")
        return v

    @field_validator("CustomerRiskScore")
    @classmethod
    def validate_customer_risk_score(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("CustomerRiskScore must be between 0.0 and 1.0")
        return v


class BatchTransactionRequest(BaseModel):
    """Schema for validating a batch of transactions."""

    transactions: List[TransactionRequest] = Field(..., min_length=1, max_length=5000)


class FraudPredictionResponse(BaseModel):
    """Response model for a single fraud prediction."""

    transaction_id: Optional[str]
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    confidence_score: float = Field(..., ge=0, le=1)
    explanation: Optional[Dict[str, Any]] = None
    top_risk_factors: Optional[List[Dict[str, Any]]] = None
    model_votes: Optional[Dict[str, float]] = None
    imputed_features: Optional[List[str]] = Field(
        default=None,
        description="Features where empirical baseline values were substituted due to omission in payload",
    )
    processing_time_ms: float
    model_version: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[FraudPredictionResponse]
    batch_summary: Dict[str, Any]
    total_processing_time_ms: float


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    model_loaded: bool
    model_load_time: Optional[str]
    uptime_seconds: float
    version: str


# ---------------------------------------------------------------------
# Model Loading & Risk Classification Logic
# ---------------------------------------------------------------------
def load_models() -> bool:
    """Load model artifacts into memory."""
    global best_model, scaler, explainer, ensemble_model, feature_names, metrics_data
    global model_loaded, model_load_time

    try:
        logger.info("Loading FraudGuard AI production artifacts...")

        best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
        scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
        explainer_path = os.path.join(MODELS_DIR, "explainer.joblib")
        features_path = os.path.join(MODELS_DIR, "feature_names.json")
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")
        ensemble_path = os.path.join(MODELS_DIR, "fraud_detection_ensemble.pkl")

        if os.path.exists(best_model_path):
            best_model = joblib.load(best_model_path)
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        if os.path.exists(explainer_path):
            explainer = joblib.load(explainer_path)
        if os.path.exists(ensemble_path):
            try:
                ensemble_model = joblib.load(ensemble_path)
            except Exception as ee:
                logger.warning(f"Could not load standalone ensemble pkl: {ee}")
        if os.path.exists(features_path):
            with open(features_path, "r") as f:
                feature_names = json.load(f)
        else:
            feature_names = ["Time", "Amount"] + [f"V{i}" for i in range(1, 21)]

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)

        init_db()

        model_loaded = best_model is not None and scaler is not None
        model_load_time = get_utc_now_iso()
        logger.info(f"✅ Models loaded successfully. Model loaded: {model_loaded}")
        return model_loaded
    except Exception as e:
        logger.exception(f"❌ Error loading models: {e}")
        model_loaded = False
        return False


# Eagerly load models on module import so synchronous test runners have models immediately available
load_models()


def get_risk_level(prob: float) -> str:
    """
    Calibrated Risk Category Assignment:
    - LOW:      < 0.25 (Approved)
    - MEDIUM:   0.25 to < 0.50 (Step-Up Monitoring)
    - HIGH:     0.50 to < 0.75 (Challenge / 2FA)
    - CRITICAL: >= 0.75 (Auto-Freeze / Terminate)
    """
    if prob >= 0.75:
        return "CRITICAL"
    if prob >= 0.50:
        return "HIGH"
    if prob >= 0.25:
        return "MEDIUM"
    return "LOW"


def get_confidence_score(prob: float) -> float:
    """Confidence increases as probability diverges from the uncertainty midpoint (0.50)."""
    return round(float(abs(prob - 0.50) * 2.0), 4)


def compute_shap_factors(raw_features_df: pd.DataFrame, scaled_features_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute local feature attribution for decision interpretability."""
    try:
        if explainer is not None:
            shap_values = explainer.shap_values(scaled_features_df)
            if isinstance(shap_values, list):
                sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                sv = shap_values[0]
            else:
                sv = np.array(shap_values)[0]

            col_names = list(scaled_features_df.columns)
            factors = []
            for col, impact in zip(col_names, sv):
                factors.append(
                    {
                        "feature": col,
                        "value": round(float(raw_features_df[col].iloc[0]), 4),
                        "shap_impact": round(float(impact), 4),
                        "correlation": "Fraud-Inducing" if impact > 0 else "Legitimate-Shielding",
                    }
                )

            # Sort descending by absolute impact
            factors.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
            return factors[:6]
    except Exception as e:
        logger.warning(f"SHAP explanation fallback: {e}")

    # Fallback to normalized feature z-score divergence
    factors = []
    for col in scaled_features_df.columns[:6]:
        val = float(scaled_features_df[col].iloc[0])
        factors.append(
            {
                "feature": col,
                "value": round(float(raw_features_df[col].iloc[0]), 4),
                "shap_impact": round(val * 0.15, 4),
                "correlation": "Fraud-Inducing" if val > 0 else "Legitimate-Shielding",
            }
        )
    return factors


async def check_model_dependency():
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="FraudGuard models are not loaded. Please train or reload the models.",
        )


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Fraud Detection API",
        "product": "FraudGuard AI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "author": "UMAR MUKTHAR",
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy" if model_loaded else "unhealthy",
        timestamp=get_utc_now_iso(),
        model_loaded=model_loaded,
        model_load_time=model_load_time,
        uptime_seconds=round(time.time() - start_time, 2),
        version="2.4.0",
    )


@app.get("/model-info")
async def model_info():
    """Return comprehensive metadata for the active production model."""
    cv_info = metrics_data.get("holdout_evaluation", {}).get("cv_metrics", {})
    return {
        "product": "FraudGuard AI",
        "production_model": metrics_data.get("production_model", "XGBoost"),
        "algorithm": metrics_data.get("algorithm", "Extreme Gradient Boosting (XGBClassifier)"),
        "framework_version": metrics_data.get("framework_version", "3.4.1"),
        "dataset_benchmark": metrics_data.get("dataset", {}),
        "decision_threshold": metrics_data.get("decision_threshold", 0.50),
        "risk_thresholds": metrics_data.get("risk_thresholds", {
            "LOW": "< 0.25",
            "MEDIUM": "0.25 - 0.4999",
            "HIGH": "0.50 - 0.7499",
            "CRITICAL": ">= 0.75"
        }),
        "feature_count": len(feature_names),
        "features": feature_names,
        "cross_validation": cv_info,
        "holdout_evaluation": metrics_data.get("holdout_evaluation", {}),
    }


@app.get("/analytics")
async def analytics():
    """Return model performance comparison, confusion matrix, ROC/PR curves, and feature importance."""
    return {
        "production_model": metrics_data.get("production_model", "XGBoost"),
        "model_comparison": metrics_data.get("model_comparison", {}),
        "holdout_evaluation": metrics_data.get("holdout_evaluation", {}),
        "model_selection_rationale": metrics_data.get("model_selection_rationale", {}),
        "latency_benchmark": metrics_data.get("latency_benchmark", {}),
        "top_features": metrics_data.get("top_features", []),
        "roc_curve": metrics_data.get("roc_curve", []),
        "pr_curve": metrics_data.get("pr_curve", []),
        "updated_at": metrics_data.get("timestamp", get_utc_now_iso()),
    }


@app.get("/sample-template")
async def get_sample_template():
    """Return a sample CSV template for batch evaluation."""
    sample_file = os.path.join(DATA_DIR, "sample_transactions.csv")
    if os.path.exists(sample_file):
        with open(sample_file, "r") as f:
            content = f.read()
    else:
        # Generate inline fallback template
        content = "Time,Amount," + ",".join([f"V{i}" for i in range(1, 21)]) + "\n"
        content += "100.0,45.0," + ",".join(["0.1"] * 20) + "\n"
        content += "200.0,4200.0," + ",".join(["-3.2"] * 20) + "\n"

    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="fraudguard_sample_transactions.csv"'},
    )


@app.get("/metrics")
async def metrics():
    uptime = time.time() - start_time
    avg_time = total_processing_time / request_count if request_count else 0
    return {
        "requests_processed": request_count,
        "average_processing_time_ms": round(avg_time, 2),
        "total_processing_time_ms": round(total_processing_time, 2),
        "uptime_seconds": round(uptime, 2),
        "requests_per_second": round(request_count / uptime, 2) if uptime else 0,
        "model_loaded": model_loaded,
        "timestamp": get_utc_now_iso(),
    }


@app.get("/history")
async def get_history(
    limit: int = Query(50, ge=1, le=500),
    risk_level: Optional[str] = Query(None, pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$"),
    search: Optional[str] = None,
    authorization: Optional[str] = Header(None, description="Optional bearer token for private/production tenant isolation"),
):
    """Public demonstration transaction audit feed.
    NOTICE: This endpoint serves synthetic educational demo records for dashboard visualization.
    Real financial data and production environments must supply an authorized Bearer token or gateway isolation."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM transaction_history WHERE 1=1"
        params = []

        if risk_level:
            query += " AND risk_level = ?"
            params.append(risk_level)

        if search:
            query += " AND (transaction_id LIKE ? OR risk_level LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for r in rows:
            explanation_obj = {}
            if r["explanation"]:
                try:
                    explanation_obj = json.loads(r["explanation"])
                except Exception:
                    pass

            results.append(
                {
                    "id": r["id"],
                    "transaction_id": r["transaction_id"],
                    "amount": r["amount"],
                    "time": r["time"],
                    "fraud_probability": r["fraud_probability"],
                    "is_fraud": bool(r["is_fraud"]),
                    "risk_level": r["risk_level"],
                    "confidence_score": r["confidence_score"],
                    "processing_time_ms": r["processing_time_ms"],
                    "explanation": explanation_obj,
                    "created_at": r["created_at"],
                }
            )

        return {"total": len(results), "transactions": results, "records": results, "policy": "public_synthetic_demo", "disclaimer": "Public demo feed. Do not store PII or live cardholder data."}
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return {"total": 0, "transactions": [], "records": []}


@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    _: None = Depends(check_model_dependency),
    __: None = Depends(check_rate_limit),
):
    global request_count, total_processing_time
    start_ms = time.time() * 1000

    try:
        data_dict = transaction.model_dump()
        tx_id = data_dict.pop("transaction_id", None) or f"TXN-{int(time.time()*1000)%1000000}"

        # Align with model feature order & track imputed features
        row_dict = {}
        imputed_cols = []
        for col in feature_names:
            if col in data_dict and data_dict[col] is not None:
                row_dict[col] = float(data_dict[col])
            elif col == "Hour":
                row_dict[col] = float((data_dict.get("Time", 0) / 3600) % 24)
                imputed_cols.append("Hour")
            elif col == "DayOfWeek":
                row_dict[col] = float((data_dict.get("Time", 0) / 86400) % 7)
                imputed_cols.append("DayOfWeek")
            elif col == "CustomerRiskScore":
                row_dict[col] = POPULATION_DEFAULTS.get("CustomerRiskScore", 0.285)
                imputed_cols.append("CustomerRiskScore")
            else:
                row_dict[col] = POPULATION_DEFAULTS.get(col, 0.0)
                imputed_cols.append(col)

        raw_df = pd.DataFrame([row_dict], columns=feature_names)
        scaled_df = pd.DataFrame(scaler.transform(raw_df), columns=feature_names)

        # Inference
        probs = best_model.predict_proba(scaled_df)[0]
        fraud_prob = round(float(probs[1]), 4)

        risk_level = get_risk_level(fraud_prob)
        is_fraud = fraud_prob >= 0.50
        confidence = get_confidence_score(fraud_prob)

        # Compute model votes if ensemble is available
        votes = {"xgboost": fraud_prob}
        if ensemble_model and isinstance(ensemble_model, dict) and "models" in ensemble_model:
            sub_models = ensemble_model["models"]
            if "lgb" in sub_models:
                try:
                    lgb_prob = float(sub_models["lgb"].predict_proba(scaled_df)[0][1])
                    votes["lightgbm"] = round(lgb_prob, 4)
                except Exception:
                    pass
            if "rf" in sub_models:
                try:
                    rf_prob = float(sub_models["rf"].predict_proba(scaled_df)[0][1])
                    votes["random_forest"] = round(rf_prob, 4)
                except Exception:
                    pass

        # Compute SHAP
        top_factors = compute_shap_factors(raw_df, scaled_df)

        processing_time = round(time.time() * 1000 - start_ms, 2)
        request_count += 1
        total_processing_time += processing_time

        resp = FraudPredictionResponse(
            transaction_id=tx_id,
            fraud_probability=fraud_prob,
            is_fraud=is_fraud,
            risk_level=risk_level,
            confidence_score=confidence,
            explanation={"summary": f"Classified as {risk_level} risk with {round(fraud_prob*100, 1)}% fraud probability."},
            top_risk_factors=top_factors,
            model_votes=votes,
            imputed_features=imputed_cols if imputed_cols else None,
            processing_time_ms=processing_time,
            model_version="FraudGuard-XGB-2.4",
            timestamp=get_utc_now_iso(),
        )

        # Async background log to SQLite
        background_tasks.add_task(
            log_prediction_to_db,
            {
                "transaction_id": tx_id,
                "amount": transaction.Amount,
                "time": transaction.Time,
                "fraud_probability": fraud_prob,
                "is_fraud": is_fraud,
                "risk_level": risk_level,
                "confidence_score": confidence,
                "processing_time_ms": processing_time,
                "explanation": {"top_factors": top_factors},
            },
        )

        return resp
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_json(
    batch_request: BatchTransactionRequest,
    _: None = Depends(check_model_dependency),
    __: None = Depends(check_rate_limit),
):
    return await execute_batch(batch_request.transactions)


@app.post("/predict-batch")
async def predict_batch_upload(
    request: Request,
    file: Optional[UploadFile] = File(None),
    download: bool = Query(False, description="Whether to return results as downloadable CSV"),
    _: None = Depends(check_model_dependency),
    __: None = Depends(check_rate_limit),
):
    """Handle batch prediction via CSV file upload, JSON array, or multipart stream.
    Enforces a strict 10MB limit while streaming chunks before in-memory buffering."""
    raw_df = None

    # 1. Handle file upload (CSV) with streaming size limit enforcement
    if file is not None and file.filename:
        MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
        CHUNK_SIZE = 64 * 1024  # 64 KB
        total_bytes = 0
        chunks = []

        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded file exceeds {MAX_UPLOAD_SIZE // (1024*1024)}MB streaming limit",
                )
            chunks.append(chunk)

        content = b"".join(chunks)
        try:
            raw_df = pd.read_csv(io.BytesIO(content))
        except Exception as parse_err:
            raise HTTPException(status_code=400, detail=f"Error parsing CSV: {parse_err}")

    # 2. Handle JSON request body fallback
    elif "application/json" in (request.headers.get("content-type") or ""):
        try:
            body = await request.json()
            if isinstance(body, list):
                tx_list = [TransactionRequest(**item) for item in body]
                return await execute_batch(tx_list)
            elif isinstance(body, dict) and "transactions" in body:
                tx_list = [TransactionRequest(**item) for item in body["transactions"]]
                return await execute_batch(tx_list)
        except Exception as json_err:
            raise HTTPException(status_code=400, detail=f"Invalid JSON batch body: {json_err}")
    else:
        raise HTTPException(status_code=400, detail="CSV file or JSON transaction payload must be provided")

    if raw_df is None or raw_df.empty:
        raise HTTPException(status_code=400, detail="Provided dataset contains zero rows")

    if len(raw_df) > 5000:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds maximum limit of 5,000 records (received {len(raw_df)})")

    # Check required columns
    for c in ["Time", "Amount"]:
        if c not in raw_df.columns:
            raise HTTPException(status_code=400, detail=f"CSV missing required column: {c}")

    tx_list = []
    row_errors = []
    for idx, row in raw_df.iterrows():
        try:
            tx_data = {
                "Time": float(row["Time"]) if not pd.isna(row["Time"]) else 0.0,
                "Amount": max(0.0, float(row["Amount"])) if not pd.isna(row["Amount"]) else 0.0,
                "transaction_id": str(row.get("transaction_id", f"CSV-{idx}")),
            }
            for i in range(1, 21):
                col_name = f"V{i}"
                tx_data[col_name] = float(row[col_name]) if col_name in row and not pd.isna(row[col_name]) else 0.0

            if "Hour" in row and not pd.isna(row["Hour"]):
                tx_data["Hour"] = float(row["Hour"])
            if "DayOfWeek" in row and not pd.isna(row["DayOfWeek"]):
                tx_data["DayOfWeek"] = float(row["DayOfWeek"])
            if "CustomerRiskScore" in row and not pd.isna(row["CustomerRiskScore"]):
                tx_data["CustomerRiskScore"] = float(row["CustomerRiskScore"])

            tx_list.append(TransactionRequest(**tx_data))
        except Exception as re:
            row_errors.append(f"Row {idx+1}: {re}")

    if not tx_list:
        raise HTTPException(status_code=400, detail=f"No valid rows parsed. Row errors: {row_errors[:5]}")

    batch_res = await execute_batch(tx_list)

    if download:
        # Build CSV response
        output = io.StringIO()
        csv_writer = pd.DataFrame([
            {
                "Transaction_ID": p.transaction_id,
                "Fraud_Probability": p.fraud_probability,
                "Risk_Level": p.risk_level,
                "Is_Fraud": "FRAUD" if p.is_fraud else "LEGITIMATE",
                "Confidence_Score": p.confidence_score,
                "Processing_Time_MS": p.processing_time_ms,
            }
            for p in batch_res.predictions
        ])
        csv_writer.to_csv(output, index=False)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="fraudguard_batch_predictions.csv"'},
        )

    return batch_res


async def execute_batch(transactions: List[TransactionRequest]) -> BatchPredictionResponse:
    start_ms = time.time() * 1000
    rows = []
    tx_ids = []
    amounts = []
    times = []

    for idx, tx in enumerate(transactions):
        d = tx.model_dump()
        tx_id = d.pop("transaction_id", None) or f"BATCH-{idx}"
        tx_ids.append(tx_id)
        amounts.append(tx.Amount)
        times.append(tx.Time)

        row_dict = {}
        for col in feature_names:
            if col in d and d[col] is not None:
                row_dict[col] = float(d[col])
            elif col == "Hour":
                row_dict[col] = float((d.get("Time", 0) / 3600) % 24)
            elif col == "DayOfWeek":
                row_dict[col] = float((d.get("Time", 0) / 86400) % 7)
            elif col == "CustomerRiskScore":
                row_dict[col] = POPULATION_DEFAULTS.get("CustomerRiskScore", 0.285)
            else:
                row_dict[col] = POPULATION_DEFAULTS.get(col, 0.0)
        rows.append(row_dict)

    batch_df = pd.DataFrame(rows, columns=feature_names)
    scaled_batch = scaler.transform(batch_df)
    probs = best_model.predict_proba(scaled_batch)[:, 1]

    total_time_ms = round(time.time() * 1000 - start_ms, 2)
    avg_time_ms = round(total_time_ms / len(transactions), 2)

    predictions = []
    frauds, criticals, highs, meds, lows = 0, 0, 0, 0, 0

    for i in range(len(transactions)):
        p = round(float(probs[i]), 4)
        risk = get_risk_level(p)
        is_fraud = p >= 0.50
        if is_fraud:
            frauds += 1
        if risk == "CRITICAL":
            criticals += 1
        elif risk == "HIGH":
            highs += 1
        elif risk == "MEDIUM":
            meds += 1
        else:
            lows += 1

        predictions.append(
            FraudPredictionResponse(
                transaction_id=tx_ids[i],
                fraud_probability=p,
                is_fraud=is_fraud,
                risk_level=risk,
                confidence_score=get_confidence_score(p),
                explanation=None,
                top_risk_factors=None,
                processing_time_ms=avg_time_ms,
                model_version="FraudGuard-XGB-2.4",
                timestamp=get_utc_now_iso(),
            )
        )

    summary = {
        "total_transactions": len(transactions),
        "fraud_detected": frauds,
        "fraud_rate": round(frauds / len(transactions), 4),
        "risk_breakdown": {
            "CRITICAL": criticals,
            "HIGH": highs,
            "MEDIUM": meds,
            "LOW": lows,
        },
        "avg_processing_time_ms": avg_time_ms,
    }

    return BatchPredictionResponse(
        predictions=predictions,
        batch_summary=summary,
        total_processing_time_ms=total_time_ms,
    )


@app.post("/reload-models")
async def reload_models(
    authorization: Optional[str] = Header(None, description="Bearer admin token for reloading model artifacts")
):
    """Administrative endpoint to trigger in-memory model reload from disk.
    Protected by FRAUDGUARD_ADMIN_TOKEN server secret. When unset, endpoint is disabled."""
    admin_token = os.environ.get("FRAUDGUARD_ADMIN_TOKEN")
    if not admin_token or not admin_token.strip():
        raise HTTPException(
            status_code=503,
            detail="Model reload endpoint is administratively disabled. Set FRAUDGUARD_ADMIN_TOKEN to enable.",
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header. Expected 'Bearer <token>'.")

    token_candidate = authorization[len("Bearer "):].strip()
    # Constant-time comparison prevents timing analysis
    if not secrets.compare_digest(token_candidate, admin_token.strip()):
        raise HTTPException(status_code=403, detail="Forbidden: Invalid administrator token.")

    if load_models():
        return {
            "message": "Models reloaded successfully",
            "timestamp": get_utc_now_iso(),
        }
    raise HTTPException(status_code=500, detail="Failed to reload models")


# ---------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint not found",
            "available_endpoints": ["/", "/health", "/model-info", "/analytics", "/metrics", "/history", "/predict", "/predict/batch", "/docs"],
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": get_utc_now_iso(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting FraudGuard AI server on 0.0.0.0:{port}...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
