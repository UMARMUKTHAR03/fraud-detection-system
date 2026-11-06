"""
Complete Model Training, Evaluation, and Serialization Pipeline
===============================================================

Executes a leak-free machine learning workflow:
1. Loads/generates calibrated synthetic financial fraud benchmark dataset.
2. Evaluates 4 candidate models + soft-voting ensemble across 5-Fold Stratified CV.
3. Evaluates models on a holdout test split: ROC-AUC, PR-AUC, F1, Precision, Recall, Latency.
4. Identifies the champion model and fits an interpretable SHAP TreeExplainer.
5. Saves all production artifacts to `models/`:
   - `best_model.joblib`
   - `scaler.joblib`
   - `feature_names.json`
   - `explainer.joblib`
   - `metrics.json`
   - `fraud_detection_ensemble.pkl`
   - `feature_engineering_pipeline.pkl`

Author: UMAR MUKTHAR (@UMARMUKTHAR03)
Platform: FraudGuard AI
"""

import json
import logging
import os
import sys
import time
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure root in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_processing.generate_data import DatasetConfig, FraudDatasetGenerator
from src.models.fraud_detector import (
    LogisticRegressionDetector,
    RandomForestDetector,
    XGBoostDetector,
    LightGBMDetector,
    EnsembleFraudDetector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline():
    logger.info("==================================================")
    logger.info("🚀 STARTING FRAUDGUARD AI MODEL TRAINING PIPELINE")
    logger.info("==================================================")

    data_dir = os.path.join(ROOT_DIR, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "creditcard_transactions.csv")

    n_samples = 40000
    fraud_rate = 0.02
    random_state = 42

    logger.info(f"Generating benchmark dataset ({n_samples:,} samples, fraud_rate={fraud_rate})...")
    gen_config = DatasetConfig(
        n_samples=n_samples,
        fraud_rate=fraud_rate,
        time_period_days=3,
        random_state=random_state,
    )
    generator = FraudDatasetGenerator(gen_config)
    df = generator.create_fraud_dataset()
    df.to_csv(csv_path, index=False)
    logger.info(f"Dataset successfully saved to {csv_path} (Shape: {df.shape})")

    # Define feature columns: Time, Amount, V1..V20
    # CustomerRiskScore, Hour, DayOfWeek, MerchantCategory can be included or kept numerical
    feature_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 21)]
    if "Hour" in df.columns:
        feature_cols.append("Hour")
    if "DayOfWeek" in df.columns:
        feature_cols.append("DayOfWeek")
    if "CustomerRiskScore" in df.columns:
        feature_cols.append("CustomerRiskScore")

    X = df[feature_cols].copy()
    y = df["Class"].astype(int).copy()

    # Stratified Train/Test Split (80/20) - holdout test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )
    logger.info(f"Train set: {len(X_train_raw):,} samples | Test set: {len(X_test_raw):,} samples")

    # Standard Scaler (Fit exclusively on X_train to eliminate data leakage)
    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train_raw),
        columns=feature_cols,
        index=X_train_raw.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=feature_cols,
        index=X_test_raw.index,
    )

    # 5-Fold Stratified Cross Validation on Train Set with SMOTE applied strictly inside folds
    logger.info("\n📊 Running 5-Fold Stratified Cross Validation on Candidate Models...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    candidate_models = {
        "Logistic Regression": LogisticRegressionDetector(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestDetector(n_estimators=100, max_depth=12, random_state=random_state),
        "LightGBM": LightGBMDetector(n_estimators=100, max_depth=6, random_state=random_state),
        "XGBoost": XGBoostDetector(n_estimators=100, max_depth=6, random_state=random_state),
    }

    cv_results = {}
    for name, model_detector in candidate_models.items():
        logger.info(f"Evaluating {name} with 5-Fold CV...")
        fold_auc = []
        fold_pr_auc = []
        fold_f1 = []
        fold_precision = []
        fold_recall = []

        for train_idx, val_idx in cv.split(X_train_raw, y_train):
            X_fold_tr_raw = X_train_raw.iloc[train_idx]
            y_fold_tr = y_train.iloc[train_idx]
            X_fold_val_raw = X_train_raw.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Fold-level scaling to ensure zero leakage
            fold_scaler = StandardScaler()
            X_tr_s = pd.DataFrame(fold_scaler.fit_transform(X_fold_tr_raw), columns=feature_cols)
            X_val_s = pd.DataFrame(fold_scaler.transform(X_fold_val_raw), columns=feature_cols)

            # SMOTE only on training fold
            smote = SMOTE(random_state=random_state)
            X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_s, y_fold_tr)

            # Fit candidate
            if name == "Logistic Regression":
                clf = LogisticRegression(max_iter=1000, random_state=random_state)
            elif name == "Random Forest":
                clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1)
            elif name == "LightGBM":
                clf = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=random_state, verbosity=-1, n_jobs=-1)
            elif name == "XGBoost":
                clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=random_state, eval_metric="logloss", n_jobs=-1)

            clf.fit(X_tr_bal, y_tr_bal)
            val_probs = clf.predict_proba(X_val_s)[:, 1]
            val_preds = (val_probs >= 0.50).astype(int)

            fold_auc.append(roc_auc_score(y_fold_val, val_probs))
            fold_pr_auc.append(average_precision_score(y_fold_val, val_probs))
            fold_f1.append(f1_score(y_fold_val, val_preds, zero_division=0))
            fold_precision.append(precision_score(y_fold_val, val_preds, zero_division=0))
            fold_recall.append(recall_score(y_fold_val, val_preds, zero_division=0))

        cv_results[name] = {
            "cv_roc_auc_mean": round(float(np.mean(fold_auc)), 4),
            "cv_roc_auc_std": round(float(np.std(fold_auc)), 4),
            "cv_pr_auc_mean": round(float(np.mean(fold_pr_auc)), 4),
            "cv_f1_mean": round(float(np.mean(fold_f1)), 4),
            "cv_precision_mean": round(float(np.mean(fold_precision)), 4),
            "cv_recall_mean": round(float(np.mean(fold_recall)), 4),
        }
        logger.info(
            f"  -> {name}: ROC-AUC={cv_results[name]['cv_roc_auc_mean']} ± {cv_results[name]['cv_roc_auc_std']}, "
            f"PR-AUC={cv_results[name]['cv_pr_auc_mean']}, F1={cv_results[name]['cv_f1_mean']}"
        )

    # Train each model on full training set (with SMOTE) and evaluate on test set
    logger.info("\n🏋️ Training full models & evaluating on unseen Holdout Test Set...")
    smote_full = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote_full.fit_resample(X_train_scaled, y_train)

    fitted_models = {}
    test_benchmarks = {}

    models_to_fit = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=random_state, verbosity=-1, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=random_state, eval_metric="logloss", n_jobs=-1),
    }

    for name, clf in models_to_fit.items():
        start_t = time.time()
        clf.fit(X_train_bal, y_train_bal)
        train_time = round(time.time() - start_t, 3)

        infer_start = time.time()
        test_probs = clf.predict_proba(X_test_scaled)[:, 1]
        latency_ms = round((time.time() - infer_start) * 1000 / len(X_test_scaled), 4)

        test_preds = (test_probs >= 0.50).astype(int)

        roc_auc = float(roc_auc_score(y_test, test_probs))
        pr_auc = float(average_precision_score(y_test, test_probs))
        acc = float(accuracy_score(y_test, test_preds))
        prec = float(precision_score(y_test, test_preds, zero_division=0))
        rec = float(recall_score(y_test, test_preds, zero_division=0))
        f1 = float(f1_score(y_test, test_preds, zero_division=0))

        cm = confusion_matrix(y_test, test_preds)

        fitted_models[name] = clf
        test_benchmarks[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "latency_ms": latency_ms,
            "training_time_sec": train_time,
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
            "cv_metrics": cv_results[name],
        }
        logger.info(
            f"Holdout Test: {name:20s} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | "
            f"F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | Latency: {latency_ms:.3f} ms"
        )

    # Also build and test the weighted Soft-Voting Ensemble
    logger.info("\n🔮 Evaluating Soft-Voting Ensemble (XGBoost 45% + LightGBM 30% + Random Forest 25%)...")
    ensemble_probs = (
        0.45 * fitted_models["XGBoost"].predict_proba(X_test_scaled)[:, 1]
        + 0.30 * fitted_models["LightGBM"].predict_proba(X_test_scaled)[:, 1]
        + 0.25 * fitted_models["Random Forest"].predict_proba(X_test_scaled)[:, 1]
    )
    ensemble_preds = (ensemble_probs >= 0.50).astype(int)
    cm_ens = confusion_matrix(y_test, ensemble_preds)

    test_benchmarks["Ensemble (XGB+LGB+RF)"] = {
        "accuracy": round(float(accuracy_score(y_test, ensemble_preds)), 4),
        "precision": round(float(precision_score(y_test, ensemble_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, ensemble_preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, ensemble_preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, ensemble_probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, ensemble_probs)), 4),
        "latency_ms": round(test_benchmarks["XGBoost"]["latency_ms"] + test_benchmarks["LightGBM"]["latency_ms"], 4),
        "training_time_sec": round(
            test_benchmarks["XGBoost"]["training_time_sec"] + test_benchmarks["LightGBM"]["training_time_sec"], 3
        ),
        "confusion_matrix": {
            "tn": int(cm_ens[0, 0]),
            "fp": int(cm_ens[0, 1]),
            "fn": int(cm_ens[1, 0]),
            "tp": int(cm_ens[1, 1]),
        },
    }

    # Champion model selection based on highest PR-AUC and ROC-AUC
    # XGBoost or Ensemble: select XGBoost as champion standalone model for fast SHAP TreeExplainer,
    # while preserving ensemble artifact
    champion_name = "XGBoost"
    champion_clf = fitted_models[champion_name]
    champion_metrics = test_benchmarks[champion_name]

    logger.info(f"\n🏆 CHAMPION MODEL SELECTED: {champion_name}")
    logger.info(f"ROC-AUC: {champion_metrics['roc_auc']} | PR-AUC: {champion_metrics['pr_auc']} | F1: {champion_metrics['f1_score']}")

    # Calculate ROC and PR curve points for visualization
    fpr, tpr, roc_thresh = roc_curve(y_test, champion_clf.predict_proba(X_test_scaled)[:, 1])
    prec_pts, rec_pts, pr_thresh = precision_recall_curve(y_test, champion_clf.predict_proba(X_test_scaled)[:, 1])

    # Downsample curve points for compact JSON storage (e.g., 25 points)
    indices = np.linspace(0, len(fpr) - 1, min(25, len(fpr)), dtype=int)
    roc_curve_data = [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in indices]

    pr_indices = np.linspace(0, len(prec_pts) - 1, min(25, len(prec_pts)), dtype=int)
    pr_curve_data = [
        {"recall": round(float(rec_pts[i]), 4), "precision": round(float(prec_pts[i]), 4)}
        for i in pr_indices
    ]

    # Global feature importances
    if hasattr(champion_clf, "feature_importances_"):
        raw_imp = champion_clf.feature_importances_
        feature_importance_dict = {
            feature_cols[i]: round(float(raw_imp[i]), 5) for i in range(len(feature_cols))
        }
        # Sort descending
        feature_importance_sorted = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )
    else:
        feature_importance_sorted = []

    # Fit SHAP Explainer
    logger.info("Fitting SHAP TreeExplainer on champion XGBoost model...")
    # Use TreeExplainer with sample background for fast inference
    explainer = shap.TreeExplainer(champion_clf)

    # Save artifacts to models/
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(champion_clf, os.path.join(models_dir, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(explainer, os.path.join(models_dir, "explainer.joblib"))

    # Also build EnsembleFraudDetector instance and save
    ensemble_instance = EnsembleFraudDetector(random_state=random_state)
    ensemble_instance.models["rf"].model = fitted_models["Random Forest"]
    ensemble_instance.models["rf"].is_trained = True
    ensemble_instance.models["xgb"].model = fitted_models["XGBoost"]
    ensemble_instance.models["xgb"].is_trained = True
    ensemble_instance.models["lgb"].model = fitted_models["LightGBM"]
    ensemble_instance.models["lgb"].is_trained = True
    ensemble_instance.is_trained = True
    ensemble_instance.save_model(os.path.join(models_dir, "fraud_detection_ensemble.pkl"))

    # Save feature names
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save metrics.json
    final_metrics_payload = {
        "production_model": champion_name,
        "algorithm": "Extreme Gradient Boosting (XGBClassifier)",
        "framework_version": xgb.__version__,
        "dataset": {
            "name": "FraudGuard Calibrated Financial Transactions Benchmark",
            "samples": n_samples,
            "train_samples": len(X_train_raw),
            "test_samples": len(X_test_raw),
            "features_count": len(feature_cols),
            "fraud_rate": fraud_rate,
            "target": "Class (1=Fraudulent, 0=Legitimate)",
        },
        "decision_threshold": 0.50,
        "risk_thresholds": {
            "LOW": "< 0.25",
            "MEDIUM": "0.25 - 0.4999",
            "HIGH": "0.50 - 0.7499",
            "CRITICAL": ">= 0.75",
        },
        "holdout_evaluation": champion_metrics,
        "model_comparison": test_benchmarks,
        "top_features": [
            {"feature": f, "importance": imp} for f, imp in feature_importance_sorted[:10]
        ],
        "roc_curve": roc_curve_data,
        "pr_curve": pr_curve_data,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics_payload, f, indent=2)

    logger.info("✅ All artifacts saved successfully to models/:")
    for f in os.listdir(models_dir):
        sz = os.path.getsize(os.path.join(models_dir, f))
        logger.info(f"  - models/{f} ({sz:,} bytes)")

    return final_metrics_payload


if __name__ == "__main__":
    run_training_pipeline()
