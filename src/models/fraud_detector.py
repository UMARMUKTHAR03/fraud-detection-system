# src/models/fraud_detector.py
"""
Fraud Detection Model Architectures & Ensemble Engine
=====================================================

Production-grade ML model wrappers for financial fraud detection:
- LogisticRegressionDetector (Linear baseline)
- RandomForestDetector (Bagging tree ensemble)
- XGBoostDetector (Extreme Gradient Boosting)
- LightGBMDetector (Light Gradient Boosting Machine)
- EnsembleFraudDetector (Weighted soft-voting ensemble)

Author: UMAR MUKTHAR (@UMARMUKTHAR03)
Platform: FraudGuard AI
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb


class BaseModelDetector:
    """Base class for fraud detection models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: Any = None
        self.is_trained: bool = False
        self.scaler: Optional[StandardScaler] = None

    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        balance_data: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split and optionally balance training data."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        if balance_data:
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        return X_train, X_test, y_train, y_test

    def evaluate_model(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        y_pred = self.model.predict(X_test)

        # Calculate proba if available
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_prob))
        else:
            auc = 0.5

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": auc,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        preds = self.model.predict(X)
        return np.column_stack([1.0 - preds, preds])

    def save_model(self, filepath: str) -> None:
        """Persist model to disk."""
        data = {
            "model": self.model,
            "is_trained": self.is_trained,
            "random_state": self.random_state,
        }
        joblib.dump(data, filepath)

    def load_model(self, filepath: str) -> None:
        """Load model from disk."""
        data = joblib.load(filepath)
        if isinstance(data, dict) and "model" in data:
            self.model = data["model"]
            self.is_trained = data["is_trained"]
        else:
            self.model = data
            self.is_trained = True


class LogisticRegressionDetector(BaseModelDetector):
    """Logistic Regression Baseline with internal scaling."""

    def __init__(
        self,
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
        balance_data: bool = False,
        random_state: int = 42,
    ):
        super().__init__(random_state=random_state)
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.balance_data = balance_data
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=max_iter,
                        class_weight=class_weight,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.balance_data:
            smote = SMOTE(random_state=self.random_state)
            X, y = smote.fit_resample(X, y)
        self.model.fit(X, y)
        self.is_trained = True

    def save(self, filepath: str) -> None:
        joblib.dump(self.model, filepath)

    def load(self, filepath: str) -> None:
        self.model = joblib.load(filepath)
        self.is_trained = True


class RandomForestDetector(BaseModelDetector):
    """Random Forest Classifier for fraud pattern detection."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ):
        super().__init__(random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        self.is_trained = True


class XGBoostDetector(BaseModelDetector):
    """XGBoost Gradient Boosting Classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 5.0,
        random_state: int = 42,
    ):
        super().__init__(random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        self.is_trained = True


class LightGBMDetector(BaseModelDetector):
    """LightGBM Classifier optimized for tabular high-velocity fraud detection."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 5.0,
        random_state: int = 42,
    ):
        super().__init__(random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        self.is_trained = True


class EnsembleFraudDetector:
    """
    Weighted Soft-Voting Ensemble Classifier combining XGBoost,
    LightGBM, and Random Forest.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.weights = weights or {"rf": 0.25, "xgb": 0.45, "lgb": 0.30}
        self.models: Dict[str, Any] = {
            "rf": RandomForestDetector(random_state=random_state),
            "xgb": XGBoostDetector(random_state=random_state),
            "lgb": LightGBMDetector(random_state=random_state),
        }
        self.is_trained: bool = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        balance_data: bool = False,
    ) -> None:
        """Train all underlying ensemble models."""
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        if balance_data:
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        for name, model_detector in self.models.items():
            model_detector.train(X_train, y_train)

        self.is_trained = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute weighted average predicted probabilities."""
        if not self.is_trained:
            raise ValueError("Ensemble is not trained.")

        total_weight = sum(self.weights.values())
        ensemble_proba = np.zeros((len(X), 2))

        for name, model_detector in self.models.items():
            weight = self.weights.get(name, 1.0)
            proba = model_detector.predict_proba(X)
            ensemble_proba += (weight / total_weight) * proba

        return ensemble_proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict class using custom decision threshold."""
        prob_fraud = self.predict_proba(X)[:, 1]
        return (prob_fraud >= threshold).astype(int)

    def evaluate_model(
        self, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5
    ) -> Dict[str, float]:
        """Evaluate ensemble metrics."""
        y_prob = self.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, y_prob)),
        }

    def save_model(self, filepath: str) -> None:
        """Persist entire ensemble."""
        data = {
            "weights": self.weights,
            "models": {k: m.model for k, m in self.models.items()},
            "is_trained": self.is_trained,
        }
        joblib.dump(data, filepath)

    def load_model(self, filepath: str) -> None:
        """Load entire ensemble."""
        data = joblib.load(filepath)
        self.weights = data.get("weights", self.weights)
        for k, raw_model in data["models"].items():
            if k in self.models:
                self.models[k].model = raw_model
                self.models[k].is_trained = True
        self.is_trained = True


def finalize_training() -> None:
    print("Training complete.")
