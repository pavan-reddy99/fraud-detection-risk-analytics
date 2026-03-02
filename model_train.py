"""
model_train.py
--------------
Train, evaluate, and select the best fraud detection classifier.
Compares Random Forest, XGBoost, and Logistic Regression.
Saves the best model based on ROC-AUC score.
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier


FEATURE_COLS = [
    "amount", "hour_of_day", "day_of_week",
    "txn_count_7d", "txn_count_30d", "avg_txn_amount",
    "stddev_txn_amount", "amount_zscore",
    "distinct_merchant_cats", "hours_since_last_txn"
]
LABEL_COL = "is_fraud"
THRESHOLD  = 0.42   # Tuned for ~12% FPR tolerance


def load_features(features_path: str) -> pd.DataFrame:
    import glob
    files = glob.glob(os.path.join(features_path, "*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=300, scale_pos_weight=10, use_label_encoder=False,
                                             eval_metric="logloss", random_state=42)
    }


def evaluate_model(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    auc   = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Precision: {report['1']['precision']:.4f}")
    print(f"  Recall   : {report['1']['recall']:.4f}")
    print(f"  F1-Score : {report['1']['f1-score']:.4f}")
    print(confusion_matrix(y_test, y_pred))

    return auc, y_prob


def plot_roc_curves(results, y_test, output_dir):
    plt.figure(figsize=(8, 6))
    for name, (_, y_prob) in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Fraud Detection Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    print(f"📊 ROC curves saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to feature parquet files")
    parser.add_argument("--output",   required=True, help="Directory to save model & plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("📥 Loading features...")
    df = load_features(args.features)
    X  = df[FEATURE_COLS]
    y  = df[LABEL_COL]
    print(f"   → {len(df):,} samples | fraud rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models  = get_models()
    results = {}
    best_auc, best_name, best_model = 0, None, None

    for name, model in models.items():
        print(f"\n🏋️  Training {name}...")
        model.fit(X_train, y_train)
        auc, y_prob = evaluate_model(name, model, X_test, y_test)
        results[name] = (auc, y_prob)

        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, model

    print(f"\n🏆 Best model: {best_name} (AUC={best_auc:.4f})")
    model_path = os.path.join(args.output, "xgb_fraud_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"✅ Model saved to {model_path}")

    plot_roc_curves(results, y_test, args.output)


if __name__ == "__main__":
    main()
