"""
scoring_pipeline.py
-------------------
Daily batch scoring pipeline.
Loads the trained model, scores all accounts, and outputs
a ranked risk CSV ready for Power BI ingestion.
"""

import argparse
import pickle
import pandas as pd


FEATURE_COLS = [
    "amount", "hour_of_day", "day_of_week",
    "txn_count_7d", "txn_count_30d", "avg_txn_amount",
    "stddev_txn_amount", "amount_zscore",
    "distinct_merchant_cats", "hours_since_last_txn"
]
THRESHOLD = 0.42


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def score_accounts(model, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS]
    df["fraud_probability"] = model.predict_proba(X)[:, 1]
    df["risk_tier"] = pd.cut(
        df["fraud_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )
    df["review_flag"] = (df["fraud_probability"] >= THRESHOLD).astype(int)
    return df.sort_values("fraud_probability", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="Path to trained model .pkl")
    parser.add_argument("--input",  required=True, help="Path to today's feature data (parquet or CSV)")
    parser.add_argument("--output", required=True, help="Output path for risk scores CSV")
    args = parser.parse_args()

    print("📥 Loading model...")
    model = load_model(args.model)

    print("📥 Loading today's account features...")
    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    print(f"   → Scoring {len(df):,} accounts...")
    scored = score_accounts(model, df)

    scored.to_csv(args.output, index=False)
    flagged = scored["review_flag"].sum()
    print(f"✅ Scores saved to {args.output}")
    print(f"   → {flagged:,} accounts flagged for review ({flagged/len(df):.1%} of total)")
    print(f"   → Risk distribution:\n{scored['risk_tier'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
