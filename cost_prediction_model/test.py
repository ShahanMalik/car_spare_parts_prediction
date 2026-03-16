"""
Car Parts Price Prediction Model – Testing Script
====================================================
Loads the trained model and runs predictions on:
  1. Held-out test set (from the CSV)
  2. Interactive single-part predictions
  3. Sample predictions for well-known parts

Usage:
    python test.py                           # full test suite
    python test.py --interactive             # interactive mode only
    python test.py --data ../cost_prediction/car_parts_final.csv
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_DIR = os.path.dirname(__file__) or "."
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
DEFAULT_DATA = os.path.join(
    os.path.dirname(__file__), "..", "cost_prediction", "car_parts_final.csv"
)

TARGET = "price_pkr"
DROP_COLS = ["available", "website", "product_url", "description"]
CAT_FEATURES = ["part_brand", "scraped_category", "compatible_make", "compatible_model", "condition"]
HIGH_CARD_CAT = "part_type"
NUM_FEATURES = ["year_from", "year_to", "price_original_pkr", "alternatives_count"]
TEXT_FEATURE = "part_name"


# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
def load_model():
    """Load trained model bundle."""
    if not os.path.exists(MODEL_PATH):
        print(f"  Model not found: {MODEL_PATH}")
        print("  Run train.py first.")
        return None
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def load_meta():
    """Load model metadata."""
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Feature Builder (mirrors train.py)
# ─────────────────────────────────────────────
def prepare_dataframe(df: pd.DataFrame, encoders: dict = None, remove_outliers: bool = False) -> pd.DataFrame:
    """Clean and add engineered features (same as train.py).
    
    If encoders is provided, uses the saved freq/mean lookup tables
    from training (correct for small batches / single predictions).
    """
    existing_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drop, errors="ignore").copy()

    df = df[df[TARGET] > 0].copy()

    # Optionally remove outliers (same as training)
    if remove_outliers:
        q1 = df[TARGET].quantile(0.01)
        q99 = df[TARGET].quantile(0.99)
        df = df[(df[TARGET] >= q1) & (df[TARGET] <= q99)].copy()

    df["year_from"] = df["year_from"].fillna(0).astype(int)
    df["year_to"] = df["year_to"].fillna(0).astype(int)
    df["price_original_pkr"] = df["price_original_pkr"].fillna(0)
    df["alternatives_count"] = df["alternatives_count"].fillna(0).astype(int)

    for col in CAT_FEATURES + [HIGH_CARD_CAT]:
        df[col] = df[col].fillna("Unknown").astype(str)

    df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna("").astype(str)

    # Engineered features
    df["year_span"] = (df["year_to"] - df["year_from"]).clip(lower=0)
    df["has_year"] = ((df["year_from"] > 0) | (df["year_to"] > 0)).astype(int)
    df["is_universal"] = (df["compatible_make"].str.lower() == "universal").astype(int)
    df["has_discount"] = (df["price_original_pkr"] > 0).astype(int)
    df["name_word_count"] = df[TEXT_FEATURE].str.split().str.len().fillna(0).astype(int)

    # Frequency & mean-price encoding using saved lookup tables
    if encoders and "freq_maps" in encoders:
        freq_maps = encoders["freq_maps"]
        for col in ["compatible_make", "compatible_model", "part_brand", "scraped_category"]:
            fmap = freq_maps.get(col, {})
            df[f"{col}_freq"] = df[col].map(fmap).fillna(0)
    else:
        for col in ["compatible_make", "compatible_model", "part_brand", "scraped_category"]:
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq).fillna(0)

    if encoders and "price_mean_maps" in encoders:
        price_mean_maps = encoders["price_mean_maps"]
        global_mean = encoders.get("global_mean_price", 10000)
        for col in ["scraped_category", "compatible_make"]:
            pmap = price_mean_maps.get(col, {})
            df[f"{col}_price_mean"] = np.log1p(df[col].map(pmap).fillna(global_mean))
    else:
        for col in ["scraped_category", "compatible_make"]:
            grp = df.groupby(col)[TARGET].transform("mean")
            df[f"{col}_price_mean"] = np.log1p(grp)

    return df


def build_X(df: pd.DataFrame, encoders: dict) -> np.ndarray:
    """Build feature matrix using stored encoders."""
    # Categorical
    cat_encoded = []
    for col in CAT_FEATURES:
        enc = encoders[col]
        vals = enc.transform(df[[col]])
        cat_encoded.append(vals)

    pt_enc = encoders[HIGH_CARD_CAT]
    pt_vals = pt_enc.transform(df[[HIGH_CARD_CAT]])
    cat_encoded.append(pt_vals)

    cat_matrix = np.hstack(cat_encoded)

    # Numeric
    eng_features = [
        "year_span", "has_year", "is_universal", "has_discount", "name_word_count",
        "compatible_make_freq", "compatible_model_freq", "part_brand_freq", "scraped_category_freq",
        "scraped_category_price_mean", "compatible_make_price_mean",
    ]
    num_cols = NUM_FEATURES + eng_features
    num_matrix = df[num_cols].values.astype(float)

    # Text (TF-IDF)
    if "tfidf" in encoders:
        tfidf = encoders["tfidf"]
        text_matrix = tfidf.transform(df[TEXT_FEATURE]).toarray()
    else:
        text_matrix = np.empty((len(df), 0))

    return np.hstack([cat_matrix, num_matrix, text_matrix])


# ─────────────────────────────────────────────
# Test on Hold-out Set
# ─────────────────────────────────────────────
def test_holdout(data_path: str, bundle: dict):
    """Evaluate model on the same test split as training."""
    print("\n[1] Hold-out Test Set Evaluation")
    print("─" * 50)

    model = bundle["model"]
    encoders = bundle["encoders"]

    df = pd.read_csv(data_path)
    df = prepare_dataframe(df, encoders=encoders, remove_outliers=True)

    y_all = np.log1p(df[TARGET].values)
    X_all = build_X(df, encoders)

    # Same split as training (random_state=42, test_size=0.2)
    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    # Predict
    y_pred_log = model.predict(X_test)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    y_actual = np.expm1(y_test)

    # Metrics
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    mape = mean_absolute_percentage_error(y_actual, y_pred) * 100
    median_ae = np.median(np.abs(y_actual - y_pred))

    print(f"  Test samples:  {len(y_test)}")
    print(f"  R² score:      {r2:.4f}")
    print(f"  MAE:           Rs {mae:,.0f}")
    print(f"  Median AE:     Rs {median_ae:,.0f}")
    print(f"  RMSE:          Rs {rmse:,.0f}")
    print(f"  MAPE:          {mape:.1f}%")

    # ── Price bucket accuracy ──
    print(f"\n  Accuracy by price range:")
    print(f"  {'Price Range':<25s} {'Count':>6s} {'MAE':>10s} {'MAPE':>8s} {'R²':>8s}")
    print(f"  {'─' * 60}")

    buckets = [
        ("Rs 0 – 2,000", 0, 2000),
        ("Rs 2,000 – 5,000", 2000, 5000),
        ("Rs 5,000 – 15,000", 5000, 15000),
        ("Rs 15,000 – 50,000", 15000, 50000),
        ("Rs 50,000+", 50000, float("inf")),
    ]
    for label, low, high in buckets:
        mask = (y_actual >= low) & (y_actual < high)
        if mask.sum() == 0:
            continue
        b_mae = mean_absolute_error(y_actual[mask], y_pred[mask])
        b_mape = mean_absolute_percentage_error(y_actual[mask], y_pred[mask]) * 100
        b_r2 = r2_score(y_actual[mask], y_pred[mask]) if mask.sum() > 1 else 0.0
        print(f"  {label:<25s} {mask.sum():>6d} Rs {b_mae:>8,.0f} {b_mape:>7.1f}% {b_r2:>7.4f}")

    # ── Sample predictions ──
    print(f"\n  Sample Predictions (first 15):")
    print(f"  {'Actual':>12s} {'Predicted':>12s} {'Error':>10s} {'%Err':>7s}  Part Name")
    print(f"  {'─' * 80}")

    # Get test indices
    _, test_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42
    )
    test_df = df.iloc[test_idx].reset_index(drop=True)

    for i in range(min(15, len(y_actual))):
        actual = y_actual[i]
        pred = y_pred[i]
        err = pred - actual
        pct = abs(err) / actual * 100 if actual > 0 else 0
        name = test_df.iloc[i][TEXT_FEATURE][:45] if i < len(test_df) else ""
        print(f"  Rs {actual:>10,.0f} Rs {pred:>10,.0f} {err:>+10,.0f} {pct:>6.1f}%  {name}")

    return {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape, "median_ae": median_ae}


# ─────────────────────────────────────────────
# Sample Known Parts Test
# ─────────────────────────────────────────────
def test_known_parts(bundle: dict):
    """Test with manually defined car parts."""
    print("\n\n[2] Known Parts Predictions")
    print("─" * 50)

    sample_parts = [
        {
            "part_name": "Denso Iridium Spark Plug Toyota Corolla",
            "part_type": "Iridium Spark Plug",
            "part_brand": "Denso",
            "scraped_category": "Spark Plugs",
            "compatible_make": "Toyota",
            "compatible_model": "Corolla",
            "year_from": 2014,
            "year_to": 2020,
            "price_original_pkr": 0,
            "alternatives_count": 3,
            "condition": "New",
        },
        {
            "part_name": "Guard Oil Filter Honda City 2009-2021",
            "part_type": "Oil Filter",
            "part_brand": "Guard",
            "scraped_category": "Oil Filter",
            "compatible_make": "Honda",
            "compatible_model": "City",
            "year_from": 2009,
            "year_to": 2021,
            "price_original_pkr": 0,
            "alternatives_count": 5,
            "condition": "New",
        },
        {
            "part_name": "Toyota Corolla LED Head Lights 2017-2021",
            "part_type": "LED Head Light",
            "part_brand": "Unknown",
            "scraped_category": "LED Head Lights",
            "compatible_make": "Toyota",
            "compatible_model": "Corolla",
            "year_from": 2017,
            "year_to": 2021,
            "price_original_pkr": 0,
            "alternatives_count": 2,
            "condition": "New",
        },
        {
            "part_name": "Yokohama Tyre 195/65R15 Earth-1",
            "part_type": "Tyre",
            "part_brand": "Yokohama",
            "scraped_category": "Tyres",
            "compatible_make": "Universal",
            "compatible_model": "Universal",
            "year_from": 0,
            "year_to": 0,
            "price_original_pkr": 0,
            "alternatives_count": 4,
            "condition": "New",
        },
        {
            "part_name": "Honda Civic Body Kit 2016-2021",
            "part_type": "Body Kit",
            "part_brand": "Unknown",
            "scraped_category": "Body Kits",
            "compatible_make": "Honda",
            "compatible_model": "Civic",
            "year_from": 2016,
            "year_to": 2021,
            "price_original_pkr": 0,
            "alternatives_count": 1,
            "condition": "New",
        },
    ]

    model = bundle["model"]
    encoders = bundle["encoders"]

    df = pd.DataFrame(sample_parts)
    # Add dummy target for the pipeline
    df[TARGET] = 1  # placeholder
    df = prepare_dataframe(df, encoders=encoders)
    X = build_X(df, encoders)

    y_pred_log = model.predict(X)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)

    print(f"\n  {'Part Name':<50s} {'Predicted Price':>15s}")
    print(f"  {'─' * 68}")
    for i, part in enumerate(sample_parts):
        name = part["part_name"][:48]
        price = y_pred[i]
        print(f"  {name:<50s} Rs {price:>12,.0f}")


# ─────────────────────────────────────────────
# Interactive Mode
# ─────────────────────────────────────────────
def interactive_mode(bundle: dict):
    """Let user input part details and get price predictions."""
    print("\n\n[3] Interactive Price Prediction")
    print("─" * 50)
    print("  Enter part details (or 'quit' to exit)\n")

    model = bundle["model"]
    encoders = bundle["encoders"]

    while True:
        try:
            name = input("  Part name: ").strip()
            if name.lower() in ("quit", "exit", "q", ""):
                break

            part_type = input("  Part type (e.g. Oil Filter, Spark Plug): ").strip() or "Unknown"
            brand = input("  Brand (e.g. Denso, Guard, Unknown): ").strip() or "Unknown"
            category = input("  Category (e.g. Spark Plugs, Oil Filter): ").strip() or "Unknown"
            make = input("  Car make (e.g. Toyota, Honda, Universal): ").strip() or "Universal"
            model_name = input("  Car model (e.g. Corolla, Civic, Universal): ").strip() or "Universal"
            year_from = input("  Year from (e.g. 2017, or 0): ").strip()
            year_to = input("  Year to (e.g. 2021, or 0): ").strip()

            year_from = int(year_from) if year_from else 0
            year_to = int(year_to) if year_to else 0

            part = {
                "part_name": name,
                "part_type": part_type,
                "part_brand": brand,
                "scraped_category": category,
                "compatible_make": make,
                "compatible_model": model_name,
                "year_from": year_from,
                "year_to": year_to,
                "price_original_pkr": 0,
                "alternatives_count": 0,
                "condition": "New",
                TARGET: 1,  # placeholder
            }

            df = pd.DataFrame([part])
            df = prepare_dataframe(df, encoders=encoders)
            X = build_X(df, encoders)

            y_pred_log = model.predict(X)
            predicted = max(0, np.expm1(y_pred_log[0]))

            print(f"\n  >>> Predicted Price: Rs {predicted:,.0f}\n")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as exc:
            print(f"  Error: {exc}\n")

    print("  Done.\n")


# ─────────────────────────────────────────────
# Category/Make/Model Price Analysis
# ─────────────────────────────────────────────
def analyze_predictions(data_path: str, bundle: dict):
    """Analyze prediction quality across different categories and makes."""
    print("\n\n[4] Prediction Analysis by Category & Make")
    print("─" * 50)

    model = bundle["model"]
    encoders = bundle["encoders"]

    df = pd.read_csv(data_path)
    df = prepare_dataframe(df, encoders=encoders, remove_outliers=True)

    y_all = np.log1p(df[TARGET].values)
    X_all = build_X(df, encoders)

    y_pred_log = model.predict(X_all)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    y_actual = np.expm1(y_all)

    df["predicted"] = y_pred
    df["actual"] = y_actual
    df["error"] = y_pred - y_actual
    df["abs_pct_error"] = (np.abs(df["error"]) / df["actual"] * 100).clip(upper=500)

    # By category
    print(f"\n  By Category:")
    print(f"  {'Category':<30s} {'Count':>6s} {'Avg Error%':>10s} {'Avg MAE':>12s}")
    print(f"  {'─' * 62}")
    cat_stats = df.groupby("scraped_category").agg(
        count=("actual", "size"),
        mape=("abs_pct_error", "mean"),
        mae=("error", lambda x: np.mean(np.abs(x))),
    ).sort_values("count", ascending=False).head(15)

    for cat, row in cat_stats.iterrows():
        print(f"  {cat:<30s} {int(row['count']):>6d} {row['mape']:>9.1f}% Rs {row['mae']:>9,.0f}")

    # By make
    print(f"\n  By Car Make:")
    print(f"  {'Make':<20s} {'Count':>6s} {'Avg Error%':>10s} {'Avg MAE':>12s}")
    print(f"  {'─' * 52}")
    make_stats = df.groupby("compatible_make").agg(
        count=("actual", "size"),
        mape=("abs_pct_error", "mean"),
        mae=("error", lambda x: np.mean(np.abs(x))),
    ).sort_values("count", ascending=False).head(10)

    for make, row in make_stats.iterrows():
        print(f"  {make:<20s} {int(row['count']):>6d} {row['mape']:>9.1f}% Rs {row['mae']:>9,.0f}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test car parts price prediction model")
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Path to car_parts_final.csv",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive prediction mode only",
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  Car Parts Price Prediction — Testing")
    print("=" * 55)

    # Load model
    bundle = load_model()
    if bundle is None:
        return

    meta = load_meta()
    if meta:
        print(f"\n  Model: {meta.get('model_name', '?')} ({meta.get('model_type', '?')})")
        print(f"  Features: {meta.get('feature_count', '?')}")
        print(f"  Trained on: {meta.get('training_samples', '?')} samples")

    if args.interactive:
        interactive_mode(bundle)
        return

    # Full test suite
    test_holdout(args.data, bundle)
    test_known_parts(bundle)
    analyze_predictions(args.data, bundle)

    print(f"\n{'=' * 55}")
    print(f"  ALL TESTS COMPLETE")
    print(f"{'=' * 55}")
    print(f"\n  For interactive mode: python test.py --interactive\n")


if __name__ == "__main__":
    main()
