"""
Car Parts Price Prediction Model – Training Script
=====================================================
Trains a model to predict car part prices (price_pkr) from features like
part_type, brand, category, compatible car make/model, year range, etc.

Models trained (picks the best):
  1. Random Forest
  2. Gradient Boosting (HistGradientBoosting)
  3. XGBoost (if installed)

Usage:
    python train.py                          # train with defaults
    python train.py --data ../cost_prediction/car_parts_final.csv
    python train.py --no-text                # skip text features
    python train.py --test-size 0.25         # custom test split
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEFAULT_DATA = os.path.join(
    os.path.dirname(__file__), "..", "cost_prediction", "car_parts_final.csv"
)
MODEL_DIR = os.path.dirname(__file__) or "."
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

# Columns to drop (not used as features)
DROP_COLS = [
    "available",        # user doesn't want this
    "website",          # constant (AutoStore.pk)
    "product_url",      # unique identifier, not a feature
    "description",      # mostly null, use part_name instead
]

# Target
TARGET = "price_pkr"

# Categorical features (will be ordinal-encoded)
CAT_FEATURES = [
    "part_brand",
    "scraped_category",
    "compatible_make",
    "compatible_model",
    "condition",
]

# Numeric features
NUM_FEATURES = [
    "year_from",
    "year_to",
    "price_original_pkr",
    "alternatives_count",
]

# Text feature (TF-IDF)
TEXT_FEATURE = "part_name"

# High-cardinality categorical (special handling)
HIGH_CARD_CAT = "part_type"


# ─────────────────────────────────────────────
# Data Loading & Cleaning
# ─────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, drop unwanted columns, handle missing values."""
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns from {path}")

    # Drop unwanted columns
    existing_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=existing_drop, inplace=True)

    # Drop rows with no price or price <= 0
    df = df[df[TARGET] > 0].copy()

    # Remove extreme price outliers (IQR method)
    q1 = df[TARGET].quantile(0.01)
    q99 = df[TARGET].quantile(0.99)
    before = len(df)
    df = df[(df[TARGET] >= q1) & (df[TARGET] <= q99)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Removed {dropped} extreme outliers (price < Rs {q1:,.0f} or > Rs {q99:,.0f})")

    # Fill missing values
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

    # Frequency encoding: how common is this make, model, brand, category
    freq_maps = {}
    for col in ["compatible_make", "compatible_model", "part_brand", "scraped_category"]:
        freq = df[col].value_counts(normalize=True).to_dict()
        freq_maps[col] = freq
        df[f"{col}_freq"] = df[col].map(freq).fillna(0)

    # Mean price by category (target encoding proxy)
    price_mean_maps = {}
    for col in ["scraped_category", "compatible_make"]:
        mean_map = df.groupby(col)[TARGET].mean().to_dict()
        price_mean_maps[col] = mean_map
        df[f"{col}_price_mean"] = np.log1p(df[col].map(mean_map).fillna(df[TARGET].mean()))

    # Store lookup tables as attributes for saving later
    df.attrs["freq_maps"] = freq_maps
    df.attrs["price_mean_maps"] = price_mean_maps
    df.attrs["global_mean_price"] = df[TARGET].mean()

    print(f"  After cleaning: {len(df)} rows")
    return df


# ─────────────────────────────────────────────
# Feature Engineering Pipeline
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame, use_text: bool = True):
    """
    Build X (features) and y (target) from the dataframe.
    Returns X, y, feature_names, encoders dict.
    """
    y = np.log1p(df[TARGET].values)  # Log-transform target for better distribution

    # ── Categorical encoding ──
    encoders = {}
    cat_encoded = []

    for col in CAT_FEATURES:
        le = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        vals = le.fit_transform(df[[col]])
        cat_encoded.append(vals)
        encoders[col] = le

    # High-cardinality: part_type → ordinal with frequency grouping
    pt_le = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    pt_vals = pt_le.fit_transform(df[[HIGH_CARD_CAT]])
    cat_encoded.append(pt_vals)
    encoders[HIGH_CARD_CAT] = pt_le

    cat_matrix = np.hstack(cat_encoded)
    cat_names = CAT_FEATURES + [HIGH_CARD_CAT]

    # ── Numeric features ──
    eng_features = [
        "year_span", "has_year", "is_universal", "has_discount", "name_word_count",
        "compatible_make_freq", "compatible_model_freq", "part_brand_freq", "scraped_category_freq",
        "scraped_category_price_mean", "compatible_make_price_mean",
    ]
    num_cols = NUM_FEATURES + eng_features
    num_matrix = df[num_cols].values.astype(float)

    # ── Text features (TF-IDF on part_name) ──
    if use_text:
        tfidf = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )
        text_matrix = tfidf.fit_transform(df[TEXT_FEATURE]).toarray()
        encoders["tfidf"] = tfidf
        tfidf_names = [f"tfidf_{i}" for i in range(text_matrix.shape[1])]
    else:
        text_matrix = np.empty((len(df), 0))
        tfidf_names = []

    # ── Combine all features ──
    X = np.hstack([cat_matrix, num_matrix, text_matrix])
    feature_names = cat_names + num_cols + tfidf_names

    # Store lookup tables into encoders for use at test time
    encoders["freq_maps"] = df.attrs.get("freq_maps", {})
    encoders["price_mean_maps"] = df.attrs.get("price_mean_maps", {})
    encoders["global_mean_price"] = df.attrs.get("global_mean_price", 0)

    print(f"  Features: {X.shape[1]} ({len(cat_names)} cat + {len(num_cols)} num + {len(tfidf_names)} text)")
    return X, y, feature_names, encoders


# ─────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────
def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and return the best one."""

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=800,
            max_depth=15,
            learning_rate=0.03,
            min_samples_leaf=3,
            l2_regularization=0.05,
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.03,
            min_samples_split=3,
            subsample=0.85,
            random_state=42,
        ),
    }

    # Try XGBoost if available
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=800,
            max_depth=12,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            gamma=0.1,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        pass

    results = {}
    best_score = -999
    best_name = None
    best_model = None

    print(f"\n  Training {len(models)} models...\n")

    for name, model in models.items():
        print(f"  [{name}]")
        model.fit(X_train, y_train)

        # Predictions (in log space)
        y_pred_log = model.predict(X_test)

        # Convert back to original scale
        y_pred = np.expm1(y_pred_log)
        y_actual = np.expm1(y_test)

        # Clip negative predictions
        y_pred = np.clip(y_pred, 0, None)

        # Metrics
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        mape = mean_absolute_percentage_error(y_actual, y_pred) * 100

        # Median absolute error (more robust)
        median_ae = np.median(np.abs(y_actual - y_pred))

        results[name] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "median_ae": median_ae,
        }

        print(f"    R²:         {r2:.4f}")
        print(f"    MAE:        Rs {mae:,.0f}")
        print(f"    Median AE:  Rs {median_ae:,.0f}")
        print(f"    RMSE:       Rs {rmse:,.0f}")
        print(f"    MAPE:       {mape:.1f}%")
        print()

        if r2 > best_score:
            best_score = r2
            best_name = name
            best_model = model

    print(f"  Best model: {best_name} (R² = {best_score:.4f})")
    return best_model, best_name, results


# ─────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────
def print_feature_importance(model, feature_names, top_n=20):
    """Print top feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        print(f"\n  Top {top_n} Feature Importances:")
        print(f"  {'─' * 45}")
        for rank, idx in enumerate(indices, 1):
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            print(f"  {rank:3d}. {name:<30s} {importances[idx]:.4f}")


# ─────────────────────────────────────────────
# Save Model
# ─────────────────────────────────────────────
def save_model(model, encoders, feature_names, model_name, results, df_stats):
    """Save model and metadata."""
    # Save model + encoders
    bundle = {
        "model": model,
        "encoders": encoders,
        "feature_names": feature_names,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Model saved: {MODEL_PATH}")

    # Save metadata (JSON-serializable)
    meta = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "feature_count": len(feature_names),
        "training_samples": df_stats["train_size"],
        "test_samples": df_stats["test_size"],
        "total_samples": df_stats["total"],
        "target": TARGET,
        "target_transform": "log1p",
        "metrics": {
            name: {k: round(v, 4) for k, v in m.items()}
            for name, m in results.items()
        },
        "best_model_metrics": {
            k: round(v, 4) for k, v in results[model_name].items()
        },
        "categorical_features": CAT_FEATURES + [HIGH_CARD_CAT],
        "numeric_features": NUM_FEATURES + [
            "year_span", "has_year", "is_universal", "has_discount", "name_word_count",
            "compatible_make_freq", "compatible_model_freq", "part_brand_freq", "scraped_category_freq",
            "scraped_category_price_mean", "compatible_make_price_mean",
        ],
        "text_feature": TEXT_FEATURE,
        "excluded_columns": DROP_COLS,
        "price_stats": df_stats["price_stats"],
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved: {META_PATH}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train car parts price prediction model")
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Path to car_parts_final.csv",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Skip TF-IDF text features from part_name",
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  Car Parts Price Prediction — Training")
    print("=" * 55 + "\n")

    # ── Load & Clean ──
    print("[1] Loading data...")
    df = load_and_clean(args.data)

    price_stats = {
        "mean": round(df[TARGET].mean(), 2),
        "median": round(df[TARGET].median(), 2),
        "min": round(df[TARGET].min(), 2),
        "max": round(df[TARGET].max(), 2),
        "std": round(df[TARGET].std(), 2),
    }
    print(f"  Price range: Rs {price_stats['min']:,.0f} – Rs {price_stats['max']:,.0f}")
    print(f"  Price median: Rs {price_stats['median']:,.0f}, mean: Rs {price_stats['mean']:,.0f}")

    # ── Build Features ──
    print("\n[2] Building features...")
    use_text = not args.no_text
    X, y, feature_names, encoders = build_features(df, use_text=use_text)

    # ── Train/Test Split ──
    print(f"\n[3] Splitting data (test={args.test_size:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # ── Train Models ──
    print("\n[4] Training models...")
    best_model, best_name, results = train_models(X_train, y_train, X_test, y_test)

    # ── Feature Importance ──
    print_feature_importance(best_model, feature_names)

    # ── Save ──
    print("\n[5] Saving model...")
    df_stats = {
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "total": len(df),
        "price_stats": price_stats,
    }
    save_model(best_model, encoders, feature_names, best_name, results, df_stats)

    # ── Summary ──
    best_metrics = results[best_name]
    print(f"\n{'=' * 55}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 55}")
    print(f"  Best model:    {best_name}")
    print(f"  R² score:      {best_metrics['r2']:.4f}")
    print(f"  MAE:           Rs {best_metrics['mae']:,.0f}")
    print(f"  Median AE:     Rs {best_metrics['median_ae']:,.0f}")
    print(f"  MAPE:          {best_metrics['mape']:.1f}%")
    print(f"\n  Files:")
    print(f"    {MODEL_PATH}")
    print(f"    {META_PATH}")
    print(f"\n  Next: python test.py")
    print()


if __name__ == "__main__":
    main()
