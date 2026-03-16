"""
Car Parts Price Prediction API
================================
FastAPI service for predicting Pakistani car part prices.
Designed for deployment on Hugging Face Spaces (Docker).

Endpoints:
  GET  /                → Health check + model info
  POST /predict         → Single part price prediction
  POST /predict/batch   → Batch predictions (up to 50)
  GET  /metadata        → Model metadata & metrics
  GET  /categories      → Valid categories, makes, models, brands
"""

import json
import os
import pickle
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
META_PATH = os.getenv("META_PATH", "model/model_meta.json")

TARGET = "price_pkr"
CAT_FEATURES = ["part_brand", "scraped_category", "compatible_make", "compatible_model", "condition"]
HIGH_CARD_CAT = "part_type"
NUM_FEATURES = ["year_from", "year_to", "price_original_pkr", "alternatives_count"]
TEXT_FEATURE = "part_name"

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
model_bundle = {}
model_meta = {}


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────
class PartInput(BaseModel):
    """Input schema for a single car part."""
    part_name: str = Field(..., min_length=1, max_length=300, description="Product name")
    part_type: str = Field(default="Unknown", max_length=200, description="Part type (e.g. Oil Filter, Spark Plug)")
    part_brand: str = Field(default="Unknown", max_length=100, description="Brand (e.g. Denso, Guard)")
    scraped_category: str = Field(default="Unknown", max_length=100, description="Category (e.g. Spark Plugs)")
    compatible_make: str = Field(default="Universal", max_length=100, description="Car make (e.g. Toyota)")
    compatible_model: str = Field(default="Universal", max_length=100, description="Car model (e.g. Corolla)")
    year_from: Optional[int] = Field(default=0, ge=0, le=2030, description="Compatibility start year")
    year_to: Optional[int] = Field(default=0, ge=0, le=2030, description="Compatibility end year")
    price_original_pkr: float = Field(default=0, ge=0, description="Original price before discount")
    alternatives_count: int = Field(default=0, ge=0, description="Number of alternatives")
    condition: str = Field(default="New", description="New / Used / Refurbished")

    model_config = {"json_schema_extra": {
        "examples": [{
            "part_name": "Denso Iridium Spark Plug Toyota Corolla",
            "part_type": "Iridium Spark Plug",
            "part_brand": "Denso",
            "scraped_category": "Spark Plugs",
            "compatible_make": "Toyota",
            "compatible_model": "Corolla",
            "year_from": 2014,
            "year_to": 2020,
            "condition": "New"
        }]
    }}


class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    predicted_price_pkr: float
    confidence_range_low: float
    confidence_range_high: float
    currency: str = "PKR"
    model_name: str
    part_name: str


class BatchInput(BaseModel):
    """Input for batch predictions."""
    parts: list[PartInput] = Field(..., min_length=1, max_length=50)


class BatchResponse(BaseModel):
    """Response for batch predictions."""
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
    feature_count: int
    training_samples: int
    r2_score: float


class CategoriesResponse(BaseModel):
    """Available categories, makes, models, brands."""
    categories: list[str]
    makes: list[str]
    models: list[str]
    brands: list[str]


# ─────────────────────────────────────────────
# Feature engineering (mirrors train.py)
# ─────────────────────────────────────────────
def prepare_features(part: PartInput) -> np.ndarray:
    """Convert a PartInput into feature vector."""
    encoders = model_bundle["encoders"]

    data = {
        "part_name": part.part_name,
        "part_type": part.part_type or "Unknown",
        "part_brand": part.part_brand or "Unknown",
        "scraped_category": part.scraped_category or "Unknown",
        "compatible_make": part.compatible_make or "Universal",
        "compatible_model": part.compatible_model or "Universal",
        "year_from": part.year_from or 0,
        "year_to": part.year_to or 0,
        "price_original_pkr": part.price_original_pkr or 0,
        "alternatives_count": part.alternatives_count or 0,
        "condition": part.condition or "New",
    }
    df = pd.DataFrame([data])

    # Engineered features
    df["year_span"] = (df["year_to"] - df["year_from"]).clip(lower=0)
    df["has_year"] = ((df["year_from"] > 0) | (df["year_to"] > 0)).astype(int)
    df["is_universal"] = (df["compatible_make"].str.lower() == "universal").astype(int)
    df["has_discount"] = (df["price_original_pkr"] > 0).astype(int)
    df["name_word_count"] = df["part_name"].str.split().str.len().fillna(0).astype(int)

    # Frequency encoding from training lookups
    freq_maps = encoders.get("freq_maps", {})
    for col in ["compatible_make", "compatible_model", "part_brand", "scraped_category"]:
        fmap = freq_maps.get(col, {})
        df[f"{col}_freq"] = df[col].map(fmap).fillna(0)

    # Price mean encoding from training lookups
    price_mean_maps = encoders.get("price_mean_maps", {})
    global_mean = encoders.get("global_mean_price", 10000)
    for col in ["scraped_category", "compatible_make"]:
        pmap = price_mean_maps.get(col, {})
        df[f"{col}_price_mean"] = np.log1p(df[col].map(pmap).fillna(global_mean))

    # Categorical encoding
    cat_encoded = []
    for col in CAT_FEATURES:
        enc = encoders[col]
        vals = enc.transform(df[[col]])
        cat_encoded.append(vals)
    pt_enc = encoders[HIGH_CARD_CAT]
    cat_encoded.append(pt_enc.transform(df[[HIGH_CARD_CAT]]))
    cat_matrix = np.hstack(cat_encoded)

    # Numeric
    eng_features = [
        "year_span", "has_year", "is_universal", "has_discount", "name_word_count",
        "compatible_make_freq", "compatible_model_freq", "part_brand_freq", "scraped_category_freq",
        "scraped_category_price_mean", "compatible_make_price_mean",
    ]
    num_cols = NUM_FEATURES + eng_features
    num_matrix = df[num_cols].values.astype(float)

    # TF-IDF
    if "tfidf" in encoders:
        text_matrix = encoders["tfidf"].transform(df["part_name"]).toarray()
    else:
        text_matrix = np.empty((1, 0))

    return np.hstack([cat_matrix, num_matrix, text_matrix])


def predict_price(part: PartInput) -> PredictionResponse:
    """Run prediction for a single part."""
    X = prepare_features(part)
    model = model_bundle["model"]

    y_log = model.predict(X)[0]
    predicted = max(0, float(np.expm1(y_log)))

    # Confidence range: ±median_ae from training metrics
    median_ae = model_meta.get("best_model_metrics", {}).get("median_ae", 1222)
    low = max(0, predicted - median_ae * 1.5)
    high = predicted + median_ae * 1.5

    return PredictionResponse(
        predicted_price_pkr=round(predicted, 0),
        confidence_range_low=round(low, 0),
        confidence_range_high=round(high, 0),
        currency="PKR",
        model_name=model_meta.get("model_name", "XGBoost"),
        part_name=part.part_name,
    )


# ─────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model_bundle, model_meta

    logger.info(f"Loading model from {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model_bundle = pickle.load(f)
    logger.info("Model loaded successfully")

    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            model_meta = json.load(f)
        logger.info("Metadata loaded")

    yield

    logger.info("Shutting down")


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Car Parts Price Prediction API",
    description=(
        "Predict Pakistani car part prices using an XGBoost model trained on "
        "3,900+ parts from AutoStore.pk. Provide part details and get instant "
        "price estimates in PKR."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
async def health():
    """Health check and model info."""
    return HealthResponse(
        status="ok",
        model_loaded=bool(model_bundle),
        model_name=model_meta.get("model_name", "Unknown"),
        feature_count=model_meta.get("feature_count", 0),
        training_samples=model_meta.get("training_samples", 0),
        r2_score=model_meta.get("best_model_metrics", {}).get("r2", 0),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(part: PartInput):
    """Predict price for a single car part."""
    if not model_bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return predict_price(part)
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(batch: BatchInput):
    """Predict prices for multiple car parts (max 50)."""
    if not model_bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        results = [predict_price(part) for part in batch.parts]
        return BatchResponse(predictions=results, count=len(results))
    except Exception as exc:
        logger.error(f"Batch prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")


@app.get("/metadata")
async def metadata():
    """Return full model metadata and metrics."""
    if not model_meta:
        raise HTTPException(status_code=503, detail="Metadata not available")
    return model_meta


@app.get("/categories", response_model=CategoriesResponse)
async def categories():
    """Return valid categories, makes, models, and brands from training data."""
    if not model_bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")

    encoders = model_bundle["encoders"]
    freq_maps = encoders.get("freq_maps", {})

    return CategoriesResponse(
        categories=sorted(freq_maps.get("scraped_category", {}).keys()),
        makes=sorted(freq_maps.get("compatible_make", {}).keys()),
        models=sorted(freq_maps.get("compatible_model", {}).keys()),
        brands=sorted(freq_maps.get("part_brand", {}).keys()),
    )
