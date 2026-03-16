# Car Parts Price Prediction Model

Machine learning model that predicts car part prices (PKR) for Pakistani auto parts based on part name, type, brand, compatible car make/model, and year range.

## Model Details

| Property | Value |
|---|---|
| **Algorithm** | XGBoost (best of 4 trained) |
| **Target** | `price_pkr` (log-transformed) |
| **Features** | 321 (6 categorical + 15 numeric + 300 TF-IDF) |
| **Training samples** | 3,092 |
| **Test samples** | 774 |
| **Data source** | AutoStore.pk (scraped via `cost_prediction/`) |

## Performance

| Metric | Value |
|---|---|
| **R² score** | 0.7918 |
| **MAE** | Rs 11,641 |
| **Median AE** | Rs 1,222 |
| **RMSE** | Rs 36,495 |
| **MAPE** | 48.6% |

Price range in training data: Rs 250 – Rs 800,000 (outliers removed at 1st/99th percentile).

## Features Used

**Categorical** (ordinal-encoded):
- `part_brand` — manufacturer (Denso, Guard, NGK, etc.)
- `scraped_category` — product category (Spark Plugs, Oil Filter, etc.)
- `compatible_make` — car make (Toyota, Honda, Suzuki, Universal)
- `compatible_model` — car model (Corolla, Civic, City, etc.)
- `condition` — New / Used / Refurbished
- `part_type` — specific type (Iridium Spark Plug, LED Head Light, etc.)

**Numeric**:
- `year_from`, `year_to` — compatibility year range
- `price_original_pkr` — original price before discount
- `alternatives_count` — number of alternative parts
- `year_span`, `has_year`, `is_universal`, `has_discount`, `name_word_count` — engineered
- Frequency encodings for make, model, brand, category
- Mean-price encodings for category and make

**Text** (TF-IDF, 300 features):
- `part_name` — product name (bigrams, min_df=2)

**Excluded**: `available`, `website`, `product_url`, `description`

## Files

```
cost_prediction_model/
├── train.py           # Training script
├── test.py            # Testing & evaluation script
├── model.pkl          # Saved model + encoders (pickle)
├── model_meta.json    # Metadata & metrics
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Usage

### Train
```bash
python train.py                                    # default settings
python train.py --data path/to/car_parts_final.csv # custom data
python train.py --test-size 0.25                   # 25% test split
python train.py --no-text                          # skip TF-IDF features
```

### Test
```bash
python test.py                    # full test suite (holdout + known parts + analysis)
python test.py --interactive      # interactive mode — type a part, get a price
```

### API
See `../cost_prediction_api/` for the FastAPI service and Hugging Face deployment.

## Requirements

```
pip install -r requirements.txt
```

- Python 3.10+
- scikit-learn >= 1.3
- xgboost >= 2.0
- pandas >= 2.0
- numpy >= 1.24
