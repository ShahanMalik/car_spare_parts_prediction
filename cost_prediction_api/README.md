---
title: Car Parts Price Prediction
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Car Parts Price Prediction API

Predict Pakistani car part prices using an XGBoost model trained on 3,900+ parts from AutoStore.pk.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + model info |
| `POST` | `/predict` | Single part price prediction |
| `POST` | `/predict/batch` | Batch predictions (up to 50) |
| `GET` | `/metadata` | Model metadata & metrics |
| `GET` | `/categories` | Valid categories, makes, models |
| `GET` | `/docs` | Interactive Swagger UI |

## Example

```bash
curl -X POST /predict \
  -H "Content-Type: application/json" \
  -d '{
    "part_name": "Denso Iridium Spark Plug Toyota Corolla",
    "part_type": "Iridium Spark Plug",
    "part_brand": "Denso",
    "scraped_category": "Spark Plugs",
    "compatible_make": "Toyota",
    "compatible_model": "Corolla",
    "year_from": 2014,
    "year_to": 2020
  }'
```

## Model Performance

- **R² Score**: 0.79
- **Median Absolute Error**: Rs 1,222
- **MAE**: Rs 11,641
