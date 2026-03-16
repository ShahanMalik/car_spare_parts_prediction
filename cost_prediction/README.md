# Car Parts Price Dataset — Pakistan

This folder scrapes car parts & accessories prices from Pakistani e-commerce websites and builds a clean, structured dataset.

## What This Folder Does

Runs two programs in order:

1. **scraper.py** — visits AutoStore.pk, collects product listings, uses ChatGPT to identify compatible cars
2. **dataset_builder.py** — cleans the raw data, links alternatives, and produces the final dataset

**Files produced after running both programs:**

| File | Description |
|------|-------------|
| `car_parts_raw.json` | Raw scraped data (from scraper.py) |
| `car_parts_raw.csv` | Same raw data as CSV |
| **`car_parts_final.csv`** | Clean final dataset (from dataset_builder.py) ✓ |
| `car_parts_final.json` | Same final data in JSON |

---

## Quick Start

```bash
# 1. Install dependencies (first time only)
pip install -r requirements.txt

# 2. Scrape car parts data
python scraper.py

# 3. Clean and finalize the dataset
python dataset_builder.py --enrich
```

---

## Step-by-Step Guide

### Step 0 — Set your API key

Edit the `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=sk-proj-...your-key...
```

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Scrape car parts

```bash
python scraper.py
```

**What happens:**
- Visits AutoStore.pk category pages (spark plugs, oil filters, lights, etc.)
- Extracts product name, price, URL, availability
- Uses ChatGPT API to identify: car make, model, year range, part brand
- Saves raw data to `car_parts_raw.json` and `car_parts_raw.csv`
- Takes ~15-30 minutes depending on internet speed

**Options:**
```bash
python scraper.py --categories car-parts/spark-plugs car-filters/oil-filters  # specific categories only
python scraper.py --skip-enrichment    # skip ChatGPT step (faster, less data)
python scraper.py --resume             # resume from last checkpoint
python scraper.py --fetch-details      # also fetch product descriptions (slower)
```

### Step 3 — Build final dataset

```bash
python dataset_builder.py --enrich
```

**What happens:**
- Reads `car_parts_raw.json`
- Normalises and cleans all fields
- Removes duplicates
- Re-enriches parts still missing car make/model info
- Links alternatives (same part type for the same car)
- Saves to `car_parts_final.csv` and `car_parts_final.json`

---

## Dataset Fields

| Field | Description | Example |
|-------|-------------|---------|
| `part_name` | Product name as listed on the website | Toyota Corolla XLI IK16 Denso Iridium Spark Plug |
| `part_type` | Specific part type (GPT-extracted) | Iridium Spark Plug |
| `part_brand` | Brand/manufacturer of the part | Denso |
| `scraped_category` | Website category it was found in | Spark Plugs |
| `compatible_make` | Car manufacturer it fits | Toyota |
| `compatible_model` | Specific car model | Corolla XLI |
| `year_from` | Start year of compatibility | 2017 |
| `year_to` | End year of compatibility | 2020 |
| `price_pkr` | Price in Pakistani Rupees | 1700 |
| `price_original_pkr` | Original price before discount (0 if none) | 0 |
| `condition` | New / Used / Refurbished | New |
| `available` | Whether in stock | true |
| `website` | Source website | AutoStore.pk |
| `product_url` | Direct link to the product | https://www.autostore.pk/shop/... |
| `description` | Product description (if --fetch-details used) | |
| `alternatives_count` | Number of alternative parts found | 3 |

---

## Sources

| Website | URL | Type | Parts |
|---------|-----|------|-------|
| AutoStore.pk | autostore.pk | E-commerce (WooCommerce) | ~3000+ products |

Categories scraped:
- **Core Parts**: Spark Plugs, Wiper Blades, Side Mirrors, Headlights, Backlights, Mud Flaps
- **Filters**: Oil, Air, Fuel, AC Cabin
- **Oils & Fluids**: Engine Oil, Brake Oil, Transmission Oil, Steering Oil
- **Lighting**: LED Headlights, Fog Lamps, Tail Lights
- **Electronics**: Cameras, Stereos, Speakers, Security Systems
- **Accessories**: Floor Mats, Seat Covers, Body Kits, Top Covers
- **Tyres & Wheels, Batteries, Tools**

---

## ChatGPT API Usage

Uses the following models (in order of preference):
1. `gpt-4.1-mini-2025-04-14` (primary)
2. `gpt-4o-mini-2024-07-18` (fallback)
3. `gpt-4.1-nano-2025-04-14` (fallback)

Approximate token usage: ~150,000–300,000 tokens for a full dataset of ~3000 parts.

---

## Troubleshooting

**"OPENAI_API_KEY not set"** → Edit `.env` and add your key

**Scraper gets stuck** → Use `--resume` flag to continue from checkpoint

**Too many "Unknown" values** → Run `python dataset_builder.py --enrich` again

**Want only specific categories** → Use `--categories` flag with category slugs from `config.py`
