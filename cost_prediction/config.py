"""
Configuration for Pakistan Car Parts Price Scraper
====================================================
All settings in one place. Adjust as needed.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"     # primary model
OPENAI_FALLBACK_MODELS = [                    # fallback chain
    "gpt-4o-mini-2024-07-18",
    "gpt-4.1-nano-2025-04-14",
]
OPENAI_MAX_TOKENS = 2048
OPENAI_TEMPERATURE = 0
GPT_MAX_RETRIES = 5           # max retries per batch on API failure
GPT_RETRY_DELAY = 2           # initial delay between retries (seconds)
GPT_RECHECK_PASSES = 2        # re-enrichment passes for unknowns
GPT_BATCH_SIZE = 15            # parts per GPT call

# ─────────────────────────────────────────────
# Scraping
# ─────────────────────────────────────────────
REQUEST_TIMEOUT = 25  # seconds
MIN_DELAY = 1.0       # minimum delay between requests (seconds)
MAX_DELAY = 2.5       # maximum delay between requests (seconds)
MAX_RETRIES = 3       # retries per failed request
CHECKPOINT_EVERY = 50 # save checkpoint every N parts

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ─────────────────────────────────────────────
# Source: AutoStore.pk  (WooCommerce – server-rendered)
# ─────────────────────────────────────────────
AUTOSTORE_BASE = "https://www.autostore.pk"

# category-slug  →  friendly display name
# Scraper visits: {BASE}/category/{slug}/page/{N}/
AUTOSTORE_CATEGORIES = {
    # ── Core Car Parts ──────────────────────
    "car-parts/back-lights":        "Back Lights",
    "car-parts/head-lights":        "Head Lights",
    "car-parts/mud-flap":           "Mud Flaps",
    "car-parts/side-mirrors":       "Side Mirrors",
    "car-parts/spark-plugs":        "Spark Plugs",
    "car-parts/wiper-blades":       "Wiper Blades",

    # ── Filters ─────────────────────────────
    "car-filters/ac-filter":        "AC Cabin Filter",
    "car-filters/air-filters":      "Air Filter",
    "car-filters/fuel-filter":      "Fuel Filter",
    "car-filters/oil-filters":      "Oil Filter",

    # ── Oils & Fluids ───────────────────────
    "oils-and-additives/brake-oils":        "Brake Oil",
    "oils-and-additives/engine-oil":        "Engine Oil",
    "oils-and-additives/oil-additives":     "Oil Additives",
    "oils-and-additives/steering-oil":      "Steering Oil",
    "oils-and-additives/transmission-oils": "Transmission Oil",

    # ── Interior Accessories ────────────────
    "car-accessories/mats/floor-mats":                                      "Floor Mats",
    "car-accessories/mats/dash-mats":                                       "Dash Mats",
    "car-accessories/interior-accessories/seat-covers-and-cushions":         "Seat Covers",
    "car-accessories/interior-accessories/steering-wheel-accessories":       "Steering Accessories",
    "car-accessories/interior-accessories/sun-shades":                       "Sun Shades",

    # ── Exterior Accessories ────────────────
    "car-accessories/exterior-accessories/top-covers":          "Car Top Covers",
    "car-accessories/exterior-accessories/body-kits":           "Body Kits",
    "car-accessories/exterior-accessories/chrome-products":     "Chrome Parts",
    "car-accessories/exterior-accessories/horns":               "Horns",
    "car-accessories/exterior-accessories/mufflers":            "Mufflers",
    "car-accessories/exterior-accessories/performance-parts":   "Performance Parts",
    "car-accessories/exterior-accessories/airpress":            "Air Press / Wind Deflectors",

    # ── Lighting ────────────────────────────
    "led-lights/drl-fog-lamps":         "DRL & Fog Lamps",
    "led-lights/hid-led-smd-lights":    "HID / LED / SMD Lights",
    "led-lights/led-head-lights":       "LED Head Lights",
    "led-lights/led-tail-lights":       "LED Tail Lights",

    # ── Electronics ─────────────────────────
    "car-electronics/car-camera":               "Car Cameras",
    "car-electronics/stereo-systems":           "Stereo Systems",
    "car-electronics/speakers-and-amplifiers":  "Speakers & Amplifiers",
    "car-electronics/security-systems":         "Security Systems",

    # ── Tyres & Wheels ──────────────────────
    "tyres-and-wheels/tyres":           "Tyres",
    "tyres-and-wheels/wheel-covers":    "Wheel Covers",

    # ── Batteries ───────────────────────────
    "batteries":  "Batteries",

    # ── Tools ───────────────────────────────
    "tools/basic-tools":                "Basic Tools",
    "tools/tool-kits":                  "Tool Kits",
    "tools/air-compressor":             "Air Compressors",
    "tools/cable-and-jump-starter":     "Jump Starters",

    # ── Car Care ────────────────────────────
    "car-care/carcare-exterior/tire-care":   "Tire Care Products",
    "car-care/coolants":                     "Coolants",
}

MAX_PAGES_PER_CATEGORY = 60   # safety limit

# ─────────────────────────────────────────────
# Output Paths
# ─────────────────────────────────────────────
RAW_JSON        = "car_parts_raw.json"
RAW_CSV         = "car_parts_raw.csv"
FINAL_JSON      = "car_parts_final.json"
FINAL_CSV       = "car_parts_final.csv"
CHECKPOINT_DIR  = "checkpoints"
LOG_FILE        = "scraper.log"
