"""
Pakistan Car Parts Web Scraper
================================
Scrapes car parts & accessories from Pakistani e-commerce websites:
  - AutoStore.pk  (WooCommerce store – primary source)

Uses OpenAI GPT to enrich each part with:
  - Compatible car make / model / year range
  - Standardised part category & brand

Usage:
    python scraper.py                        # full scrape (all categories)
    python scraper.py --categories spark-plugs oil-filters   # specific slugs
    python scraper.py --skip-enrichment      # skip GPT step
    python scraper.py --resume               # resume from last checkpoint
    python scraper.py --fetch-details        # also visit each product page
"""

import argparse
import json
import logging
import os
import re
import time

from bs4 import BeautifulSoup
from collections import defaultdict
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

import config

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Checkpoint Manager
# ─────────────────────────────────────────────
class CheckpointManager:
    """Save / load scraping progress so we can resume after interruption."""

    def __init__(self, source_name: str):
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        self.path = os.path.join(config.CHECKPOINT_DIR, f"{source_name}.json")

    def save(self, data: list, metadata: dict = None):
        payload = {
            "timestamp": datetime.now().isoformat(),
            "count": len(data),
            "metadata": metadata or {},
            "data": data,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Checkpoint saved → {self.path}  ({len(data)} items)")

    def load(self) -> tuple:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            data = payload.get("data", [])
            meta = payload.get("metadata", {})
            logger.info(f"Checkpoint loaded ← {self.path}  ({len(data)} items)")
            return data, meta
        except (FileNotFoundError, json.JSONDecodeError):
            return [], {}

    def exists(self) -> bool:
        return os.path.isfile(self.path)


# ─────────────────────────────────────────────
# HTTP Client with retry + polite delays
# ─────────────────────────────────────────────
class HttpClient:
    """Requests session with retry, rate limiting, and polite delays."""

    def __init__(self):
        import requests
        self.session = requests.Session()
        self.session.headers.update(config.HEADERS)
        self._last_request_time = 0
        self._request_count = 0

    def get(self, url: str, retries: int = None):
        retries = retries or config.MAX_RETRIES
        import requests as req_lib

        for attempt in range(retries):
            self._rate_limit()
            try:
                resp = self.session.get(url, timeout=config.REQUEST_TIMEOUT)
                self._request_count += 1
                # Return 404 responses without retrying (used for pagination)
                if resp.status_code == 404:
                    return resp
                resp.raise_for_status()
                return resp
            except req_lib.RequestException as exc:
                wait = config.MIN_DELAY * (2 ** attempt)
                logger.warning(
                    f"Request failed ({attempt + 1}/{retries}): {url} — {exc}. "
                    f"Retrying in {wait:.1f}s"
                )
                time.sleep(wait)

        logger.error(f"All {retries} retries failed: {url}")
        return None

    def _rate_limit(self):
        import random
        elapsed = time.time() - self._last_request_time
        delay = random.uniform(config.MIN_DELAY, config.MAX_DELAY)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.time()

    @property
    def request_count(self):
        return self._request_count


# ─────────────────────────────────────────────
# Price Parser
# ─────────────────────────────────────────────
def parse_price(price_str: str) -> float:
    """Extract numeric price from 'Rs 1,700', 'Rs. 35,000', etc."""
    if not price_str:
        return 0.0
    try:
        cleaned = re.sub(r"[^\d.]", "", price_str.replace(",", ""))
        return round(float(cleaned), 2) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0


def parse_price_range(text: str) -> tuple:
    """Parse 'Price range: Rs 3,000 through Rs 14,000' → (3000, 14000)."""
    nums = re.findall(r"[\d,]+(?:\.\d+)?", text.replace(",", ""))
    prices = []
    for n in nums:
        try:
            prices.append(round(float(n), 2))
        except ValueError:
            pass
    if len(prices) >= 2:
        return min(prices), max(prices)
    if len(prices) == 1:
        return prices[0], prices[0]
    return 0.0, 0.0


# ─────────────────────────────────────────────
# AutoStore.pk Scraper
# ─────────────────────────────────────────────
class AutoStoreScraper:
    """
    Scrapes product listings from autostore.pk category pages.

    URL pattern:
        Page 1 : /category/{slug}/
        Page N : /category/{slug}/page/{N}/

    WooCommerce HTML layout per product card:
        <li class="product ...">
          <a href="product-url">
            <img ... />
            <h2 class="woocommerce-loop-product__title">Name</h2>
            <span class="price">
              <span class="woocommerce-Price-amount">Rs X,XXX</span>
            </span>
          </a>
        </li>
    """

    def __init__(self, client: HttpClient):
        self.client = client
        self.checkpoint = CheckpointManager("autostore")

    # ── Parse one listing page ────────────────
    def _parse_page(self, soup: BeautifulSoup, category_name: str, category_slug: str) -> list:
        """Extract products from a single category page."""
        products = []

        # WooCommerce product items
        items = soup.select("li.product, li.type-product")
        if not items:
            # fallback: look for product links with prices
            items = soup.select(".products .product, ul.products > li")

        for item in items:
            try:
                # ── Product URL ──
                link_tag = item.select_one("a[href*='/shop/']")
                if not link_tag:
                    link_tag = item.find("a", href=True)
                if not link_tag:
                    continue
                product_url = link_tag.get("href", "").strip()
                if not product_url.startswith("http"):
                    product_url = config.AUTOSTORE_BASE + product_url

                # ── Product Name ──
                name_tag = item.select_one(
                    "h2.woocommerce-loop-product__title, "
                    "h2.wc-block-components-product-name, "
                    ".product-title, "
                    "h2, h3"
                )
                name = name_tag.get_text(strip=True) if name_tag else ""
                if not name:
                    # try from the link text
                    name = link_tag.get_text(strip=True)
                if not name:
                    continue

                # ── Price ──
                price_tag = item.select_one("span.price, .woocommerce-Price-amount, p.price")
                price_text = price_tag.get_text(" ", strip=True) if price_tag else ""

                price_low = 0.0
                price_high = 0.0

                if "through" in price_text.lower() or "–" in price_text or "-" in price_text:
                    price_low, price_high = parse_price_range(price_text)
                else:
                    # May have sale price + original price
                    del_tag = item.select_one("del .woocommerce-Price-amount, del")
                    ins_tag = item.select_one("ins .woocommerce-Price-amount, ins")
                    if del_tag and ins_tag:
                        price_high = parse_price(del_tag.get_text(strip=True))
                        price_low = parse_price(ins_tag.get_text(strip=True))
                    else:
                        price_low = parse_price(price_text)
                        price_high = price_low

                # ── Availability ──
                out_of_stock = bool(
                    item.select_one(".outofstock, .out-of-stock")
                    or "out of stock" in item.get_text().lower()
                )
                available = not out_of_stock

                products.append({
                    "part_name": name,
                    "scraped_category": category_name,
                    "category_slug": category_slug,
                    "price_pkr": price_low,
                    "price_original_pkr": price_high if price_high != price_low else 0.0,
                    "available": available,
                    "product_url": product_url,
                    "website": "AutoStore.pk",
                    "source_base": config.AUTOSTORE_BASE,
                    # Filled by GPT enrichment later
                    "part_brand": "",
                    "part_type": "",
                    "compatible_make": "",
                    "compatible_model": "",
                    "year_from": None,
                    "year_to": None,
                    "condition": "New",
                    "description": "",
                })
            except Exception as exc:
                logger.debug(f"Error parsing product card: {exc}")
                continue

        return products

    # ── Scrape one category (all pages) ───────
    def scrape_category(self, slug: str, category_name: str) -> list:
        """Paginate through all pages of a single category."""
        all_products = []
        page = 1

        while page <= config.MAX_PAGES_PER_CATEGORY:
            if page == 1:
                url = f"{config.AUTOSTORE_BASE}/category/{slug}/"
            else:
                url = f"{config.AUTOSTORE_BASE}/category/{slug}/page/{page}/"

            resp = self.client.get(url)
            if resp is None or resp.status_code == 404:
                break  # no more pages

            soup = BeautifulSoup(resp.text, "lxml")
            products = self._parse_page(soup, category_name, slug)

            if not products:
                break  # empty page → done

            all_products.extend(products)
            logger.debug(f"  {slug} page {page}: {len(products)} products")
            page += 1

        return all_products

    # ── Fetch individual product details ──────
    def fetch_product_detail(self, product: dict) -> dict:
        """Visit a product page to get description + extra info."""
        resp = self.client.get(product["product_url"])
        if resp is None:
            return product

        soup = BeautifulSoup(resp.text, "lxml")

        # Description
        desc_tag = soup.select_one(
            ".woocommerce-product-details__short-description, "
            ".product-short-description, "
            ".entry-content, "
            "#tab-description"
        )
        if desc_tag:
            product["description"] = desc_tag.get_text(" ", strip=True)[:500]

        # SKU
        sku_tag = soup.select_one(".sku_wrapper .sku, .sku")
        if sku_tag:
            product["sku"] = sku_tag.get_text(strip=True)

        return product
    # ── Fix price from detail page ─────────
    def _fix_price_from_detail(self, product: dict) -> dict:
        """Visit a product's detail page to extract real price & availability."""
        resp = self.client.get(product["product_url"])
        if resp is None or resp.status_code == 404:
            product["available"] = False
            return product

        soup = BeautifulSoup(resp.text, "lxml")

        # Price extraction (multiple strategies)
        price = 0.0

        # Strategy 1: sale price
        ins_tag = soup.select_one("p.price ins .woocommerce-Price-amount, .summary ins .woocommerce-Price-amount")
        if ins_tag:
            price = parse_price(ins_tag.get_text(strip=True))

        # Strategy 2: single price
        if price == 0.0:
            price_el = soup.select_one(
                "p.price .woocommerce-Price-amount, "
                ".summary .price .woocommerce-Price-amount"
            )
            if price_el:
                price = parse_price(price_el.get_text(strip=True))

        # Strategy 3: variable product price range
        if price == 0.0:
            price_block = soup.select_one("p.price, .summary .price")
            if price_block:
                amounts = price_block.select(".woocommerce-Price-amount")
                prices_found = [parse_price(a.get_text(strip=True)) for a in amounts]
                prices_found = [p for p in prices_found if p > 0]
                if prices_found:
                    price = min(prices_found)
                    if len(prices_found) >= 2:
                        product["price_original_pkr"] = max(prices_found)

        # Strategy 4: JSON-LD structured data
        if price == 0.0:
            for script in soup.select('script[type="application/ld+json"]'):
                try:
                    ld = json.loads(script.string or "")
                    offers = ld.get("offers") or {}
                    if isinstance(offers, list):
                        offers = offers[0] if offers else {}
                    p = offers.get("price") or offers.get("lowPrice")
                    if p:
                        price = float(p)
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

        if price > 0:
            product["price_pkr"] = price

        # Out-of-stock detection
        stock_el = soup.select_one(".stock.out-of-stock, .out-of-stock, .outofstock")
        if stock_el:
            product["available"] = False
        summary_area = soup.select_one(".summary")
        if summary_area and "out of stock" in summary_area.get_text(" ").lower():
            product["available"] = False

        # Description
        if not product.get("description"):
            desc_el = soup.select_one(
                ".woocommerce-product-details__short-description, #tab-description"
            )
            if desc_el:
                product["description"] = desc_el.get_text(" ", strip=True)[:500]

        return product
    # ── Main run ──────────────────────────────
    def run(
        self,
        categories: dict = None,
        resume: bool = False,
        fetch_details: bool = False,
    ) -> list:
        """
        Scrape all configured categories.
        Returns list of product dicts.
        """
        cats = categories or config.AUTOSTORE_CATEGORIES
        all_products = []
        completed_slugs = set()

        # Resume from checkpoint?
        if resume and self.checkpoint.exists():
            existing, meta = self.checkpoint.load()
            all_products = existing
            completed_slugs = set(meta.get("completed_slugs", []))
            logger.info(f"Resuming — {len(all_products)} products, "
                        f"{len(completed_slugs)} categories already done")

        slugs = [s for s in cats if s not in completed_slugs]
        pbar = tqdm(slugs, desc="AutoStore.pk categories", unit="cat")

        for slug in pbar:
            cat_name = cats[slug]
            pbar.set_postfix_str(cat_name)
            logger.info(f"Scraping category: {cat_name} ({slug})")

            products = self.scrape_category(slug, cat_name)
            logger.info(f"  → {len(products)} products from {cat_name}")

            # Optionally fetch detail pages
            if fetch_details and products:
                logger.info(f"  Fetching detail pages for {len(products)} products…")
                for i, p in enumerate(products):
                    products[i] = self.fetch_product_detail(p)
                    if (i + 1) % 20 == 0:
                        logger.info(f"    Details: {i + 1}/{len(products)}")

            # Auto-fetch detail pages for zero-price items
            zero_price_items = [p for p in products if not p.get("price_pkr") or p["price_pkr"] <= 0]
            if zero_price_items:
                logger.info(f"  Fetching detail pages for {len(zero_price_items)} zero-price items…")
                for zp in zero_price_items:
                    self._fix_price_from_detail(zp)

            all_products.extend(products)
            completed_slugs.add(slug)

            # Checkpoint
            if len(all_products) % config.CHECKPOINT_EVERY < len(products):
                self.checkpoint.save(
                    all_products,
                    {"completed_slugs": list(completed_slugs)},
                )

        # Final checkpoint
        self.checkpoint.save(
            all_products,
            {"completed_slugs": list(completed_slugs)},
        )
        logger.info(f"AutoStore.pk total: {len(all_products)} products")
        return all_products


# ─────────────────────────────────────────────
# GPT Enrichment
# ─────────────────────────────────────────────
class GPTEnricher:
    """
    Uses OpenAI GPT to extract structured car-part metadata from
    product names.

    Extracts:
      - part_brand   (Denso, Guard, NGK, Genuine, …)
      - part_type    (Iridium Spark Plug, Oil Filter, …)
      - compatible_make   (Toyota, Honda, Suzuki, Universal)
      - compatible_model  (Corolla, Civic, Swift, Universal)
      - year_from / year_to  (int or null)
      - condition    (New / Used / Refurbished)

    Features:
      - Batch processing (GPT_BATCH_SIZE parts per call)
      - Retry with exponential backoff
      - Fallback model chain
      - Re-check passes for unknowns
    """

    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env")
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
        self.fallback_models = config.OPENAI_FALLBACK_MODELS
        self.batch_size = config.GPT_BATCH_SIZE

    # ── Validation ────────────────────────────
    def _is_valid(self, part: dict) -> bool:
        make = (part.get("compatible_make") or "").strip().lower()
        ptype = (part.get("part_type") or "").strip().lower()
        if not make or make in ("unknown", "n/a", "none"):
            return False
        if not ptype or ptype in ("unknown", "n/a", "none"):
            return False
        return True

    # ── GPT Call ──────────────────────────────
    def _call_gpt(self, prompt: str, model: str = None) -> dict:
        """Call GPT and return parsed JSON response."""
        model = model or self.model
        for attempt in range(config.GPT_MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a car parts data extraction expert. "
                                "You return ONLY valid JSON arrays. No markdown, "
                                "no explanation, no extra text."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=config.OPENAI_MAX_TOKENS,
                    temperature=config.OPENAI_TEMPERATURE,
                )
                text = resp.choices[0].message.content.strip()
                # Strip markdown code fences if present
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)
                return json.loads(text)

            except json.JSONDecodeError as exc:
                logger.warning(f"JSON parse error (attempt {attempt + 1}): {exc}")
            except Exception as exc:
                logger.warning(f"GPT API error (attempt {attempt + 1}, model={model}): {exc}")

            delay = config.GPT_RETRY_DELAY * (2 ** attempt)
            time.sleep(delay)

        # Try fallback models
        for fb_model in self.fallback_models:
            logger.info(f"Trying fallback model: {fb_model}")
            try:
                resp = self.client.chat.completions.create(
                    model=fb_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a car parts data extraction expert. "
                                "You return ONLY valid JSON arrays."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=config.OPENAI_MAX_TOKENS,
                    temperature=config.OPENAI_TEMPERATURE,
                )
                text = resp.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)
                return json.loads(text)
            except Exception as exc:
                logger.warning(f"Fallback {fb_model} failed: {exc}")

        logger.error("All GPT models failed for this batch")
        return []

    # ── Prompt Builder ────────────────────────
    def _build_prompt(self, parts: list) -> str:
        lines = []
        for i, p in enumerate(parts, 1):
            lines.append(
                f'{i}. Name: "{p["part_name"]}", '
                f'Category: "{p.get("scraped_category", "")}"'
            )

        parts_text = "\n".join(lines)

        return f"""Given these car parts from a Pakistani auto parts website, extract structured information for each.

For EACH part, return a JSON object with these exact keys:
- "part_brand": manufacturer/brand of the part (e.g. "Denso", "Guard", "NGK", "Genuine", "Rain-X") or "Unknown"
- "part_type": specific type (e.g. "Iridium Spark Plug", "Oil Filter", "LED Fog Lamp") or repeat the category
- "compatible_make": car manufacturer it fits (e.g. "Toyota", "Honda", "Suzuki") or "Universal" if it fits any car
- "compatible_model": specific car model (e.g. "Corolla", "Civic", "City", "Mehran") or "Universal"
- "year_from": start year of compatibility (integer like 2017) or null if not stated
- "year_to": end year of compatibility (integer like 2021) or null if not stated
- "condition": "New" (default for store items), "Used", or "Refurbished"

RULES:
- Extract info ONLY from what is stated in the name. Do NOT guess.
- If a part name mentions multiple cars (e.g. "Aqua/Prius"), use the FIRST one for compatible_model.
- If no specific car is mentioned, set compatible_make and compatible_model to "Universal".
- Return a JSON array with exactly {len(parts)} objects, in the same order as the input.

Parts to process:
{parts_text}"""

    # ── Enrich a batch ────────────────────────
    def enrich_batch(self, parts: list) -> list:
        """Enrich a batch of parts with GPT-extracted metadata."""
        prompt = self._build_prompt(parts)
        result = self._call_gpt(prompt)

        if not isinstance(result, list):
            logger.warning("GPT returned non-list; skipping batch")
            return parts

        for i, enrichment in enumerate(result):
            if i >= len(parts):
                break
            if not isinstance(enrichment, dict):
                continue
            parts[i]["part_brand"] = enrichment.get("part_brand", parts[i].get("part_brand", ""))
            parts[i]["part_type"] = enrichment.get("part_type", parts[i].get("part_type", ""))
            parts[i]["compatible_make"] = enrichment.get("compatible_make", "")
            parts[i]["compatible_model"] = enrichment.get("compatible_model", "")
            parts[i]["year_from"] = enrichment.get("year_from")
            parts[i]["year_to"] = enrichment.get("year_to")
            parts[i]["condition"] = enrichment.get("condition", "New")

        return parts

    # ── Find parts needing enrichment ─────────
    def _find_unenriched(self, parts: list) -> list:
        return [
            (i, p) for i, p in enumerate(parts)
            if not self._is_valid(p)
        ]

    # ── Enrich all ────────────────────────────
    def enrich_all(self, parts: list) -> list:
        """
        Run enrichment on all parts in batches.
        Then do GPT_RECHECK_PASSES passes on remaining unknowns.
        """
        logger.info(f"GPT enrichment: {len(parts)} parts, batch size={self.batch_size}")

        # First pass — enrich everything
        batches = [
            parts[i : i + self.batch_size]
            for i in range(0, len(parts), self.batch_size)
        ]
        pbar = tqdm(batches, desc="GPT enrichment", unit="batch")
        for batch in pbar:
            self.enrich_batch(batch)

        valid = sum(1 for p in parts if self._is_valid(p))
        logger.info(f"After first pass: {valid}/{len(parts)} fully enriched")

        # Re-check passes for unknowns
        for pass_num in range(1, config.GPT_RECHECK_PASSES + 1):
            unknowns = self._find_unenriched(parts)
            if not unknowns:
                logger.info("All parts enriched — no re-check needed")
                break

            logger.info(f"Re-check pass {pass_num}: {len(unknowns)} unknowns")
            unknown_parts = [p for _, p in unknowns]
            batches = [
                unknown_parts[i : i + self.batch_size]
                for i in range(0, len(unknown_parts), self.batch_size)
            ]
            for batch in tqdm(batches, desc=f"Re-check pass {pass_num}", unit="batch"):
                self.enrich_batch(batch)

            # Copy enriched data back
            for idx, part in unknowns:
                parts[idx] = part

            valid = sum(1 for p in parts if self._is_valid(p))
            logger.info(f"After re-check pass {pass_num}: {valid}/{len(parts)} enriched")

        return parts


# ─────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────
def deduplicate(parts: list) -> list:
    """Remove duplicate parts by normalised (name + url)."""
    seen = {}
    unique = []

    for part in parts:
        key = part.get("product_url", "").strip().lower()
        if not key:
            key = part.get("part_name", "").strip().lower()
        if not key:
            continue

        if key in seen:
            # Keep the one with more data
            existing = seen[key]
            if (part.get("price_pkr", 0) > 0 and existing.get("price_pkr", 0) == 0):
                seen[key] = part
                # replace in unique list
                for i, u in enumerate(unique):
                    if u.get("product_url", "").strip().lower() == key or \
                       u.get("part_name", "").strip().lower() == key:
                        unique[i] = part
                        break
        else:
            seen[key] = part
            unique.append(part)

    logger.info(f"Deduplication: {len(parts)} → {len(unique)} unique parts")
    return unique


# ─────────────────────────────────────────────
# Save Functions
# ─────────────────────────────────────────────
def save_raw(parts: list):
    """Save raw scraped data to JSON and CSV."""
    with open(config.RAW_JSON, "w", encoding="utf-8") as f:
        json.dump(parts, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved: {config.RAW_JSON} ({len(parts)} records)")

    try:
        import pandas as pd
        df = pd.json_normalize(parts)
        df.to_csv(config.RAW_CSV, index=False, encoding="utf-8-sig")
        logger.info(f"Saved: {config.RAW_CSV}")
    except ImportError:
        logger.warning("pandas not installed — CSV export skipped")


def print_summary(parts: list, start_time: float, request_count: int):
    """Print scraping summary."""
    elapsed = time.time() - start_time
    total = len(parts)
    available = sum(1 for p in parts if p.get("available", True))
    enriched = sum(1 for p in parts if p.get("compatible_make") and
                   p["compatible_make"].lower() not in ("", "unknown", "n/a"))

    cats = defaultdict(int)
    for p in parts:
        cats[p.get("scraped_category", "Unknown")] += 1

    websites = defaultdict(int)
    for p in parts:
        websites[p.get("website", "Unknown")] += 1

    print(f"\n{'=' * 55}")
    print("  CAR PARTS SCRAPING COMPLETE")
    print(f"{'=' * 55}")
    print(f"  Total parts:      {total}")
    print(f"  Available:        {available}")
    print(f"  Unavailable:      {total - available}")
    print(f"  GPT enriched:     {enriched}")
    print(f"  HTTP requests:    {request_count}")
    print(f"  Time elapsed:     {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"\n  Sources:")
    for src, count in sorted(websites.items()):
        print(f"    {src}: {count}")
    print(f"\n  Top Categories:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1])[:15]:
        print(f"    {cat}: {count}")
    print(f"{'=' * 55}")
    print(f"\n  Output: {config.RAW_JSON}, {config.RAW_CSV}")
    print(f"  Next:   python dataset_builder.py --enrich")
    print()


# ─────────────────────────────────────────────
# CLI & Main
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Pakistan Car Parts Scraper")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Specific category slugs (e.g., --categories car-parts/spark-plugs car-filters/oil-filters)",
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip GPT enrichment step",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--fetch-details",
        action="store_true",
        help="Also visit each product's detail page for description (slower)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    print("\n" + "=" * 55)
    print("  Pakistan Car Parts Price Scraper")
    print("=" * 55 + "\n")

    client = HttpClient()
    all_parts = []

    # ── Filter categories if specified ──
    categories = config.AUTOSTORE_CATEGORIES
    if args.categories:
        categories = {
            slug: name
            for slug, name in config.AUTOSTORE_CATEGORIES.items()
            if any(c in slug for c in args.categories)
        }
        if not categories:
            print(f"No matching categories found for: {args.categories}")
            print(f"Available: {list(config.AUTOSTORE_CATEGORIES.keys())}")
            return

    # ── Scrape AutoStore.pk ──
    print("[1] Scraping AutoStore.pk ...")
    scraper = AutoStoreScraper(client)
    parts = scraper.run(
        categories=categories,
        resume=args.resume,
        fetch_details=args.fetch_details,
    )
    all_parts.extend(parts)
    print(f"    → {len(parts)} products from AutoStore.pk")

    if not all_parts:
        print("\nNo data scraped. Check your internet connection and try again.")
        return

    # ── Deduplicate ──
    print("\n[2] Deduplicating …")
    unique_parts = deduplicate(all_parts)

    # ── GPT Enrichment ──
    if not args.skip_enrichment:
        print(f"\n[3] GPT Enrichment ({len(unique_parts)} parts) …")
        print(f"    Model: {config.OPENAI_MODEL}")
        print(f"    Batch size: {config.GPT_BATCH_SIZE}")

        try:
            enricher = GPTEnricher()
            unique_parts = enricher.enrich_all(unique_parts)
        except ValueError as exc:
            logger.error(f"GPT enrichment skipped: {exc}")
            print(f"    ⚠ Skipped: {exc}")
    else:
        print("\n[3] GPT enrichment skipped (--skip-enrichment)")

    # ── Save ──
    print("\n[4] Saving raw data …")
    save_raw(unique_parts)

    # ── Summary ──
    print_summary(unique_parts, start_time, client.request_count)


if __name__ == "__main__":
    main()
