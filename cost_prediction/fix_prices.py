"""
Price Fixer – Visit product pages to fill missing prices & detect stock status
===============================================================================
Reads the checkpoint file, visits each zero-price product's detail page,
extracts the real price and availability, then re-saves everything.

Usage:
    python fix_prices.py                 # fix prices in checkpoint, rebuild dataset
    python fix_prices.py --keep-zero     # keep items that truly have no price
    python fix_prices.py --dry-run       # preview what would change, don't write
"""

import argparse
import json
import logging
import os
import re
import time

from bs4 import BeautifulSoup
from tqdm import tqdm

import config

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
# Price extraction helpers
# ─────────────────────────────────────────────
def extract_price_from_text(text: str) -> float:
    """Extract a numeric PKR price from text like 'Rs 1,700' or '₨1700'."""
    if not text:
        return 0.0
    cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
    try:
        return round(float(cleaned), 2) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0


def extract_price_from_detail_page(html: str) -> dict:
    """
    Parse a product detail page and extract:
      - price (current/sale price)
      - original_price (before discount)
      - available (True/False)
      - description (short text)
    """
    soup = BeautifulSoup(html, "lxml")
    result = {
        "price": 0.0,
        "original_price": 0.0,
        "available": True,
        "description": "",
    }

    # ── Out-of-stock detection ─────────────
    # WooCommerce adds class or text for out-of-stock
    stock_el = soup.select_one(".stock.out-of-stock, .out-of-stock, .outofstock")
    if stock_el:
        result["available"] = False
    body_classes = soup.select_one("body")
    if body_classes and "outofstock" in (body_classes.get("class") or []):
        result["available"] = False
    # Also check for explicit "Out of stock" text near the add-to-cart area
    cart_area = soup.select_one(".summary, .product-summary, .single-product")
    if cart_area:
        cart_text = cart_area.get_text(" ", strip=True).lower()
        if "out of stock" in cart_text:
            result["available"] = False

    # ── Price extraction (multiple strategies) ──
    price = 0.0
    original = 0.0

    # Strategy 1: WooCommerce sale price structure (ins/del)
    ins_tag = soup.select_one("p.price ins .woocommerce-Price-amount, .summary ins .woocommerce-Price-amount")
    del_tag = soup.select_one("p.price del .woocommerce-Price-amount, .summary del .woocommerce-Price-amount")
    if ins_tag:
        price = extract_price_from_text(ins_tag.get_text(strip=True))
    if del_tag:
        original = extract_price_from_text(del_tag.get_text(strip=True))

    # Strategy 2: Single price (no sale)
    if price == 0.0:
        price_el = soup.select_one(
            "p.price .woocommerce-Price-amount, "
            ".summary .price .woocommerce-Price-amount, "
            ".product-page .price .amount, "
            ".woocommerce-Price-amount"
        )
        if price_el:
            price = extract_price_from_text(price_el.get_text(strip=True))

    # Strategy 3: Variable product – price range
    if price == 0.0:
        price_block = soup.select_one("p.price, .summary .price")
        if price_block:
            amounts = price_block.select(".woocommerce-Price-amount")
            if len(amounts) >= 2:
                p1 = extract_price_from_text(amounts[0].get_text(strip=True))
                p2 = extract_price_from_text(amounts[-1].get_text(strip=True))
                price = min(p1, p2) if p1 > 0 else p2
                original = max(p1, p2) if p1 != p2 else 0.0
            elif len(amounts) == 1:
                price = extract_price_from_text(amounts[0].get_text(strip=True))

    # Strategy 4: JSON-LD structured data
    if price == 0.0:
        for script in soup.select('script[type="application/ld+json"]'):
            try:
                ld = json.loads(script.string or "")
                offers = ld.get("offers") or ld.get("Offers") or {}
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                p = offers.get("price") or offers.get("lowPrice")
                if p:
                    price = float(p)
                    hp = offers.get("highPrice")
                    if hp:
                        original = float(hp)
            except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                pass

    # Strategy 5: Meta tag
    if price == 0.0:
        meta = soup.select_one('meta[property="product:price:amount"], meta[name="twitter:data1"]')
        if meta:
            price = extract_price_from_text(meta.get("content", ""))

    # Strategy 6: Regex on the full page text near the product area
    if price == 0.0:
        summary = soup.select_one(".summary, .product-info, .single-product")
        if summary:
            text = summary.get_text(" ")
            matches = re.findall(r"(?:Rs\.?\s*|₨\s*)([\d,]+(?:\.\d+)?)", text)
            for m in matches:
                p = extract_price_from_text(m)
                if p > 0:
                    price = p
                    break

    result["price"] = price
    result["original_price"] = original if original != price else 0.0

    # ── Description ───────────────────────
    desc_el = soup.select_one(
        ".woocommerce-product-details__short-description, "
        "#tab-description, "
        ".product-short-description"
    )
    if desc_el:
        result["description"] = desc_el.get_text(" ", strip=True)[:500]

    return result


# ─────────────────────────────────────────────
# HTTP Client (reuse from scraper)
# ─────────────────────────────────────────────
class SimpleHttpClient:
    """Lightweight HTTP client with rate limiting."""

    def __init__(self):
        import requests
        self.session = requests.Session()
        self.session.headers.update(config.HEADERS)
        self._last = 0

    def get(self, url: str, timeout: int = 20):
        import requests as req_lib
        import random

        elapsed = time.time() - self._last
        delay = random.uniform(0.8, 1.5)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last = time.time()

        try:
            resp = self.session.get(url, timeout=timeout)
            if resp.status_code == 404:
                return None  # product removed
            resp.raise_for_status()
            return resp
        except req_lib.RequestException as exc:
            logger.debug(f"Request failed: {url} — {exc}")
            return None


# ─────────────────────────────────────────────
# Main Fix Logic
# ─────────────────────────────────────────────
def fix_prices(dry_run: bool = False, keep_zero: bool = False):
    """Fix zero-price items by visiting their product pages."""

    # Load checkpoint
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "autostore.json")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run scraper.py first.")
        return

    with open(ckpt_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    data = payload.get("data", [])
    meta = payload.get("metadata", {})
    total = len(data)

    # Find zero-price items
    zero_indices = [i for i, d in enumerate(data) if not d.get("price_pkr") or d["price_pkr"] <= 0]
    print(f"\n{'='*55}")
    print(f"  Price Fixer")
    print(f"{'='*55}")
    print(f"  Total items:        {total}")
    print(f"  Zero-price items:   {len(zero_indices)}")
    print(f"  Items with price:   {total - len(zero_indices)}")

    if not zero_indices:
        print("  All items already have prices. Nothing to fix.")
        return

    if dry_run:
        print(f"\n  [DRY RUN] Would visit {len(zero_indices)} product pages")
        for i in zero_indices[:10]:
            print(f"    - {data[i]['part_name'][:60]}")
        if len(zero_indices) > 10:
            print(f"    ... and {len(zero_indices) - 10} more")
        return

    # Visit each zero-price product page
    print(f"\n  Visiting {len(zero_indices)} product pages to extract prices...\n")
    client = SimpleHttpClient()

    fixed = 0
    removed_404 = 0
    marked_oos = 0

    pbar = tqdm(zero_indices, desc="Fixing prices", unit="item")
    for idx in pbar:
        item = data[idx]
        url = item.get("product_url", "")
        if not url:
            continue

        resp = client.get(url)
        if resp is None:
            # 404 or connection error — mark as unavailable
            item["available"] = False
            item["_status"] = "404_or_error"
            removed_404 += 1
            continue

        detail = extract_price_from_detail_page(resp.text)

        # Update price
        if detail["price"] > 0:
            item["price_pkr"] = detail["price"]
            if detail["original_price"] > 0:
                item["price_original_pkr"] = detail["original_price"]
            fixed += 1

        # Update availability
        if not detail["available"]:
            item["available"] = False
            marked_oos += 1

        # Update description if empty
        if not item.get("description") and detail.get("description"):
            item["description"] = detail["description"]

        pbar.set_postfix(fixed=fixed, oos=marked_oos, gone=removed_404)

    # Summary
    still_zero = sum(1 for d in data if (not d.get("price_pkr") or d["price_pkr"] <= 0))

    print(f"\n{'='*55}")
    print(f"  RESULTS")
    print(f"{'='*55}")
    print(f"  Prices fixed:            {fixed}")
    print(f"  Marked out-of-stock:     {marked_oos}")
    print(f"  Removed (404/gone):      {removed_404}")
    print(f"  Still zero-price:        {still_zero}")

    # Filter out truly dead items
    if not keep_zero:
        before = len(data)
        data = [
            d for d in data
            if (d.get("price_pkr") and d["price_pkr"] > 0)  # has price
            or d.get("available", True)  # or still available (variable pricing)
        ]
        dropped = before - len(data)
        if dropped:
            print(f"  Dropped unavailable+no-price: {dropped}")
        # Also mark truly-zero + available as special
        for d in data:
            if (not d.get("price_pkr") or d["price_pkr"] <= 0) and d.get("available"):
                d["price_note"] = "Variable pricing - contact seller"

    print(f"  Final item count:        {len(data)}")

    # Save updated checkpoint
    payload["data"] = data
    payload["count"] = len(data)
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n  Checkpoint updated: {ckpt_path}")

    # Also save as raw JSON (what dataset_builder reads)
    from scraper import save_raw, GPTEnricher, deduplicate

    # Deduplicate
    unique = deduplicate(data)

    # Re-run GPT enrichment on items that still need it
    unenriched = [p for p in unique if not p.get("compatible_make") or
                  p["compatible_make"].lower() in ("", "unknown", "n/a", "none")]
    if unenriched:
        print(f"\n  Re-enriching {len(unenriched)} items with GPT...")
        try:
            enricher = GPTEnricher()
            enricher.enrich_all(unenriched)
        except Exception as exc:
            logger.warning(f"GPT enrichment error: {exc}")

    # Save raw
    save_raw(unique)
    print(f"  Raw data saved: {config.RAW_JSON}")

    # Build final dataset
    print(f"\n  Building final dataset...")
    from dataset_builder import DatasetBuilder
    builder = DatasetBuilder()
    builder.build(enrich=False)
    builder.save()

    print(f"\n  Done! Output: {config.FINAL_CSV}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix zero-price items in scraped data")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    parser.add_argument("--keep-zero", action="store_true", help="Keep unavailable items with zero price")
    args = parser.parse_args()

    fix_prices(dry_run=args.dry_run, keep_zero=args.keep_zero)
