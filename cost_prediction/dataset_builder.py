"""
Car Parts Dataset Builder
===========================
Cleans raw scraped data, standardises fields, removes duplicates,
and produces a final structured dataset.

Can also run GPT enrichment on parts still missing structured info.

Usage:
    python dataset_builder.py                    # build from car_parts_raw.json
    python dataset_builder.py --input my.json    # custom input file
    python dataset_builder.py --enrich           # re-run GPT enrichment on unknowns
"""

import argparse
import json
import logging
import re
from collections import defaultdict

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


class DatasetBuilder:
    """Processes raw scraped car-parts data into a clean, structured dataset."""

    def __init__(self, input_path: str = None):
        self.input_path = input_path or config.RAW_JSON
        self.parts = []

    # ── Load ──────────────────────────────────
    def load_data(self) -> list:
        """Load raw JSON from scraper output."""
        try:
            with open(self.input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                logger.info(f"Loaded {len(data)} records from {self.input_path}")
                return data
        except FileNotFoundError:
            logger.error(f"{self.input_path} not found. Run scraper.py first.")
        except json.JSONDecodeError:
            logger.error(f"{self.input_path} is not valid JSON.")
        return []

    # ── Clean ─────────────────────────────────
    def clean_part(self, part: dict) -> dict:
        """Normalise and clean a single car-part record."""
        return {
            "part_name":        self._clean_str(part.get("part_name", "")),
            "part_type":        self._clean_str(part.get("part_type", "")),
            "part_brand":       self._clean_str(part.get("part_brand", "")),
            "scraped_category": self._clean_str(part.get("scraped_category", "")),
            "compatible_make":  self._clean_str(part.get("compatible_make", "")),
            "compatible_model": self._clean_str(part.get("compatible_model", "")),
            "year_from":        self._clean_year(part.get("year_from")),
            "year_to":          self._clean_year(part.get("year_to")),
            "price_pkr":        self._clean_price(part.get("price_pkr", 0)),
            "price_original_pkr": self._clean_price(part.get("price_original_pkr", 0)),
            "condition":        self._clean_str(part.get("condition", "New")) or "New",
            "available":        bool(part.get("available", True)),
            "website":          part.get("website", ""),
            "product_url":      part.get("product_url", ""),
            "description":      self._clean_str(part.get("description", ""))[:500],
            "alternatives":     [],
        }

    def _clean_str(self, val) -> str:
        if not val:
            return ""
        text = str(val).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _clean_price(self, val) -> float:
        try:
            cleaned = str(val).replace(",", "").strip()
            return round(float(cleaned), 2)
        except (ValueError, TypeError):
            return 0.0

    def _clean_year(self, val) -> int | None:
        if val is None:
            return None
        try:
            y = int(val)
            return y if 1980 <= y <= 2030 else None
        except (ValueError, TypeError):
            return None

    # ── Deduplicate ───────────────────────────
    def remove_duplicates(self, parts: list) -> list:
        """Remove duplicates by product URL, keeping the richer record."""
        seen = {}
        unique = []

        for part in parts:
            key = part["product_url"].strip().lower()
            if not key:
                key = part["part_name"].lower().strip()
            if not key:
                continue

            if key in seen:
                existing = seen[key]
                # Keep whichever has more filled fields
                new_score = sum(1 for v in part.values() if v)
                old_score = sum(1 for v in existing.values() if v)
                if new_score > old_score:
                    # Replace
                    for i, u in enumerate(unique):
                        uid = u["product_url"].strip().lower() or u["part_name"].lower().strip()
                        if uid == key:
                            unique[i] = part
                            seen[key] = part
                            break
            else:
                seen[key] = part
                unique.append(part)

        logger.info(f"Deduplication: {len(parts)} → {len(unique)}")
        return unique

    # ── Alternatives ──────────────────────────
    def build_alternatives(self, parts: list) -> list:
        """
        Auto-link alternatives: parts with the same part_type + compatible_make
        that could substitute each other (different brands, same function).
        """
        group_map = defaultdict(list)
        for part in parts:
            ptype = part.get("part_type", "").strip().lower()
            make = part.get("compatible_make", "").strip().lower()
            model = part.get("compatible_model", "").strip().lower()
            if ptype and make and make != "universal":
                group_key = f"{ptype}|{make}|{model}"
                group_map[group_key].append(part)

        for part in parts:
            ptype = part.get("part_type", "").strip().lower()
            make = part.get("compatible_make", "").strip().lower()
            model = part.get("compatible_model", "").strip().lower()
            if not ptype or not make or make == "universal":
                continue
            group_key = f"{ptype}|{make}|{model}"
            alts = [
                {
                    "part_name": alt["part_name"],
                    "part_brand": alt.get("part_brand", ""),
                    "price_pkr": alt.get("price_pkr", 0),
                    "product_url": alt.get("product_url", ""),
                }
                for alt in group_map[group_key]
                if alt["product_url"] != part["product_url"]
            ]
            part["alternatives"] = alts[:10]  # cap at 10

        alt_count = sum(1 for p in parts if p["alternatives"])
        logger.info(f"Alternatives set for {alt_count} parts")
        return parts

    # ── Validation ────────────────────────────
    def _needs_enrichment(self, part: dict) -> bool:
        make = (part.get("compatible_make") or "").strip().lower()
        ptype = (part.get("part_type") or "").strip().lower()
        if not make or make in ("unknown", "n/a", "none"):
            return True
        if not ptype or ptype in ("unknown", "n/a", "none"):
            return True
        return False

    # ── GPT Re-enrichment ─────────────────────
    def enrich_unknowns(self, parts: list) -> list:
        """Re-enrich parts missing structured info."""
        unknowns = [
            (i, p) for i, p in enumerate(parts)
            if self._needs_enrichment(p)
        ]

        if not unknowns:
            logger.info("All parts already have structured info — nothing to enrich")
            return parts

        logger.info(f"Re-enriching {len(unknowns)} parts with missing data …")

        try:
            from scraper import GPTEnricher
            enricher = GPTEnricher()

            unknown_parts = [p for _, p in unknowns]
            batches = [
                unknown_parts[i : i + config.GPT_BATCH_SIZE]
                for i in range(0, len(unknown_parts), config.GPT_BATCH_SIZE)
            ]

            from tqdm import tqdm
            for batch in tqdm(batches, desc="Re-enriching", unit="batch"):
                enricher.enrich_batch(batch)

            # Copy back
            for (idx, _), enriched in zip(unknowns, unknown_parts):
                parts[idx] = enriched

            valid = sum(1 for p in parts if not self._needs_enrichment(p))
            logger.info(f"After re-enrichment: {valid}/{len(parts)} have full info")

        except Exception as exc:
            logger.error(f"GPT re-enrichment failed: {exc}")

        return parts

    # ── Build Pipeline ────────────────────────
    def build(self, enrich: bool = False) -> list:
        """Full pipeline: load → clean → dedupe → enrich → alternatives."""
        print("\n" + "=" * 55)
        print("  Car Parts Dataset Builder")
        print("=" * 55 + "\n")

        # Load
        raw = self.load_data()
        if not raw:
            print("  No data found. Run scraper.py first.\n")
            return []
        print(f"  Raw records:         {len(raw)}")

        # Clean
        cleaned = [self.clean_part(p) for p in raw if p.get("part_name")]
        print(f"  After cleaning:      {len(cleaned)}")

        # Drop unavailable items with no price
        before_drop = len(cleaned)
        cleaned = [
            p for p in cleaned
            if p["price_pkr"] > 0 or p.get("available", True)
        ]
        dropped = before_drop - len(cleaned)
        if dropped:
            print(f"  Dropped (no price + unavailable): {dropped}")

        # Deduplicate
        unique = self.remove_duplicates(cleaned)
        print(f"  After deduplication: {len(unique)}")

        # Optional GPT re-enrichment
        if enrich:
            unique = self.enrich_unknowns(unique)

        # Build alternatives
        final = self.build_alternatives(unique)

        self.parts = final
        return final

    # ── Save ──────────────────────────────────
    def save(self):
        """Save final dataset to JSON and CSV."""
        if not self.parts:
            logger.warning("No data to save")
            return

        # JSON
        with open(config.FINAL_JSON, "w", encoding="utf-8") as f:
            json.dump(self.parts, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved: {config.FINAL_JSON} ({len(self.parts)} records)")

        # CSV
        try:
            import pandas as pd
            df = pd.json_normalize(self.parts)
            # Flatten alternatives to a count for CSV
            if "alternatives" in df.columns:
                df["alternatives_count"] = df["alternatives"].apply(
                    lambda x: len(x) if isinstance(x, list) else 0
                )
                df.drop(columns=["alternatives"], inplace=True)
            df.to_csv(config.FINAL_CSV, index=False, encoding="utf-8-sig")
            logger.info(f"Saved: {config.FINAL_CSV}")
        except ImportError:
            logger.warning("pandas not installed — CSV export skipped")

        # Stats
        self._print_stats()

    def _print_stats(self):
        """Print dataset statistics."""
        total = len(self.parts)
        available = sum(1 for p in self.parts if p["available"])
        has_price = sum(1 for p in self.parts if p.get("price_pkr", 0) > 0)
        has_make = sum(
            1 for p in self.parts
            if p.get("compatible_make") and
               p["compatible_make"].lower() not in ("unknown", "n/a", "none", "")
        )
        has_model = sum(
            1 for p in self.parts
            if p.get("compatible_model") and
               p["compatible_model"].lower() not in ("unknown", "n/a", "none", "", "universal")
        )
        has_brand = sum(
            1 for p in self.parts
            if p.get("part_brand") and
               p["part_brand"].lower() not in ("unknown", "n/a", "none", "")
        )
        has_alts = sum(1 for p in self.parts if p.get("alternatives"))
        universal = sum(
            1 for p in self.parts
            if (p.get("compatible_make") or "").lower() == "universal"
        )

        # Category breakdown
        categories = defaultdict(int)
        for p in self.parts:
            categories[p.get("scraped_category", "Unknown")] += 1

        # Website breakdown
        websites = defaultdict(int)
        for p in self.parts:
            websites[p.get("website", "Unknown")] += 1

        # Car make breakdown
        makes = defaultdict(int)
        for p in self.parts:
            m = p.get("compatible_make", "Unknown")
            if m:
                makes[m] += 1

        print(f"\n{'=' * 55}")
        print("  DATASET STATISTICS")
        print(f"{'=' * 55}")
        print(f"  Total car parts:          {total}")
        print(f"  Available (in stock):     {available}")
        print(f"  Unavailable:              {total - available}")
        print(f"  With price:               {has_price}")
        print(f"  With car make identified: {has_make}")
        print(f"  With car model:           {has_model}")
        print(f"  Universal (fits any car): {universal}")
        print(f"  With part brand:          {has_brand}")
        print(f"  With alternatives:        {has_alts}")

        print(f"\n  Sources:")
        for src, count in sorted(websites.items()):
            print(f"    {src}: {count}")

        print(f"\n  Top Categories:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:15]:
            print(f"    {cat}: {count}")

        print(f"\n  Top Car Makes:")
        for make, count in sorted(makes.items(), key=lambda x: -x[1])[:10]:
            print(f"    {make}: {count}")

        print(f"{'=' * 55}")
        print(f"\n  Output: {config.FINAL_JSON}, {config.FINAL_CSV}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Car Parts Dataset Builder")
    parser.add_argument(
        "--input", default=None,
        help=f"Input JSON file (default: {config.RAW_JSON})",
    )
    parser.add_argument(
        "--enrich", action="store_true",
        help="Re-run GPT enrichment on parts with missing info",
    )
    args = parser.parse_args()

    builder = DatasetBuilder(input_path=args.input)
    builder.build(enrich=args.enrich)
    builder.save()


if __name__ == "__main__":
    main()
