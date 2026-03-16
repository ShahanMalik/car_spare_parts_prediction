"""
API Test Suite for Car Parts Price Prediction
==============================================
Run:  python test_api.py [--url http://127.0.0.1:7860]
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

passed = 0
failed = 0


def test(name: str, method: str, path: str, base: str, body=None, checks=None):
    """Run a single API test."""
    global passed, failed
    url = f"{base}{path}"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        status = e.code
        result = json.loads(e.read().decode()) if e.fp else {}
    except Exception as e:
        print(f"  {RED}FAIL{RESET} {name} → {e}")
        failed += 1
        return

    ok = True
    msgs = []

    # Status check
    expected_status = (checks or {}).get("status", 200)
    if status != expected_status:
        ok = False
        msgs.append(f"status={status} (expected {expected_status})")

    # Field checks
    for key, expected in (checks or {}).items():
        if key == "status":
            continue
        if key == "has_keys":
            for k in expected:
                if k not in result:
                    ok = False
                    msgs.append(f"missing key '{k}'")
        elif key == "min_count":
            field, minimum = expected
            val = result.get(field, 0)
            if isinstance(val, list):
                val = len(val)
            if val < minimum:
                ok = False
                msgs.append(f"{field}={val} (expected >= {minimum})")
        elif key == "price_range":
            price = result.get("predicted_price_pkr", 0)
            lo, hi = expected
            if not (lo <= price <= hi):
                ok = False
                msgs.append(f"price={price} not in [{lo}, {hi}]")

    if ok:
        passed += 1
        extra = ""
        if "predicted_price_pkr" in result:
            extra = f" → Rs {result['predicted_price_pkr']:,.0f}"
        elif "count" in result:
            extra = f" → {result['count']} predictions"
        print(f"  {GREEN}PASS{RESET} {name}{extra}")
    else:
        failed += 1
        print(f"  {RED}FAIL{RESET} {name}: {'; '.join(msgs)}")
        print(f"        Response: {json.dumps(result, indent=2)[:300]}")


def main():
    parser = argparse.ArgumentParser(description="Test Car Parts Price Prediction API")
    parser.add_argument("--url", default="http://127.0.0.1:7860", help="Base URL")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print(f"\n{'='*60}")
    print(f"  Car Parts Price Prediction API — Test Suite")
    print(f"  Target: {base}")
    print(f"{'='*60}\n")

    # ── 1. Health Check ──
    print(f"{YELLOW}[Health Check]{RESET}")
    test("GET /", "GET", "/", base, checks={
        "has_keys": ["status", "model_loaded", "model_name", "r2_score"],
    })

    # ── 2. Single Predictions ──
    print(f"\n{YELLOW}[Single Predictions]{RESET}")
    test("Spark Plug (Toyota Corolla)", "POST", "/predict", base, body={
        "part_name": "Denso Iridium Spark Plug Toyota Corolla",
        "part_type": "Iridium Spark Plug",
        "part_brand": "Denso",
        "scraped_category": "Spark Plugs",
        "compatible_make": "Toyota",
        "compatible_model": "Corolla",
        "year_from": 2014,
        "year_to": 2020,
        "condition": "New",
    }, checks={"has_keys": ["predicted_price_pkr", "confidence_range_low", "confidence_range_high"]})

    test("Oil Filter (Honda Civic)", "POST", "/predict", base, body={
        "part_name": "Guard Oil Filter Honda Civic 2016-2020",
        "part_type": "Oil Filter",
        "part_brand": "Guard",
        "scraped_category": "Oil Filters",
        "compatible_make": "Honda",
        "compatible_model": "Civic",
        "year_from": 2016,
        "year_to": 2020,
        "condition": "New",
    }, checks={"has_keys": ["predicted_price_pkr"]})

    test("Side Mirror (Suzuki Alto)", "POST", "/predict", base, body={
        "part_name": "Suzuki Alto 2019 Side Mirror Chrome Right",
        "part_type": "Side Mirror",
        "part_brand": "Suzuki",
        "scraped_category": "Side Mirrors",
        "compatible_make": "Suzuki",
        "compatible_model": "Alto",
        "year_from": 2019,
        "year_to": 2024,
        "condition": "New",
    }, checks={"has_keys": ["predicted_price_pkr"]})

    test("Minimal input (just name)", "POST", "/predict", base, body={
        "part_name": "Generic Brake Pad",
    }, checks={"has_keys": ["predicted_price_pkr"]})

    # ── 3. Batch Predictions ──
    print(f"\n{YELLOW}[Batch Predictions]{RESET}")
    test("Batch of 3 parts", "POST", "/predict/batch", base, body={
        "parts": [
            {"part_name": "Toyota Corolla Headlight Left", "compatible_make": "Toyota"},
            {"part_name": "Honda City Air Filter 2020", "compatible_make": "Honda"},
            {"part_name": "Suzuki Cultus Brake Disc Front", "compatible_make": "Suzuki"},
        ]
    }, checks={"min_count": ("count", 3)})

    # ── 4. Metadata ──
    print(f"\n{YELLOW}[Metadata]{RESET}")
    test("GET /metadata", "GET", "/metadata", base, checks={
        "has_keys": ["model_name", "feature_count", "training_samples", "best_model_metrics"],
    })

    # ── 5. Categories ──
    print(f"\n{YELLOW}[Categories]{RESET}")
    test("GET /categories", "GET", "/categories", base, checks={
        "has_keys": ["categories", "makes", "models", "brands"],
        "min_count": ("categories", 10),
    })

    # ── 6. Error handling ──
    print(f"\n{YELLOW}[Error Handling]{RESET}")
    test("Empty part_name (422)", "POST", "/predict", base, body={
        "part_name": "",
    }, checks={"status": 422})

    test("Missing body (422)", "POST", "/predict", base, body={}, checks={"status": 422})

    # ── Summary ──
    total = passed + failed
    print(f"\n{'='*60}")
    color = GREEN if failed == 0 else RED
    print(f"  {color}Results: {passed}/{total} passed, {failed} failed{RESET}")
    print(f"{'='*60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
