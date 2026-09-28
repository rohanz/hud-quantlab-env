"""Regenerate src/quantlab_env/fixture.json from SEC EDGAR XBRL company facts.

Usage:
    SEC_USER_AGENT="Your Name, contact email" uv run python scripts/build_fixture.py

SEC asks every client to send a User-Agent with contact details, so the script
refuses to run without SEC_USER_AGENT. The environment itself never touches the
network; it only reads the JSON this script writes.

What is kept: annual (10-K / 10-K/A, roughly 12-month duration) facts for two
concepts, every filing that reported each period, for fiscal periods ending in
FIRST_YEAR..LAST_YEAR. Keeping every filing (not just the latest) is what makes
point-in-time queries possible: a later 10-K that restates a prior year adds a
second row for the same period with a later `filed` date.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import httpx

COMPANIES = {
    "AAPL": ("Apple Inc.", 320193),
    "MSFT": ("Microsoft Corp.", 789019),
    "NVDA": ("NVIDIA Corp.", 1045810),
    "KO": ("The Coca-Cola Co.", 21344),
    "JNJ": ("Johnson & Johnson", 200406),
}

# Ordered XBRL tags per concept; within each filing, the first tag that reports a period wins.
CONCEPTS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
}

FIRST_YEAR, LAST_YEAR = 2021, 2024
OUT = Path(__file__).resolve().parent.parent / "src" / "quantlab_env" / "fixture.json"


def fetch_companyfacts(cik: int, user_agent: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    resp = httpx.get(url, headers={"User-Agent": user_agent}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def annual_rows(facts: dict, tags: list[str]) -> list[dict]:
    """One row per (fiscal period, filing), from the first tag that filing used for the period.

    Precedence is per filing, not per period: when a company switches tags, the
    older filings' values under the fallback tag are still kept.
    """
    gaap = facts["facts"].get("us-gaap", {})
    # Keyed by (period, filing); companyfacts repeats a fact once per frame, so duplicates collapse here.
    by_filing: dict[tuple[str, str], dict] = {}
    for tag in tags:
        for row in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if row.get("form") not in ("10-K", "10-K/A") or "start" not in row:
                continue
            start, end = date.fromisoformat(row["start"]), date.fromisoformat(row["end"])
            if not 350 <= (end - start).days <= 380:
                continue
            if not FIRST_YEAR <= end.year <= LAST_YEAR:
                continue
            key = (row["end"], row["accn"])
            if key in by_filing and by_filing[key]["tag"] != tag:
                continue  # an earlier tag already owns this period in this filing
            by_filing[key] = {
                "period_start": row["start"],
                "period_end": row["end"],
                "value": row["val"],
                "form": row["form"],
                "filed": row["filed"],
                "accn": row["accn"],
                "tag": tag,
            }
    return sorted(by_filing.values(), key=lambda r: (r["period_end"], r["filed"], r["accn"]))


def main() -> None:
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        sys.exit("set SEC_USER_AGENT to 'Name email' (SEC fair-access policy)")
    fixture = {
        "source": "SEC EDGAR XBRL companyfacts API (data.sec.gov), public domain",
        "concepts": CONCEPTS,
        "companies": {},
    }
    for ticker, (name, cik) in COMPANIES.items():
        facts = fetch_companyfacts(cik, user_agent)
        fixture["companies"][ticker] = {
            "name": name,
            "cik": cik,
            "facts": {concept: annual_rows(facts, tags) for concept, tags in CONCEPTS.items()},
        }
        print(ticker, {c: len(r) for c, r in fixture["companies"][ticker]["facts"].items()})
    OUT.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
