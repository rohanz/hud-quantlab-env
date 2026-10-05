"""scripts/build_fixture.py on synthetic EDGAR companyfacts JSON, and the checked-in fixture against it. Offline."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "src" / "quantlab_env" / "fixture.json"
_spec = importlib.util.spec_from_file_location("build_fixture", ROOT / "scripts" / "build_fixture.py")
build_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_fixture)

REVENUE_TAGS = build_fixture.CONCEPTS["revenue"]
PREFERRED, FALLBACK = REVENUE_TAGS[0], REVENUE_TAGS[1]


def fact(start: str, end: str, val: int, accn: str, filed: str, form: str = "10-K", frame: str | None = None) -> dict:
    row = {"start": start, "end": end, "val": val, "accn": accn, "fy": int(filed[:4]), "form": form, "filed": filed}
    if frame:
        row["frame"] = frame
    return row


def companyfacts(by_tag: dict[str, list[dict]]) -> dict:
    return {"facts": {"us-gaap": {tag: {"units": {"USD": rows}} for tag, rows in by_tag.items()}}}


def summary(rows: list[dict]) -> list[tuple[str, str, int, str]]:
    return [(r["period_end"], r["accn"], r["value"], r["tag"]) for r in rows]


FY21 = ("2021-01-01", "2021-12-31")
FY22 = ("2022-01-01", "2022-12-31")


def test_tag_transition_keeps_every_filing():
    # The 2022 10-K used the fallback tag; from 2023 the company switched to the preferred tag.
    # Precedence applies per filing, so the 2022 filing's value for FY2021 must survive.
    facts = companyfacts(
        {
            FALLBACK: [
                fact(*FY21, 100, "A-22", "2022-02-15"),
                fact(*FY21, 100, "A-23", "2023-02-15"),
            ],
            PREFERRED: [
                fact(*FY21, 100, "A-23", "2023-02-15"),
                fact(*FY22, 120, "A-23", "2023-02-15"),
            ],
        }
    )
    assert summary(build_fixture.annual_rows(facts, REVENUE_TAGS)) == [
        ("2021-12-31", "A-22", 100, FALLBACK),
        ("2021-12-31", "A-23", 100, PREFERRED),  # both tags in one filing: the preferred one wins
        ("2022-12-31", "A-23", 120, PREFERRED),
    ]


def test_restatement_adds_a_row_per_filing():
    facts = companyfacts(
        {
            PREFERRED: [
                fact(*FY21, 100, "A-22", "2022-02-15"),
                fact(*FY21, 90, "A-24", "2024-02-15"),
                fact(*FY21, 95, "A-24A", "2024-05-01", form="10-K/A"),
            ]
        }
    )
    assert summary(build_fixture.annual_rows(facts, REVENUE_TAGS)) == [
        ("2021-12-31", "A-22", 100, PREFERRED),
        ("2021-12-31", "A-24", 90, PREFERRED),
        ("2021-12-31", "A-24A", 95, PREFERRED),
    ]


def test_fiscal_year_focus_is_kept():
    facts = companyfacts({PREFERRED: [fact(*FY21, 100, "A-22", "2022-02-15")]})
    assert build_fixture.annual_rows(facts, REVENUE_TAGS)[0]["fy"] == 2022


def test_duplicate_facts_collapse_to_one_row_per_filing():
    # companyfacts repeats a fact once per frame it is assigned to.
    facts = companyfacts(
        {
            PREFERRED: [
                fact(*FY21, 100, "A-22", "2022-02-15", frame="CY2021"),
                fact(*FY21, 100, "A-22", "2022-02-15"),
            ]
        }
    )
    assert summary(build_fixture.annual_rows(facts, REVENUE_TAGS)) == [("2021-12-31", "A-22", 100, PREFERRED)]


def test_non_annual_and_out_of_range_facts_are_dropped():
    facts = companyfacts(
        {
            PREFERRED: [
                fact("2021-10-01", "2021-12-31", 25, "A-22", "2022-02-15"),  # a quarter inside a 10-K
                fact(*FY21, 100, "Q-22", "2022-05-01", form="10-Q"),
                fact("2019-01-01", "2019-12-31", 80, "A-20", "2020-02-15"),  # before FIRST_YEAR
                {"end": "2021-12-31", "val": 7, "accn": "A-22", "form": "10-K", "filed": "2022-02-15"},  # instant
            ]
        }
    )
    assert build_fixture.annual_rows(facts, REVENUE_TAGS) == []


def fixture_as_companyfacts(rows_by_concept: dict[str, list[dict]]) -> dict:
    by_tag: dict[str, list[dict]] = {}
    for rows in rows_by_concept.values():
        for r in rows:
            f = fact(r["period_start"], r["period_end"], r["value"], r["accn"], r["filed"], r["form"])
            f.pop("fy")
            if "fy" in r:  # fixtures built before the fy field have none
                f["fy"] = r["fy"]
            by_tag.setdefault(r["tag"], []).append(f)
    return companyfacts(by_tag)


def test_checked_in_fixture_is_a_fixed_point_of_the_builder():
    # Feeding the fixture's own rows back through the builder must reproduce them exactly:
    # no row in the fixture depends on the old per-period precedence.
    fixture = json.loads(FIXTURE.read_text())
    for company in fixture["companies"].values():
        facts = fixture_as_companyfacts(company["facts"])
        for concept, rows in company["facts"].items():
            tags = company.get("concepts", fixture["concepts"])[concept]
            assert build_fixture.annual_rows(facts, tags) == rows


def test_walmart_revenue_is_total_revenues_not_net_sales():
    # Walmart files net sales under the default first tag and total revenues under Revenues.
    tags = {**build_fixture.CONCEPTS, **build_fixture.TAG_OVERRIDES["WMT"]}["revenue"]
    facts = companyfacts(
        {
            PREFERRED: [fact("2024-02-01", "2025-01-31", 674_538, "W-25", "2025-03-14")],
            FALLBACK: [fact("2024-02-01", "2025-01-31", 680_985, "W-25", "2025-03-14")],
        }
    )
    assert summary(build_fixture.annual_rows(facts, tags)) == [("2025-01-31", "W-25", 680_985, FALLBACK)]
    wmt = json.loads(FIXTURE.read_text())["companies"]["WMT"]
    assert {r["tag"] for r in wmt["facts"]["revenue"]} == {"Revenues"}


# Filings that reported a period's net income but whose revenue row is missing. Empty since the
# 2026-10-06 rebuild with per-filing precedence, which restored NVIDIA's three missing rows.
KNOWN_MISSING_REVENUE: set[tuple[str, str, str]] = set()


def test_every_filing_with_net_income_also_has_revenue():
    fixture = json.loads(FIXTURE.read_text())
    missing = set()
    for ticker, company in fixture["companies"].items():
        revenue = {(r["period_end"], r["accn"]) for r in company["facts"]["revenue"]}
        for r in company["facts"]["net_income"]:
            if (r["period_end"], r["accn"]) not in revenue:
                missing.add((ticker, r["period_end"], r["accn"]))
    assert missing == KNOWN_MISSING_REVENUE
