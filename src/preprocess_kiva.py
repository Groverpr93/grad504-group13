"""Build the modeling table for the Kiva GNN from all supplied CSV files.

Join path
---------
    kiva_loans.id
        -> loan_theme_ids.id
        -> loan_themes_by_region.(Loan Theme ID, Partner ID)
        -> kiva_mpi_region_locations.(country, region) / (LocationName)
    kiva_loans.country
        -> kiva_country_profile_variables.country

The script streams the large loan table and writes CSVs, so preprocessing does
not require pandas or loading the 187 MB source file into memory. It also
writes ``join_report.json`` with match rates and unmatched-key examples.

Example:
    python preprocess_kiva.py --data-dir . --output-dir processed
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def id_key(value: str) -> str:
    """Normalize CSV numeric IDs such as ``247`` and ``247.0`` to one key."""
    value = clean(value)
    if value.endswith(".0"):
        value = value[:-2]
    return value


def read_dicts(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        yield from csv.DictReader(f)


def load_theme_ids(path: Path) -> Dict[str, Dict[str, str]]:
    result = {}
    for row in read_dicts(path):
        key = id_key(row.get("id", ""))
        if key and key not in result:
            result[key] = row
    return result


def load_theme_regions(path: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, str]], Dict[str, Dict[str, str]]]:
    """Index theme metadata by exact theme+partner and theme-only fallback."""
    by_pair: Dict[Tuple[str, str], Dict[str, str]] = {}
    by_theme: Dict[str, Dict[str, str]] = {}
    for row in read_dicts(path):
        theme = clean(row.get("Loan Theme ID", "")); partner = id_key(row.get("Partner ID", ""))
        if not theme:
            continue
        by_pair.setdefault((theme, partner), row)
        by_theme.setdefault(theme, row)
    return by_pair, by_theme


def load_country_profiles(path: Path) -> Dict[str, Dict[str, str]]:
    result = {}
    for row in read_dicts(path):
        key = clean(row.get("country", ""))
        if key:
            result.setdefault(key, row)
    return result


def load_mpi(path: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, str]], Dict[Tuple[str, str], Dict[str, str]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    by_country_region: Dict[Tuple[str, str], Dict[str, str]] = {}
    by_iso_region: Dict[Tuple[str, str], Dict[str, str]] = {}
    by_location: Dict[str, Dict[str, str]] = {}
    country_values: Dict[str, List[float]] = defaultdict(list)
    for row in read_dicts(path):
        country, region = clean(row.get("country", "")), clean(row.get("region", ""))
        iso = clean(row.get("ISO", ""))
        location = clean(row.get("LocationName", ""))
        if country and region:
            by_country_region.setdefault((country, region), row)
        if iso and region:
            by_iso_region.setdefault((iso, region), row)
        if location:
            by_location.setdefault(location, row)
        if country:
            try:
                country_values[country].append(float(row.get("MPI", "")))
            except ValueError:
                pass
    # Build an explicit country-average proxy for locations with no matching
    # subnational MPI row. This is preferable to silently choosing one region.
    by_country = {country: {"MPI": f"{sum(values) / len(values):.6f}", "region": ""}
                  for country, values in country_values.items() if values}
    return by_country_region, by_iso_region, by_location, by_country


def first_gender(value: str) -> str:
    value = clean(value)
    return value.split(",", 1)[0] if value else "unknown"


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in row.items()})
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    ap.add_argument("--output-dir", type=Path, default=Path("processed"))
    ap.add_argument("--max-loans", type=int, default=0,
                    help="Optional deterministic row limit; 0 processes all loans")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load the smaller reference tables into dictionaries for fast row joins.
    theme_ids = load_theme_ids(args.data_dir / "loan_theme_ids.csv")
    theme_pair, theme_fallback = load_theme_regions(args.data_dir / "loan_themes_by_region.csv")
    profiles = load_country_profiles(args.data_dir / "kiva_country_profile_variables.csv")
    mpi_pair, mpi_iso_region, mpi_location, mpi_country = load_mpi(args.data_dir / "kiva_mpi_region_locations.csv")

    output_fields = [
        "loan_id", "partner_id", "date", "posted_time", "country", "country_code", "loan_region",
        "activity", "sector", "use", "currency", "funded_amount", "loan_amount", "term_in_months",
        "lender_count", "repayment_interval", "borrower_gender", "theme_id", "theme_type",
        "field_partner_name", "theme_country", "theme_region", "theme_iso", "theme_amount",
        "theme_number", "rural_pct", "mpi", "mpi_region", "mpi_match_type", "latitude", "longitude",
        "country_profile_region", "population_density", "sex_ratio", "gdp_per_capita",
        "gdp_growth_rate", "agriculture_gva", "services_gva", "female_labor_participation",
        "internet_use", "urban_population",
    ]
    edge_fields = ["loan_id", "partner_id", "date", "country", "country_code", "region",
                   "gender", "sector", "activity", "theme_id", "mpi", "rural_pct"]
    node_fields = ["node_id", "node_type", "node_key"]

    join_counts = Counter(); examples: Dict[str, List[str]] = defaultdict(list)
    nodes: Dict[Tuple[str, str], str] = {}
    processed = 0
    edge_handle = (args.output_dir / "loan_partner_edges.csv").open("w", encoding="utf-8", newline="")
    edge_writer = csv.DictWriter(edge_handle, fieldnames=edge_fields)
    edge_writer.writeheader()

    def remember(kind: str, value: str) -> None:
        if value and value not in examples[kind] and len(examples[kind]) < 10:
            examples[kind].append(value)

    def add_node(kind: str, key: str) -> None:
        if not key:
            return
        nodes.setdefault((kind, key), f"{kind}:{key}")

    # Stream the large loans table so we do not hold every joined row in RAM.
    def loan_rows() -> Iterable[Dict[str, object]]:
        nonlocal processed
        for raw in read_dicts(args.data_dir / "kiva_loans.csv"):
            if args.max_loans and processed >= args.max_loans:
                break
            loan = id_key(raw.get("id", "")); partner = id_key(raw.get("partner_id", ""))
            country = raw.get("country", "")
            if not loan or not partner:
                join_counts["loans_skipped_missing_id"] += 1; continue
            # First connect the loan to its theme, then use theme/partner
            # metadata to add regional information.
            theme_link = theme_ids.get(loan)
            if theme_link:
                join_counts["loan_to_theme_id_matched"] += 1
            else:
                join_counts["loan_to_theme_id_unmatched"] += 1; remember("loan_theme_id", loan)
            theme_id = clean(theme_link.get("Loan Theme ID", "")) if theme_link else ""
            theme = theme_pair.get((theme_id, partner)) or theme_fallback.get(theme_id)
            if theme:
                join_counts["theme_to_region_matched"] += 1
            else:
                join_counts["theme_to_region_unmatched"] += 1; remember("theme_id", theme_id)
            # Country profile data is joined by normalized country name.
            profile = profiles.get(clean(country))
            if profile: join_counts["country_profile_matched"] += 1
            else: join_counts["country_profile_unmatched"] += 1; remember("country", country)
            theme_country, theme_region = (theme.get("country", ""), theme.get("region", "")) if theme else (country, raw.get("region", ""))
            # Try the loan's own country/region first. Theme regions can refer
            # to a different operational geography than the loan's region.
            mpi = mpi_pair.get((clean(country), clean(raw.get("region", ""))))
            mpi_method = "loan_country_region" if mpi else ""
            if not mpi and raw.get("country_code"):
                mpi = mpi_iso_region.get((clean(raw.get("country_code", "")), clean(raw.get("region", ""))))
                mpi_method = "loan_iso_region" if mpi else ""
            if not mpi:
                mpi = mpi_pair.get((clean(theme_country), clean(theme_region)))
                mpi_method = "theme_country_region" if mpi else ""
            if not mpi and theme:
                mpi = mpi_location.get(clean(theme.get("LocationName", "")))
                mpi_method = "theme_location" if mpi else ""
            if not mpi:
                # Country fallback is an explicit proxy, not a claim that the
                # country average equals the borrower's local MPI.
                mpi = mpi_country.get(clean(country))
                mpi_method = "country_proxy" if mpi else ""
            if mpi: join_counts["mpi_matched"] += 1
            else: join_counts["mpi_unmatched"] += 1; remember("mpi_region", theme_region)
            if mpi_method:
                join_counts[f"mpi_{mpi_method}"] += 1

            # Keep original loan fields and append the joined feature columns.
            row = {
                "loan_id": loan, "partner_id": partner, "date": raw.get("date", ""),
                "posted_time": raw.get("posted_time", ""), "country": country,
                "country_code": raw.get("country_code", ""), "loan_region": raw.get("region", ""),
                "activity": raw.get("activity", ""), "sector": raw.get("sector", ""),
                "use": raw.get("use", ""), "currency": raw.get("currency", ""),
                "funded_amount": raw.get("funded_amount", ""), "loan_amount": raw.get("loan_amount", ""),
                "term_in_months": raw.get("term_in_months", ""), "lender_count": raw.get("lender_count", ""),
                "repayment_interval": raw.get("repayment_interval", ""),
                "borrower_gender": first_gender(raw.get("borrower_genders", "")),
                "theme_id": theme_id, "theme_type": theme.get("Loan Theme Type", "") if theme else "",
                "field_partner_name": theme.get("Field Partner Name", "") if theme else "",
                "theme_country": theme_country, "theme_region": theme_region,
                "theme_iso": theme.get("ISO", "") if theme else "", "theme_amount": theme.get("amount", "") if theme else "",
                "theme_number": theme.get("number", "") if theme else "",
                "rural_pct": mpi.get("rural_pct", "") if mpi else theme.get("rural_pct", "") if theme else "",
                "mpi": mpi.get("MPI", "") if mpi else "", "mpi_region": mpi.get("region", "") if mpi else "",
                "mpi_match_type": mpi_method,
                "latitude": mpi.get("lat", "") if mpi else theme.get("lat", "") if theme else "",
                "longitude": mpi.get("lon", "") if mpi else theme.get("lon", "") if theme else "",
                "country_profile_region": profile.get("Region", "") if profile else "",
                "population_density": profile.get("Population density (per km2, 2017)", "") if profile else "",
                "sex_ratio": profile.get("Sex ratio (m per 100 f, 2017)", "") if profile else "",
                "gdp_per_capita": profile.get("GDP per capita (current US$)", "") if profile else "",
                "gdp_growth_rate": profile.get("GDP growth rate (annual %, const. 2005 prices)", "") if profile else "",
                "agriculture_gva": profile.get("Economy: Agriculture (% of GVA)", "") if profile else "",
                "services_gva": profile.get("Economy: Services and other activity (% of GVA)", "") if profile else "",
                "female_labor_participation": profile.get("Labour force participation (female/male pop. %)", "") if profile else "",
                "internet_use": profile.get("Individuals using the Internet (per 100 inhabitants)", "") if profile else "",
                "urban_population": profile.get("Urban population (% of total population)", "") if profile else "",
            }
            edge_writer.writerow({"loan_id": loan, "partner_id": partner, "date": row["date"],
                                  "country": country, "country_code": row["country_code"], "region": row["loan_region"],
                                  "gender": row["borrower_gender"], "sector": row["sector"], "activity": row["activity"],
                                  "theme_id": theme_id, "mpi": row["mpi"], "rural_pct": row["rural_pct"]})
            add_node("loan", loan); add_node("partner", partner); add_node("country", clean(country))
            add_node("region", clean(row["loan_region"])); add_node("sector", clean(row["sector"]))
            processed += 1
            if processed % 50000 == 0:
                print(f"  processed {processed:,} loans | "
                      f"theme={join_counts['loan_to_theme_id_matched']:,} | "
                      f"country={join_counts['country_profile_matched']:,} | "
                      f"mpi={join_counts['mpi_matched']:,}", flush=True)
            yield row

    loan_count = write_csv(args.output_dir / "loans_joined.csv", output_fields, loan_rows())
    edge_handle.close()
    write_csv(args.output_dir / "graph_nodes.csv", node_fields,
              ({"node_id": node_id, "node_type": kind, "node_key": key}
               for (kind, key), node_id in nodes.items()))
    denominator = max(1, loan_count)
    report = {"input_files": ["kiva_loans.csv", "loan_theme_ids.csv", "loan_themes_by_region.csv",
              "kiva_mpi_region_locations.csv", "kiva_country_profile_variables.csv"],
              "processed_loans": loan_count, "join_counts": dict(join_counts),
              "join_rates_percent": {
                  "loan_to_theme_id": round(join_counts["loan_to_theme_id_matched"] / denominator * 100, 2),
                  "theme_to_region": round(join_counts["theme_to_region_matched"] / denominator * 100, 2),
                  "country_profile": round(join_counts["country_profile_matched"] / denominator * 100, 2),
                  "mpi": round(join_counts["mpi_matched"] / denominator * 100, 2),
                  "mpi_exact_region": round((join_counts["mpi_loan_country_region"] +
                                               join_counts["mpi_loan_iso_region"] +
                                               join_counts["mpi_theme_country_region"] +
                                               join_counts["mpi_theme_location"]) / denominator * 100, 2),
                  "mpi_country_proxy": round(join_counts["mpi_country_proxy"] / denominator * 100, 2),
              },
              "unmatched_examples": dict(examples), "output_files": ["loans_joined.csv",
              "loan_partner_edges.csv", "graph_nodes.csv", "join_report.json"]}
    (args.output_dir / "join_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
