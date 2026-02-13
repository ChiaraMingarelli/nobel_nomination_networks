#!/usr/bin/env python3
"""
Nobel Prize Nomination Network Analysis — Streamlit App
========================================================
Standalone app for interactive network analysis of Nobel nomination data.
Scrapes the Nobel Prize Nomination Archive, caches edge data locally,
and renders three network views via network_analysis.py.

Usage:
    streamlit run network_app.py

Author: Chiara Mingarelli (Yale University)
This code was written with the assistance of Claude (Anthropic).
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from network_analysis import render_network_page


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://www.nobelprize.org/nomination/archive/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

DATA_DIR = Path(__file__).parent / "network_data"

# Map UI category names to archive prize codes (for list.php)
CATEGORY_TO_PRIZE = {
    "Physics": 1,
    "Chemistry": 2,
    "Physiology or Medicine": 3,
    "Literature": 4,
    "Peace": 5,
}

PRECOMPUTED_STATS_FILE = Path(__file__).parent / "precomputed_stats.json"


# ---------------------------------------------------------------------------
# Dataclasses (same structure as the main nominations app)
# ---------------------------------------------------------------------------

@dataclass
class NominationEntry:
    """A single nomination entry."""
    category: str
    year: int
    other_party: str  # Who nominated them (if nominee) or who they nominated (if nominator)
    nomination_id: str


@dataclass
class NominationResult:
    """Container for nomination search results."""
    person_id: str
    name: str
    url: str
    nominee_count: int
    nominator_count: int
    nominations_as_nominee: list  # List of NominationEntry
    nominations_as_nominator: list  # List of NominationEntry
    won_prize: bool
    prize_info: Optional[dict] = None


# ---------------------------------------------------------------------------
# Scraping — Archive Access
# ---------------------------------------------------------------------------

def get_person_details(person_id: str) -> Optional[NominationResult]:
    """
    Fetch detailed nomination information for a person from the Nobel archive.

    Parses the show_people.php page to extract:
    - Name, nomination counts
    - All nominations as nominee (category, year, nominator, nomination_id)
    - All nominations as nominator
    - Whether they won the prize
    """
    url = f"{BASE_URL}show_people.php?id={person_id}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text()

        # Extract nomination counts
        nominee_match = re.search(r"Nominee in (\d+) nomination", page_text)
        nominator_match = re.search(r"Nominator in (\d+) nomination", page_text)

        nominee_count = int(nominee_match.group(1)) if nominee_match else 0
        nominator_count = int(nominator_match.group(1)) if nominator_match else 0

        # Extract name from page — look for h2 with the person's name
        name = ""
        for h2 in soup.find_all("h2"):
            text = h2.get_text(strip=True)
            if text and not any(x in text.lower() for x in ["share", "archive", "nomination"]):
                name = text
                break

        # Fallback: try Firstname/Lastname pattern
        if not name or "organisation" in name.lower():
            firstname_match = re.search(r"Firstname:\s*(\S+)", page_text)
            lastname_match = re.search(r"Lastname/org:\s*(\S+)", page_text)
            if firstname_match and lastname_match:
                name = f"{firstname_match.group(1)} {lastname_match.group(1)}"

        # Parse nomination entries
        nominations_as_nominee = []
        nominations_as_nominator = []

        # Regex for nomination link text
        cat_pattern = (
            r"(Physics|Chemistry|Physiology or Medicine|Medicine"
            r"|Literature|Peace|Economic Sciences|Economics)"
        )

        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)

            if "show.php?id=" not in href:
                continue

            nom_id_match = re.search(r"id=(\d+)", href)
            nom_id = nom_id_match.group(1) if nom_id_match else ""

            # Pattern for nominee: "Physics 1910 by Wilhelm Ostwald"
            nominee_match = re.match(
                cat_pattern + r"\s+(\d{4})\s+by\s+(.+)", text
            )
            if nominee_match:
                nominations_as_nominee.append(NominationEntry(
                    category=nominee_match.group(1),
                    year=int(nominee_match.group(2)),
                    other_party=nominee_match.group(3).strip(),
                    nomination_id=nom_id,
                ))
                continue

            # Pattern for nominator: "Physics 1919 for Max Planck"
            nominator_match = re.match(
                cat_pattern + r"\s+(\d{4})\s+for\s+(.+)", text
            )
            if nominator_match:
                nominations_as_nominator.append(NominationEntry(
                    category=nominator_match.group(1),
                    year=int(nominator_match.group(2)),
                    other_party=nominator_match.group(3).strip(),
                    nomination_id=nom_id,
                ))

        # Check if they won
        won_prize = "Awarded the Nobel" in page_text
        prize_info = None
        if won_prize:
            prize_match = re.search(
                r"Awarded the Nobel Prize in (\w+(?:\s+\w+)*)\s+(\d{4})", page_text
            )
            if prize_match:
                prize_info = {
                    "category": prize_match.group(1),
                    "year": int(prize_match.group(2)),
                }
            else:
                peace_match = re.search(
                    r"Awarded the Nobel Peace Prize\s+(\d{4})", page_text
                )
                if peace_match:
                    prize_info = {"category": "Peace", "year": int(peace_match.group(1))}

        return NominationResult(
            person_id=person_id,
            name=name,
            url=url,
            nominee_count=nominee_count,
            nominator_count=nominator_count,
            nominations_as_nominee=nominations_as_nominee,
            nominations_as_nominator=nominations_as_nominator,
            won_prize=won_prize,
            prize_info=prize_info,
        )

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data Collection Pipeline
# ---------------------------------------------------------------------------

def collect_nominee_ids(category: str, year_from: int, year_to: int) -> dict:
    """
    For each year in range, fetch list.php?prize={code}&year={year}
    and extract all show_people.php?id=X links.

    Returns dict of {person_id: person_name} (deduplicated).
    """
    prize_code = CATEGORY_TO_PRIZE[category]
    results = {}

    progress = st.progress(0, text="Collecting nominee IDs...")
    total_years = year_to - year_from + 1

    for i, year in enumerate(range(year_from, year_to + 1)):
        url = f"{BASE_URL}list.php"
        params = {"prize": prize_code, "year": year}

        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    if "show_people.php?id=" in href:
                        pid_match = re.search(r"id=(\d+)", href)
                        if pid_match:
                            pid = pid_match.group(1)
                            name_text = link.get_text(strip=True)
                            if name_text and pid not in results:
                                results[pid] = name_text
        except Exception:
            pass

        progress.progress(
            (i + 1) / total_years,
            text=f"Scanning year {year}... ({len(results)} people found)",
        )
        time.sleep(0.05)

    progress.empty()
    return results


def collect_nomination_edges(person_ids: dict, category: str) -> pd.DataFrame:
    """
    For each person ID, call get_person_details() and extract
    nominations_as_nominee entries into rows.

    Returns DataFrame with columns:
        nomination_id, year, category, nominee_name, nominee_id, nominator_name
    """
    rows = []
    seen_edges = set()  # (nomination_id, nominee_id) — same nomination can cover multiple nominees

    progress = st.progress(0, text="Collecting nomination edges...")
    total = len(person_ids)

    for i, (pid, name) in enumerate(person_ids.items()):
        result = get_person_details(pid)
        if result and result.nominations_as_nominee:
            for entry in result.nominations_as_nominee:
                edge_key = (entry.nomination_id, pid)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                rows.append({
                    "nomination_id": entry.nomination_id,
                    "year": entry.year,
                    "category": entry.category,
                    "nominee_name": result.name or name,
                    "nominee_id": pid,
                    "nominator_name": entry.other_party,
                })

        progress.progress(
            (i + 1) / total,
            text=f"Fetching details {i + 1}/{total}: {name[:30]}... ({len(rows)} edges)",
        )
        time.sleep(0.15)

    progress.empty()
    return pd.DataFrame(rows)


@st.cache_data
def load_precomputed_stats() -> dict:
    """Load precomputed laureate statistics from JSON file (if present)."""
    if PRECOMPUTED_STATS_FILE.exists():
        try:
            with open(PRECOMPUTED_STATS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def enrich_with_country(df: pd.DataFrame, precomputed: dict) -> pd.DataFrame:
    """
    Add nominee_country, nominator_country, and nominee_prize_year columns.

    For nominees: look up Country and Year Won from precomputed_stats.json.
    For nominators: country set to "Unknown" (would require per-nominator API calls).
    """
    if df.empty:
        df["nominee_country"] = []
        df["nominator_country"] = []
        df["nominee_prize_year"] = []
        return df

    # Build lookups from precomputed stats
    country_by_id = {}
    country_by_name = {}
    prize_year_by_id = {}
    prize_year_by_name = {}
    for cat_key, entries in precomputed.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            eid = str(entry.get("ID", ""))
            name_key = entry.get("Name", "").lower().strip()
            if eid and entry.get("Country"):
                country_by_id[eid] = entry["Country"]
            if name_key and entry.get("Country"):
                country_by_name[name_key] = entry["Country"]
            if eid and entry.get("Year Won"):
                prize_year_by_id[eid] = int(entry["Year Won"])
            if name_key and entry.get("Year Won"):
                prize_year_by_name[name_key] = int(entry["Year Won"])

    def lookup_country(row):
        nominee_id = str(row.get("nominee_id", ""))
        if nominee_id in country_by_id:
            return country_by_id[nominee_id]
        name = row.get("nominee_name", "").lower().strip()
        if name in country_by_name:
            return country_by_name[name]
        for known_name, country in country_by_name.items():
            if name and known_name and (name in known_name or known_name in name):
                return country
        return "Unknown"

    def lookup_prize_year(row):
        nominee_id = str(row.get("nominee_id", ""))
        if nominee_id in prize_year_by_id:
            return prize_year_by_id[nominee_id]
        name = row.get("nominee_name", "").lower().strip()
        if name in prize_year_by_name:
            return prize_year_by_name[name]
        for known_name, year in prize_year_by_name.items():
            if name and known_name and (name in known_name or known_name in name):
                return year
        return None

    df["nominee_country"] = df.apply(lookup_country, axis=1)
    df["nominator_country"] = "Unknown"
    df["nominee_prize_year"] = df.apply(lookup_prize_year, axis=1)

    return df


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def cache_filename(category: str) -> Path:
    """Return the cache file path for a given category."""
    safe_name = category.lower().replace(" ", "_").replace("/", "_")
    return DATA_DIR / f"{safe_name}_edges.json"


@st.cache_data
def load_cached_edges(category: str) -> pd.DataFrame | None:
    """Load cached edge data if available."""
    path = cache_filename(category)
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except Exception:
            return None
    return None


def save_cached_edges(category: str, df: pd.DataFrame):
    """Save edge data to cache."""
    DATA_DIR.mkdir(exist_ok=True)
    path = cache_filename(category)
    with open(path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)


@st.cache_data
def load_all_cached_edges() -> pd.DataFrame:
    """Load all 5 category caches, concatenate, and deduplicate."""
    frames = []
    for category in CATEGORY_TO_PRIZE:
        cat_df = load_cached_edges(category)
        if cat_df is not None:
            frames.append(cat_df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # Deduplicate on (nomination_id, nominee_id)
    if "nomination_id" in combined.columns and "nominee_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["nomination_id", "nominee_id"])
    return combined


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Nobel Nomination Networks",
        page_icon="🔗",
        layout="wide",
    )

    st.title("Nobel Prize Nomination Networks")
    st.markdown(
        "Interactive network analysis of [Nobel Prize Nomination Archive]"
        "(https://www.nobelprize.org/nomination/archive/) data. "
        "Explore who nominated whom, find co-nomination clusters, "
        "and detect coordinated campaigns."
    )

    # Load precomputed stats for country enrichment (optional — works without it)
    precomputed = load_precomputed_stats()

    # Load combined cross-category data
    combined_df = load_all_cached_edges()

    # --- Sidebar ---
    st.sidebar.header("Data Selection")

    category = st.sidebar.selectbox(
        "Prize category",
        list(CATEGORY_TO_PRIZE.keys()),
    )

    year_range = st.sidebar.slider(
        "Year range",
        min_value=1901,
        max_value=1974,
        value=(1901, 1953),
    )

    # Check for cached data
    cached_df = load_cached_edges(category)
    has_cache = cached_df is not None

    if has_cache:
        st.sidebar.success(f"Cached data: {len(cached_df)} edges")

        # Filter cached data to selected year range
        df = cached_df[
            (cached_df["year"] >= year_range[0])
            & (cached_df["year"] <= year_range[1])
        ].copy()

        # Enrich with country data if columns are missing
        if "nominee_country" not in df.columns:
            df = enrich_with_country(df, precomputed)

        if st.sidebar.button("Rebuild data (re-scrape)"):
            _build_data(category, year_range[0], year_range[1], precomputed)
            st.rerun()

    else:
        st.sidebar.warning("No cached data for this category.")
        n_years = year_range[1] - year_range[0] + 1
        st.sidebar.caption(
            f"Building data will scrape ~{n_years} year pages "
            f"plus individual nominee pages. "
            f"Estimated time: {n_years * 0.5 + 50:.0f}s – {n_years * 1 + 200:.0f}s."
        )
        if st.sidebar.button("Build Network Data", type="primary"):
            _build_data(category, year_range[0], year_range[1], precomputed)
            st.rerun()
        else:
            st.info(
                "Select a category and year range, then click **Build Network Data** "
                "in the sidebar to start collecting nomination data from the archive."
            )
            return

    if df.empty:
        st.warning("No nomination data found for the selected filters.")
        return

    # --- Main content: render network page ---
    render_network_page(df, precomputed=precomputed, combined_df=combined_df)


def _build_data(category: str, year_from: int, year_to: int, precomputed: dict):
    """Run the full data collection pipeline."""
    with st.status(f"Building network data for {category}...", expanded=True) as status:
        st.write("Step 1/3: Collecting nominee IDs from archive...")
        person_ids = collect_nominee_ids(category, year_from, year_to)
        st.write(f"Found {len(person_ids)} unique people.")

        st.write("Step 2/3: Fetching nomination details for each person...")
        df = collect_nomination_edges(person_ids, category)
        st.write(f"Collected {len(df)} nomination edges.")

        st.write("Step 3/3: Enriching with country data...")
        df = enrich_with_country(df, precomputed)
        known = (
            (df["nominee_country"] != "Unknown").sum()
            if "nominee_country" in df.columns
            else 0
        )
        st.write(f"Country data: {known}/{len(df)} nominees matched.")

        # Save full dataset for future use
        save_cached_edges(category, df)
        status.update(label=f"Done! {len(df)} edges cached.", state="complete")


if __name__ == "__main__":
    main()
