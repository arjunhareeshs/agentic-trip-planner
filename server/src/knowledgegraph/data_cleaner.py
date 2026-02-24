"""
Data Cleaner Module — Stage 1 of the KG Pipeline
Loads, cleans, normalizes, and deduplicates all CSV data.
"""
import os
import re
import hashlib
import pandas as pd
from fuzzywuzzy import fuzz
try:
    from .config import (
        PRIORITY_FILES, BLOCKED_FILES, COUNTRY_ALIASES, CONTINENT_MAP,
        SYNTHETIC_NAME_PATTERNS, FUZZY_MATCH_THRESHOLD, SEASON_ALIASES,
        VALID_SEASONS, DEFAULT_EMOTION_MAP,
    )
except ImportError:
    from config import (
        PRIORITY_FILES, BLOCKED_FILES, COUNTRY_ALIASES, CONTINENT_MAP,
        SYNTHETIC_NAME_PATTERNS, FUZZY_MATCH_THRESHOLD, SEASON_ALIASES,
        VALID_SEASONS, DEFAULT_EMOTION_MAP,
    )


def load_csv_safely(filepath):
    """Load CSV with safety checks — blocks garbage files."""
    basename = os.path.basename(filepath)
    if basename in BLOCKED_FILES:
        print(f"  ❌ BLOCKED: {basename} (known garbage data)")
        return None

    if not os.path.exists(filepath):
        print(f"  ⚠️  NOT FOUND: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        print(f"  ✅ Loaded: {basename} ({len(df)} rows, {len(df.columns)} columns)")
        return df
    except Exception as e:
        print(f"  ❌ ERROR loading {basename}: {e}")
        return None


def normalize_name(name):
    """Standardize entity names for deduplication."""
    if pd.isna(name) or not isinstance(name, str):
        return ""
    name = name.strip()
    name = re.sub(r'^(the|a|an)\s+', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[^\w\s\'-]', '', name)  # Keep apostrophes and hyphens
    name = re.sub(r'\s+', ' ', name)
    return name.strip().title()


def normalize_country(raw_country):
    """Map all country variants to one canonical name."""
    if pd.isna(raw_country) or not isinstance(raw_country, str):
        return None
    key = raw_country.lower().strip()
    canonical = COUNTRY_ALIASES.get(key, raw_country.strip().title())
    return canonical


def normalize_season(raw_season):
    """Map season variants to canonical season names."""
    if pd.isna(raw_season) or not isinstance(raw_season, str):
        return None
    key = raw_season.lower().strip()
    # Direct match
    if key in SEASON_ALIASES:
        return SEASON_ALIASES[key]
    # Partial match
    for alias, canonical in SEASON_ALIASES.items():
        if alias in key:
            return canonical
    return raw_season.strip().title()


def normalize_place_type(raw_type):
    """Standardize place type names."""
    if pd.isna(raw_type) or not isinstance(raw_type, str):
        return "Other"
    t = raw_type.lower().strip()
    # Common mappings
    type_aliases = {
        "historical site": "Historical", "historic": "Historical",
        "heritage": "Heritage Site", "unesco": "Heritage Site",
        "natural": "Nature", "nature reserve": "Nature Reserve",
        "wildlife": "Wildlife Sanctuary",
        "religious site": "Religious", "pilgrimage site": "Pilgrimage",
        "amusement": "Amusement Park", "theme": "Theme Park",
        "shopping area": "Market", "bazaar": "Market",
    }
    for alias, canonical in type_aliases.items():
        if alias in t:
            return canonical
    return raw_type.strip().title()


def clean_cost(raw_value, default_currency="USD"):
    """Standardize cost values to (numeric, currency)."""
    if pd.isna(raw_value):
        return None, None
    text = str(raw_value).strip()
    if text == '' or text.lower() in ('nan', 'none', 'n/a', '-'):
        return None, None

    # Detect currency
    currency = default_currency
    if '₹' in text or 'INR' in text.upper():
        currency = "INR"
    elif '€' in text or 'EUR' in text.upper():
        currency = "EUR"
    elif '£' in text or 'GBP' in text.upper():
        currency = "GBP"
    elif '$' in text or 'USD' in text.upper():
        currency = "USD"

    # Extract numeric
    numeric = re.sub(r'[^\d.]', '', text)
    if not numeric:
        return None, None
    try:
        return round(float(numeric), 2), currency
    except ValueError:
        return None, None


def is_synthetic_name(name):
    """Check if a destination name is likely synthetic/fake."""
    if not isinstance(name, str):
        return True
    for pattern in SYNTHETIC_NAME_PATTERNS:
        if pattern.lower() in name.lower():
            return True
    return False


def create_entity_id(name, country, place_type):
    """Generate unique compound key — NEVER use name alone."""
    parts = [
        normalize_name(name).lower() if name else "unknown",
        normalize_country(country).lower() if country else "unknown",
        normalize_place_type(place_type).lower() if place_type else "other",
    ]
    key = "_".join(parts)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def find_fuzzy_duplicate(new_name, existing_names, threshold=None):
    """Fuzzy match to detect near-duplicates."""
    if threshold is None:
        threshold = FUZZY_MATCH_THRESHOLD
    normalized_new = normalize_name(new_name).lower()
    if not normalized_new:
        return None
    for existing in existing_names:
        normalized_existing = normalize_name(existing).lower()
        if fuzz.ratio(normalized_new, normalized_existing) >= threshold:
            return existing
    return None


# ============================================================
# CSV-SPECIFIC CLEANERS
# ============================================================

def clean_indian_places(df):
    """Clean 'Top Indian Places to Visit.csv'"""
    records = []
    for _, row in df.iterrows():
        name = normalize_name(row.get("Name", ""))
        if not name:
            continue

        place_type = normalize_place_type(row.get("Type", "Other"))
        city = normalize_name(row.get("City", ""))
        state = normalize_name(row.get("State", ""))
        zone = row.get("Zone", "")

        # Parse rating
        rating = row.get("Google review rating", None)
        try:
            rating = float(rating) if not pd.isna(rating) else None
        except (ValueError, TypeError):
            rating = None

        # Parse entrance fee
        fee_value, fee_currency = clean_cost(
            row.get("Entrance Fee in INR", None), default_currency="INR"
        )

        # Parse visit time
        visit_time = row.get("time needed to visit in hrs", None)
        try:
            visit_time = float(visit_time) if not pd.isna(visit_time) else None
        except (ValueError, TypeError):
            visit_time = None

        # Significance → useful for emotion mapping
        significance = str(row.get("Significance", "")).strip()

        # Best time to visit
        best_time = normalize_season(row.get("Best Time to visit", None))

        records.append({
            "name": name,
            "country": "India",
            "continent": "Asia",
            "state": state,
            "city": city,
            "zone": zone,
            "type": place_type,
            "rating": rating,
            "entrance_fee": fee_value,
            "entrance_fee_currency": fee_currency,
            "visit_time_hrs": visit_time,
            "significance": significance,
            "best_season": best_time,
            "airport_nearby": str(row.get("Airport with 50km Radius", "")).strip(),
            "weekly_off": str(row.get("Weekly Off", "")).strip(),
            "dslr_allowed": str(row.get("DSLR Allowed", "")).strip(),
            "review_count_lakhs": row.get("Number of google review in lakhs", None),
            "source_file": "Top Indian Places to Visit.csv",
            "data_quality": "verified",
            "is_synthetic": False,
        })
    return records


def clean_expanded_destinations(df):
    """Clean 'Expanded_Destinations.csv'
    Actual columns: DestinationID, Name, State, Type, Popularity, BestTimeToVisit
    Note: No Country column — these are all Indian destinations.
    """
    records = []
    for _, row in df.iterrows():
        name = normalize_name(row.get("Name", ""))
        if not name:
            continue

        country = "India"
        state = normalize_name(row.get("State", ""))
        place_type = normalize_place_type(row.get("Type", "Other"))
        is_synth = is_synthetic_name(name)

        # Parse popularity as rating (scale 0-10)
        popularity = row.get("Popularity", None)
        rating = None
        try:
            if not pd.isna(popularity):
                rating = round(float(popularity), 1)
        except (ValueError, TypeError):
            pass

        best_time = normalize_season(row.get("BestTimeToVisit", None))

        records.append({
            "name": name,
            "country": country,
            "continent": "Asia",
            "state": state,
            "city": "",
            "zone": "",
            "type": place_type,
            "rating": rating,
            "best_season": best_time,
            "source_file": "Expanded_Destinations.csv",
            "data_quality": "synthetic" if is_synth else "verified",
            "is_synthetic": is_synth,
        })
    return records


def clean_tourist_destinations(df):
    """Clean 'Tourist_Destinations.csv' — mostly synthetic, use cautiously."""
    records = []
    for _, row in df.iterrows():
        name = normalize_name(row.get("Destination Name", ""))
        if not name:
            continue

        country = normalize_country(row.get("Country", ""))
        if not country:
            continue

        is_synth = is_synthetic_name(name)

        records.append({
            "name": name,
            "country": country,
            "continent": row.get("Continent", CONTINENT_MAP.get(country, "Unknown")),
            "state": "",
            "city": "",
            "zone": "",
            "type": normalize_place_type(row.get("Type", "Other")),
            "rating": float(row["Avg Rating"]) if not pd.isna(row.get("Avg Rating")) else None,
            "avg_cost_usd": float(row["Avg Cost (USD/day)"]) if not pd.isna(row.get("Avg Cost (USD/day)")) else None,
            "best_season": normalize_season(row.get("Best Season", None)),
            "annual_visitors_m": row.get("Annual Visitors (M)", None),
            "unesco": str(row.get("UNESCO Site", "")).strip(),
            "source_file": "Tourist_Destinations.csv",
            "data_quality": "synthetic" if is_synth else "unverified",
            "is_synthetic": is_synth,
        })
    return records


def clean_reviews(df):
    """Clean review data for emotion extraction.
    Actual columns: ReviewID, DestinationID, UserID, Rating, ReviewText
    """
    records = []
    for _, row in df.iterrows():
        # Try multiple column name variants
        review_text = ""
        for col in ["ReviewText", "Review_Text", "Review Text", "review_text"]:
            if col in row.index:
                review_text = str(row[col]).strip()
                break

        if not review_text or review_text.lower() in ('nan', 'none', ''):
            continue

        rating = row.get("Rating", None)
        try:
            rating = float(rating) if not pd.isna(rating) else None
        except (ValueError, TypeError):
            rating = None

        # Try multiple column name variants for IDs
        dest_id = ""
        for col in ["DestinationID", "Destination_ID", "Destination ID"]:
            if col in row.index:
                dest_id = str(row[col]).strip()
                break

        review_id = ""
        for col in ["ReviewID", "Review_ID", "Review ID"]:
            if col in row.index:
                review_id = str(row[col]).strip()
                break

        user_id = ""
        for col in ["UserID", "User_ID", "User ID"]:
            if col in row.index:
                user_id = str(row[col]).strip()
                break

        records.append({
            "review_id": review_id,
            "destination_id": dest_id,
            "user_id": user_id,
            "rating": rating,
            "review_text": review_text,
        })
    return records


# ============================================================
# MASTER CLEANER — Runs all cleaning
# ============================================================

def clean_all_data():
    """
    Stage 1: Load and clean all CSV data.
    Returns: (destinations_list, reviews_list)
    """
    print("\n" + "=" * 60)
    print("📋 STAGE 1: DATA CLEANING")
    print("=" * 60)

    all_destinations = []
    all_reviews = []
    seen_ids = set()

    # 1. Indian Places (highest priority — real data)
    print("\n📍 Loading Indian Places...")
    df = load_csv_safely(PRIORITY_FILES["indian_places"])
    if df is not None:
        records = clean_indian_places(df)
        for rec in records:
            eid = create_entity_id(rec["name"], rec["country"], rec["type"])
            if eid not in seen_ids:
                rec["entity_id"] = eid
                all_destinations.append(rec)
                seen_ids.add(eid)
            else:
                print(f"    ⚠️  Duplicate skipped: {rec['name']}")
        print(f"    → {len(records)} cleaned, {len([r for r in records if create_entity_id(r['name'], r['country'], r['type']) in seen_ids])} unique")

    # 2. Expanded Destinations (second priority)
    print("\n📍 Loading Expanded Destinations...")
    df = load_csv_safely(PRIORITY_FILES["expanded_destinations"])
    if df is not None:
        records = clean_expanded_destinations(df)
        added = 0
        for rec in records:
            eid = create_entity_id(rec["name"], rec["country"], rec["type"])
            if eid not in seen_ids:
                rec["entity_id"] = eid
                all_destinations.append(rec)
                seen_ids.add(eid)
                added += 1
        print(f"    → {len(records)} cleaned, {added} new unique added")

    # 3. Tourist Destinations (use cautiously — synthetic)
    print("\n📍 Loading Tourist Destinations (synthetic — filtering)...")
    df = load_csv_safely(PRIORITY_FILES["tourist_destinations"])
    if df is not None:
        records = clean_tourist_destinations(df)
        added = 0
        synthetic_skipped = 0
        for rec in records:
            if rec["is_synthetic"]:
                synthetic_skipped += 1
                continue  # Skip synthetic names
            eid = create_entity_id(rec["name"], rec["country"], rec["type"])
            if eid not in seen_ids:
                rec["entity_id"] = eid
                all_destinations.append(rec)
                seen_ids.add(eid)
                added += 1
        print(f"    → {len(records)} cleaned, {synthetic_skipped} synthetic skipped, {added} real ones added")

    # 4. Reviews (for emotion extraction)
    print("\n📍 Loading Reviews...")
    df = load_csv_safely(PRIORITY_FILES["reviews"])
    if df is not None:
        all_reviews = clean_reviews(df)
        print(f"    → {len(all_reviews)} reviews cleaned")

    # Summary
    print("\n" + "-" * 40)
    print(f"📊 CLEANING COMPLETE:")
    print(f"   Total destinations: {len(all_destinations)}")
    print(f"   Verified: {len([d for d in all_destinations if d['data_quality'] == 'verified'])}")
    print(f"   Unverified: {len([d for d in all_destinations if d['data_quality'] == 'unverified'])}")
    print(f"   Total reviews: {len(all_reviews)}")
    print("-" * 40)

    return all_destinations, all_reviews


if __name__ == "__main__":
    destinations, reviews = clean_all_data()
    print(f"\nSample destination: {destinations[0] if destinations else 'None'}")
    print(f"Sample review: {reviews[0] if reviews else 'None'}")
