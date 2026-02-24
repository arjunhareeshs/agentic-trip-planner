"""
Smart CSV Analyzer — LLM-powered universal CSV ingestion.
Sends sample rows to the LLM to auto-detect column mapping,
then cleans data using the mapped columns.

Usage:
    python main.py --smart-add "any_tourism_data.csv"
"""
import json
import pandas as pd
import re

try:
    from .entity_extractor import call_ollama_llm, parse_json_response
except ImportError:
    from entity_extractor import call_ollama_llm, parse_json_response
try:
    from .data_cleaner import normalize_name, load_csv_safely, create_entity_id
except ImportError:
    from data_cleaner import normalize_name, load_csv_safely, create_entity_id


# ============================================================
# LLM COLUMN MAPPING
# ============================================================

COLUMN_MAP_SYSTEM_PROMPT = """You are a data schema analyst for a tourism knowledge graph.
Given sample rows from a CSV file, map each column to a standard field.

STEP 1 — IDENTIFY THE PRIMARY ENTITY:
Every CSV describes ONE main thing per row (a city, a destination, a place, an attraction).
Find which column holds this primary entity and map it to "name".
Examples: "City" → name, "Place" → name, "Destination Name" → name, "Attraction" → name

STEP 2 — MAP REMAINING COLUMNS:
- city        : A DIFFERENT column showing which city the entity is IN (only if separate from name)
- state       : The state/province/region
- country     : The country
- type        : Category/type (e.g., Beach, Temple, Fort, City)
- rating      : Numerical rating/score
- season      : Best time/season to visit
- description : Text description
- cost        : Entry fee or cost
- skip        : Column is irrelevant

RULES:
1. Output ONLY valid JSON — no explanations, no markdown.
2. Map EVERY column to exactly one standard field.
3. "name" is MANDATORY — the primary entity column MUST be mapped to "name".
4. If the CSV is about cities, the city column IS the name (map to "name", not "city").
5. Only map a column to "city" if it's DIFFERENT from the name column."""


def detect_column_mapping(df):
    """
    Use LLM to auto-detect what each CSV column represents.
    Sends the first 5 rows as a sample.
    Returns: dict mapping original_column → standard_field
    """
    print("\n  🤖 Asking LLM to analyze CSV structure...")

    # Build sample for LLM
    sample = df.head(5).to_string(index=False)
    columns_list = list(df.columns)

    prompt = f"""Here is a CSV file with these columns:
{json.dumps(columns_list)}

Sample data (first 5 rows):
{sample}

Map each column name to the correct standard field.
Output a JSON object where keys are the original column names and values are the standard field names.

Example output format:
{{"Place Name": "name", "Location": "city", "Stars": "rating", "ID": "skip"}}

Now map these columns:
{json.dumps(columns_list)}"""

    response = call_ollama_llm(prompt, system_prompt=COLUMN_MAP_SYSTEM_PROMPT)
    mapping = parse_json_response(response)

    if not mapping:
        print("  ❌ LLM failed to produce column mapping.")
        return None

    # Fix LLM typos: match keys back to actual column names
    actual_cols = list(df.columns)
    fixed_mapping = {}
    for llm_key, field in mapping.items():
        matched = False
        for actual_col in actual_cols:
            if actual_col.lower().strip() == llm_key.lower().strip():
                fixed_mapping[actual_col] = field
                matched = True
                break
        if not matched:
            # Fuzzy match: LLM returned "Ratinggs" but actual is "Ratings"
            for actual_col in actual_cols:
                if actual_col.lower().strip()[:4] == llm_key.lower().strip()[:4]:
                    fixed_mapping[actual_col] = field
                    matched = True
                    break
        if not matched:
            fixed_mapping[llm_key] = field
    mapping = fixed_mapping

    # Validate: must have at least "name"
    if "name" not in mapping.values():
        print(f"  ⚠️  LLM mapping missing 'name' field: {mapping}")
        # Try to find the most likely name column by keyword
        for col, field in mapping.items():
            if any(kw in col.lower() for kw in ["name", "place", "destination", "spot", "attraction"]):
                mapping[col] = "name"
                break

    # Fallback: if still no "name" but "city" exists and there's no separate place column,
    # then the city IS the main entity (e.g., City.csv lists cities as destinations)
    if "name" not in mapping.values() and "city" in mapping.values():
        # Find the column mapped to "city" and promote it to "name"
        for col, field in mapping.items():
            if field == "city":
                mapping[col] = "name"
                print(f"  🔄 Promoted '{col}' from 'city' → 'name' (city-as-destination CSV)")
                break

    if "name" not in mapping.values():
        print("  ❌ Could not identify a 'name' column. Aborting.")
        return None

    return mapping


# ============================================================
# SMART CLEANING
# ============================================================

def smart_clean_csv(df, column_mapping, city_filter=None):
    """
    Clean any CSV using the LLM-detected column mapping.
    Renames columns to standard names, then applies deterministic cleaning.
    """
    # Build reverse map: standard_field → original_column
    field_to_col = {}
    for col, field in column_mapping.items():
        if field != "skip":
            field_to_col[field] = col

    records = []
    for _, row in df.iterrows():
        # Extract fields using mapping
        name_raw = str(row.get(field_to_col.get("name", ""), "")).strip()
        if not name_raw or name_raw.lower() in ("nan", "none", ""):
            continue

        # Clean numbering from name (e.g., "1. Marina Beach" → "Marina Beach")
        name_raw = re.sub(r'^\d+\.\s*', '', name_raw)
        name = normalize_name(name_raw)
        if not name:
            continue

        # City
        city = ""
        if "city" in field_to_col:
            city = normalize_name(str(row.get(field_to_col["city"], "")).strip())

        if city_filter and city and city.lower() != city_filter.lower():
            continue

        # Country
        country = "India"
        if "country" in field_to_col:
            raw_country = str(row.get(field_to_col["country"], "")).strip()
            if raw_country and raw_country.lower() not in ("nan", "none", ""):
                country = normalize_name(raw_country)

        # State
        state = ""
        if "state" in field_to_col:
            state = normalize_name(str(row.get(field_to_col["state"], "")).strip())

        # Type
        place_type = "Tourist Attraction"
        if "type" in field_to_col:
            raw_type = str(row.get(field_to_col["type"], "")).strip()
            if raw_type and raw_type.lower() not in ("nan", "none", ""):
                place_type = normalize_name(raw_type)
        else:
            # Infer from description if available
            desc = ""
            if "description" in field_to_col:
                desc = str(row.get(field_to_col["description"], "")).strip()
            place_type = _infer_type(name, desc)

        # Rating
        rating = None
        if "rating" in field_to_col:
            try:
                val = row.get(field_to_col["rating"], None)
                if not pd.isna(val):
                    rating = float(val)
            except (ValueError, TypeError):
                rating = None

        # Description
        desc = ""
        if "description" in field_to_col:
            desc = str(row.get(field_to_col["description"], "")).strip()
            if desc.lower() in ("nan", "none"):
                desc = ""

        # Continent lookup
        continent = "Asia" if country == "India" else ""

        records.append({
            "name": name,
            "country": country,
            "continent": continent,
            "state": state,
            "city": city,
            "zone": "",
            "type": place_type,
            "rating": rating,
            "description": desc[:500] if desc else "",
            "source_file": "smart_import",
            "data_quality": "verified",
            "is_synthetic": False,
            "entity_id": create_entity_id(name, country, place_type),
        })

    return records


def _infer_type(name, description):
    """Quick keyword-based type inference."""
    text = f"{name} {description}".lower()
    keywords = {
        "Beach": ["beach", "shore", "coast"],
        "Temple": ["temple", "shrine", "mandir", "kovil"],
        "Fort": ["fort", "fortress", "citadel"],
        "Museum": ["museum", "gallery"],
        "Waterfall": ["waterfall", "falls", "cascade"],
        "Lake": ["lake", "pond", "reservoir"],
        "Park": ["park", "garden", "sanctuary"],
        "Historical": ["historical", "heritage", "ancient", "ruins", "tomb", "monument"],
        "Religious": ["church", "mosque", "cathedral", "gurudwara"],
        "Adventure": ["trek", "rafting", "diving", "paragliding"],
        "Palace": ["palace", "haveli", "mahal"],
        "Market": ["market", "shopping", "bazaar"],
    }
    for ptype, kws in keywords.items():
        if any(kw in text for kw in kws):
            return ptype
    return "Tourist Attraction"


# ============================================================
# MAIN SMART PIPELINE
# ============================================================

def run_smart_pipeline(csv_path, city_filter=None):
    """
    LLM-powered universal CSV ingestion pipeline.
    1. Load CSV
    2. LLM detects column mapping (1 API call)
    3. Clean data using mapped columns
    4. Hand off to incremental pipeline for dedup + extract + merge
    """
    print("\n" + "=" * 60)
    print("🧠 SMART CSV ANALYZER (LLM-Powered)")
    print("=" * 60)
    print(f"  📂 File: {csv_path}")

    # Step 1: Load CSV
    df = load_csv_safely(csv_path)
    if df is None:
        print("  ❌ Failed to load CSV.")
        return None

    print(f"  📊 Rows: {len(df)}, Columns: {list(df.columns)}")

    # Step 2: LLM column detection
    column_mapping = detect_column_mapping(df)
    if not column_mapping:
        return None

    print(f"\n  ✅ LLM Column Mapping:")
    for col, field in column_mapping.items():
        emoji = "✅" if field != "skip" else "⏭️"
        print(f"     {emoji} '{col}' → {field}")

    # Step 3: Clean using mapping
    print(f"\n  🧹 Cleaning data with mapped columns...")
    records = smart_clean_csv(df, column_mapping, city_filter=city_filter)
    print(f"  📊 Cleaned records: {len(records)}")

    if not records:
        print("  ⚠️  No records after cleaning.")
        return None

    # Step 4: Hand off to incremental pipeline
    #   (dedup → extract → merge → save)
    try:
        from .incremental_loader import find_new_destinations, merge_into_graph
    except ImportError:
        from incremental_loader import find_new_destinations, merge_into_graph
    try:
        from .graph_store import load_graph_state, build_state_from_existing_output, save_graph_state
    except ImportError:
        from graph_store import load_graph_state, build_state_from_existing_output, save_graph_state
    try:
        from .entity_extractor import extract_all_entities
    except ImportError:
        from entity_extractor import extract_all_entities
    try:
        from .emotion_extractor import assign_emotions_to_destinations
    except ImportError:
        from emotion_extractor import assign_emotions_to_destinations
    try:
        from .graph_builder import GraphAuditor
    except ImportError:
        from graph_builder import GraphAuditor
    try:
        from .config import OUTPUT_DIR
    except ImportError:
        from config import OUTPUT_DIR
    import time
    import os
    import networkx as nx

    start_time = time.time()

    # Load existing state
    print(f"\n  📥 Loading existing graph state...")
    existing_graph, processed_ids = load_graph_state()
    if existing_graph is None:
        existing_graph, processed_ids = build_state_from_existing_output()
    if existing_graph is None:
        existing_graph = nx.DiGraph()

    # Deduplicate
    print(f"\n  🔍 Deduplicating...")
    new_records, skipped, fuzzy_matches = find_new_destinations(
        records, existing_graph, processed_ids
    )
    print(f"     Total: {len(records)}, Skipped: {skipped}, New: {len(new_records)}")

    if fuzzy_matches:
        print(f"\n     🔗 Fuzzy matches:")
        for new_name, existing in fuzzy_matches[:5]:
            print(f"        '{new_name}' ≈ '{existing}'")

    if not new_records:
        print("\n  ✅ All destinations already in graph. Nothing to add.")
        return None

    # LLM entity extraction
    print(f"\n  🤖 Extracting entities for {len(new_records)} new destinations...")
    new_entities, new_relationships = extract_all_entities(new_records)

    # Emotions
    new_emotions = assign_emotions_to_destinations(new_records, {})

    # Merge
    print(f"\n  🔀 Merging into graph...")
    builder = merge_into_graph(
        existing_graph, new_entities, new_relationships,
        new_emotions, new_records
    )

    # Save
    for rec in new_records:
        processed_ids.add(rec.get("entity_id", rec["name"].lower()))

    builder.export_graphml()
    builder.export_triples()
    save_graph_state(builder, processed_ids)

    auditor = GraphAuditor(builder)
    auditor.run_full_audit()

    elapsed = time.time() - start_time
    stats = builder.get_stats()

    print("\n" + "=" * 60)
    print("✅ SMART IMPORT COMPLETE")
    print("=" * 60)
    print(f"  ⏱️  Time: {elapsed:.1f}s")
    print(f"  ➕ New destinations: {len(new_records)}")
    print(f"  📊 Total nodes: {stats['total_nodes']}, edges: {stats['total_edges']}")
    print(f"  📁 Output: {OUTPUT_DIR}")
    print()

    return builder


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python smart_csv_analyzer.py "path/to/any.csv" [city_filter]')
        sys.exit(1)
    csv = sys.argv[1]
    city = sys.argv[2] if len(sys.argv) > 2 else None
    run_smart_pipeline(csv, city_filter=city)
