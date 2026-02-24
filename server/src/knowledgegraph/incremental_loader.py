"""
Incremental Loader Module — Add new CSV data to an existing Knowledge Graph
without re-processing everything.

Usage:
    python main.py --add-data "data/archive (2)/New folder/Places.csv"
"""
import os
import re
import json
import time
import pandas as pd
import networkx as nx
from fuzzywuzzy import fuzz

try:
    from .config import OUTPUT_DIR, GRAPH_OUTPUT_FILE, TRIPLES_OUTPUT_FILE
    from .data_cleaner import (
        normalize_name, normalize_place_type, normalize_season,
        create_entity_id, find_fuzzy_duplicate, load_csv_safely,
    )
    from .entity_extractor import extract_all_entities
    from .emotion_extractor import assign_emotions_to_destinations
    from .graph_builder import (
        TripleValidator, KnowledgeGraphBuilder, GraphAuditor,
        generate_quality_report,
    )
    from .graph_store import (
        save_graph_state, load_graph_state, build_state_from_existing_output,
    )
except ImportError:
    from config import OUTPUT_DIR, GRAPH_OUTPUT_FILE, TRIPLES_OUTPUT_FILE
    from data_cleaner import (
        normalize_name, normalize_place_type, normalize_season,
        create_entity_id, find_fuzzy_duplicate, load_csv_safely,
    )
    from entity_extractor import extract_all_entities
    from emotion_extractor import assign_emotions_to_destinations
    from graph_builder import (
        TripleValidator, KnowledgeGraphBuilder, GraphAuditor,
        generate_quality_report,
    )
    from graph_store import (
        save_graph_state, load_graph_state, build_state_from_existing_output,
    )


# ============================================================
# PLACES.CSV CLEANER
# ============================================================

def clean_places_csv(df, city_filter=None):
    """
    Clean 'Places.csv' format (City, Place, Ratings, Distance, Place_desc).
    
    Args:
        df: Loaded DataFrame
        city_filter: Optional city name to filter (e.g. "Chennai")
    
    Returns: list of destination records
    """
    records = []
    for _, row in df.iterrows():
        city = normalize_name(str(row.get("City", "")).strip())
        if not city:
            continue

        # Apply city filter if specified
        if city_filter and city.lower() != city_filter.lower():
            continue

        # Clean place name — strip leading numbers like "1. " or "2. "
        raw_place = str(row.get("Place", "")).strip()
        raw_place = re.sub(r'^\d+\.\s*', '', raw_place)
        name = normalize_name(raw_place)
        if not name:
            continue

        # Parse rating
        rating = row.get("Ratings", None)
        try:
            rating = float(rating) if not pd.isna(rating) else None
        except (ValueError, TypeError):
            rating = None

        # Parse distance
        distance = str(row.get("Distance", "")).strip()

        # Description for context
        desc = str(row.get("Place_desc", "")).strip()
        if desc.lower() in ("nan", "none", ""):
            desc = ""

        # Infer type from description keywords
        place_type = _infer_place_type(name, desc)

        records.append({
            "name": name,
            "country": "India",
            "continent": "Asia",
            "state": "",
            "city": city,
            "zone": "",
            "type": place_type,
            "rating": rating,
            "distance": distance,
            "description": desc[:500] if desc else "",  # Truncate long descriptions
            "source_file": "Places.csv",
            "data_quality": "verified",
            "is_synthetic": False,
        })

    return records


def _infer_place_type(name, description):
    """Infer place type from name and description keywords."""
    text = f"{name} {description}".lower()

    type_keywords = {
        "Beach": ["beach", "shore", "coastline", "sand dune"],
        "Temple": ["temple", "shrine", "mandir", "kovil", "pagoda"],
        "Fort": ["fort", "fortress", "citadel"],
        "Museum": ["museum", "gallery", "exhibition"],
        "Waterfall": ["waterfall", "falls", "cascade"],
        "Lake": ["lake", "pond", "reservoir"],
        "Park": ["park", "garden", "sanctuary", "reserve"],
        "Historical": ["historical", "heritage", "ancient", "ruins", "tomb", "monument"],
        "Religious": ["church", "mosque", "cathedral", "basilica", "gurudwara", "monastery", "gompa"],
        "Adventure": ["trek", "trekking", "rafting", "diving", "paragliding", "skiing", "bungee"],
        "Market": ["market", "shopping", "bazaar", "shop"],
        "Palace": ["palace", "haveli", "mahal"],
        "Island": ["island", "atoll"],
    }

    for ptype, keywords in type_keywords.items():
        for kw in keywords:
            if kw in text:
                return ptype

    return "Tourist Attraction"


# ============================================================
# DEDUPLICATION
# ============================================================

def find_new_destinations(cleaned_records, existing_graph, existing_ids):
    """
    Compare new records against existing graph to find truly new destinations.
    
    Returns: (new_records, skipped_count, fuzzy_matches)
    """
    new_records = []
    skipped = 0
    fuzzy_matches = []

    # Build set of existing destination names (lowercase)
    existing_names = set()
    if existing_graph is not None:
        for node, attrs in existing_graph.nodes(data=True):
            if attrs.get("node_type") == "Destination":
                existing_names.add(node.lower())
    
    # Also include names from existing_ids (entity-id based tracking)
    existing_names.update(existing_ids)

    existing_name_list = list(existing_names)

    for rec in cleaned_records:
        name_lower = rec["name"].lower()

        # Check 1: Exact match
        if name_lower in existing_names:
            skipped += 1
            continue

        # Check 2: Entity ID match
        eid = create_entity_id(rec["name"], rec["country"], rec["type"])
        if eid in existing_ids:
            skipped += 1
            continue

        # Check 3: Fuzzy match (catch near-duplicates)
        fuzzy_hit = find_fuzzy_duplicate(rec["name"], existing_name_list, threshold=85)
        if fuzzy_hit:
            skipped += 1
            fuzzy_matches.append((rec["name"], fuzzy_hit))
            continue

        # Truly new
        rec["entity_id"] = eid
        new_records.append(rec)
        # Add to existing sets to avoid intra-batch duplicates
        existing_names.add(name_lower)
        existing_name_list.append(name_lower)

    return new_records, skipped, fuzzy_matches


# ============================================================
# MERGE NEW DATA INTO EXISTING GRAPH
# ============================================================

def merge_into_graph(existing_graph, new_entities, new_relationships, new_emotion_assignments, source_destinations):
    """
    Merge new entities and relationships into an existing graph.
    Returns an updated KnowledgeGraphBuilder.
    """
    builder = KnowledgeGraphBuilder()

    # Step 1: Import existing graph
    if existing_graph is not None:
        builder.graph = existing_graph.copy()
        # Rebuild stats from existing graph
        for _, attrs in builder.graph.nodes(data=True):
            ntype = attrs.get("node_type", "Unknown")
            builder.node_count_by_type[ntype] = builder.node_count_by_type.get(ntype, 0) + 1
        for _, _, data in builder.graph.edges(data=True):
            rtype = data.get("relationship", "UNKNOWN")
            builder.rel_count_by_type[rtype] = builder.rel_count_by_type.get(rtype, 0) + 1

    # Step 2: Validate and add new entities
    validator = TripleValidator(source_destinations)

    valid_entities = []
    for entity in new_entities:
        is_valid, reason = validator.validate_entity(entity)
        if is_valid:
            valid_entities.append(entity)

    valid_relationships = []
    for rel in new_relationships:
        is_valid, reason = validator.validate_relationship(rel, valid_relationships)
        if is_valid:
            valid_relationships.append(rel)

    print(f"\n  📊 Validation (new data):")
    report = validator.get_report()
    print(f"     Entities: {report['stats']['entities_validated']} valid, "
          f"{report['stats']['entities_rejected']} rejected")
    print(f"     Relationships: {report['stats']['relationships_validated']} valid, "
          f"{report['stats']['relationships_rejected']} rejected")

    # Step 3: Add to graph
    new_nodes_before = builder.graph.number_of_nodes()
    new_edges_before = builder.graph.number_of_edges()

    for entity in valid_entities:
        builder.add_entity(entity)

    for rel in valid_relationships:
        builder.add_relationship(rel)

    # Add emotions for new destinations
    builder.add_emotion_relationships(new_emotion_assignments)

    new_nodes_added = builder.graph.number_of_nodes() - new_nodes_before
    new_edges_added = builder.graph.number_of_edges() - new_edges_before

    print(f"\n  ➕ Merged into graph:")
    print(f"     New nodes added: {new_nodes_added}")
    print(f"     New edges added: {new_edges_added}")

    return builder


# ============================================================
# MAIN INCREMENTAL PIPELINE
# ============================================================

def run_incremental_pipeline(csv_path, city_filter=None):
    """
    Run the incremental pipeline:
    1. Load existing graph state
    2. Clean new CSV data
    3. Deduplicate against existing graph
    4. LLM-extract entities for new destinations only
    5. Merge into existing graph
    6. Save updated state + export files
    """
    start_time = time.time()

    print("\n" + "=" * 60)
    print("🔄 INCREMENTAL KNOWLEDGE GRAPH UPDATE")
    print("=" * 60)
    print(f"  📂 New data: {csv_path}")
    if city_filter:
        print(f"  🏙️  City filter: {city_filter}")

    # ----------------------------
    # Step 1: Load existing state
    # ----------------------------
    print("\n" + "-" * 40)
    print("📥 Step 1: Loading existing graph state...")
    print("-" * 40)

    existing_graph, processed_ids = load_graph_state()
    if existing_graph is None:
        # Try bootstrapping from existing output files
        existing_graph, processed_ids = build_state_from_existing_output()

    if existing_graph is not None:
        print(f"  Existing graph: {existing_graph.number_of_nodes()} nodes, {existing_graph.number_of_edges()} edges")
    else:
        print("  Starting with empty graph.")
        existing_graph = nx.DiGraph()

    # ----------------------------
    # Step 2: Clean new CSV data
    # ----------------------------
    print("\n" + "-" * 40)
    print("🧹 Step 2: Cleaning new CSV data...")
    print("-" * 40)

    df = load_csv_safely(csv_path)
    if df is None:
        print("  ❌ Failed to load CSV. Aborting.")
        return None

    # Detect CSV format
    columns = [c.lower().strip() for c in df.columns]
    if "city" in columns and "place" in columns:
        cleaned = clean_places_csv(df, city_filter=city_filter)
    else:
        print(f"  ⚠️  Unrecognized CSV format. Columns: {list(df.columns)}")
        print("  Supported formats: Places.csv (City, Place, Ratings, Distance, Place_desc)")
        return None

    print(f"  Cleaned records: {len(cleaned)}")

    if not cleaned:
        print("  ⚠️  No records after cleaning. Nothing to add.")
        return None

    # ----------------------------
    # Step 3: Deduplicate
    # ----------------------------
    print("\n" + "-" * 40)
    print("🔍 Step 3: Deduplicating against existing graph...")
    print("-" * 40)

    new_records, skipped, fuzzy_matches = find_new_destinations(
        cleaned, existing_graph, processed_ids
    )

    print(f"  Total cleaned: {len(cleaned)}")
    print(f"  Already in graph (skipped): {skipped}")
    print(f"  Truly new destinations: {len(new_records)}")

    if fuzzy_matches:
        print(f"\n  🔗 Fuzzy matches (treated as duplicates):")
        for new_name, existing_name in fuzzy_matches[:10]:
            print(f"     '{new_name}' ≈ '{existing_name}'")
        if len(fuzzy_matches) > 10:
            print(f"     ... and {len(fuzzy_matches) - 10} more")

    if not new_records:
        print("\n  ✅ All destinations already exist in the graph. Nothing to add.")
        return None

    # ----------------------------
    # Step 4: LLM entity extraction
    # ----------------------------
    print("\n" + "-" * 40)
    print("🤖 Step 4: Extracting entities for new destinations...")
    print("-" * 40)

    new_entities, new_relationships = extract_all_entities(new_records)

    # Emotion assignment (using type-based defaults since no reviews for new data)
    new_emotions = assign_emotions_to_destinations(new_records, {})

    # ----------------------------
    # Step 5: Merge into graph
    # ----------------------------
    print("\n" + "-" * 40)
    print("🔀 Step 5: Merging into existing graph...")
    print("-" * 40)

    builder = merge_into_graph(
        existing_graph, new_entities, new_relationships,
        new_emotions, new_records
    )

    # ----------------------------
    # Step 6: Export and save state
    # ----------------------------
    print("\n" + "-" * 40)
    print("💾 Step 6: Saving updated graph...")
    print("-" * 40)

    # Update processed IDs
    for rec in new_records:
        processed_ids.add(rec.get("entity_id", rec["name"].lower()))

    # Export files
    builder.export_graphml()
    builder.export_triples()

    # Save state for next incremental run
    save_graph_state(builder, processed_ids)

    # Run audit
    auditor = GraphAuditor(builder)
    audit_report = auditor.run_full_audit()

    # ----------------------------
    # Summary
    # ----------------------------
    elapsed = time.time() - start_time
    stats = builder.get_stats()

    print("\n" + "=" * 60)
    print("✅ INCREMENTAL UPDATE COMPLETE")
    print("=" * 60)
    print(f"\n  ⏱️  Time: {elapsed:.1f} seconds")
    print(f"  ➕ New destinations processed: {len(new_records)}")
    print(f"  📊 Updated graph:")
    print(f"     Total nodes: {stats['total_nodes']}")
    print(f"     Total edges: {stats['total_edges']}")
    print(f"\n  📊 Nodes by type:")
    for ntype, count in sorted(stats.get("nodes_by_type", {}).items()):
        print(f"     {ntype}: {count}")
    print(f"\n  📁 Output files updated in: {OUTPUT_DIR}")
    print()

    return builder


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python incremental_loader.py <csv_path> [city_filter]")
        print('Example: python incremental_loader.py "data/archive (2)/New folder/Places.csv" Chennai')
        sys.exit(1)

    csv = sys.argv[1]
    city = sys.argv[2] if len(sys.argv) > 2 else None
    run_incremental_pipeline(csv, city_filter=city)
