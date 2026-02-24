"""
Tourism Knowledge Graph Pipeline — Main Orchestrator
5-Stage Pipeline: Clean → Extract → Validate → Insert → Audit

Supports two modes:
  1. Full pipeline:        python main.py
  2. Incremental update:   python main.py --add-data "path/to/new.csv" [--city-filter Chennai]

Uses DeepSeek V3.1 671B via Ollama
"""
import sys
import os
import time
import json
import argparse
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .data_cleaner import clean_all_data
    from .entity_extractor import extract_all_entities
    from .emotion_extractor import extract_emotions_from_reviews, assign_emotions_to_destinations
    from .graph_builder import build_and_validate_graph
    from .config import OUTPUT_DIR, OLLAMA_MODEL, AGENT_OUTPUT_DIR
except ImportError:
    from data_cleaner import clean_all_data
    from entity_extractor import extract_all_entities
    from emotion_extractor import extract_emotions_from_reviews, assign_emotions_to_destinations
    from graph_builder import build_and_validate_graph
    from config import OUTPUT_DIR, OLLAMA_MODEL, AGENT_OUTPUT_DIR


def sync_to_agent_output():
    """Copy output files to agents/tools/kg_output/ for agent access."""
    os.makedirs(AGENT_OUTPUT_DIR, exist_ok=True)
    for fname in os.listdir(OUTPUT_DIR):
        src = os.path.join(OUTPUT_DIR, fname)
        dst = os.path.join(AGENT_OUTPUT_DIR, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    print(f"  📤 Synced output to agent tools: {AGENT_OUTPUT_DIR}")


def print_banner(mode="full"):
    """Print welcome banner."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║     🗺️  TOURISM KNOWLEDGE GRAPH PIPELINE              ║")
    print("║     Powered by DeepSeek V3.1 via Ollama               ║")
    if mode == "incremental":
        print("║     🔄 MODE: INCREMENTAL UPDATE                       ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Model: {OLLAMA_MODEL}")
    print(f"  Output: {OUTPUT_DIR}")
    print()


def main():
    """Run the full 5-stage pipeline."""
    start_time = time.time()
    print_banner()

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========================================
    # STAGE 1: DATA CLEANING
    # ========================================
    destinations, reviews = clean_all_data()

    if not destinations:
        print("\n❌ FATAL: No destinations found after cleaning. Check your CSV files.")
        sys.exit(1)

    # ========================================
    # STAGE 2a: ENTITY EXTRACTION (LLM)
    # ========================================
    entities, relationships = extract_all_entities(destinations)

    # ========================================
    # STAGE 2b: EMOTION EXTRACTION (LLM)
    # ========================================
    review_emotions = {}
    if reviews:
        review_emotions = extract_emotions_from_reviews(reviews)

    emotion_assignments = assign_emotions_to_destinations(destinations, review_emotions)

    # ========================================
    # STAGES 3-5: VALIDATE → BUILD → AUDIT
    # ========================================
    graph_builder, quality_report = build_and_validate_graph(
        entities=entities,
        relationships=relationships,
        emotion_assignments=emotion_assignments,
        source_destinations=destinations,
    )

    # Save state for future incremental runs
    from graph_store import save_graph_state
    processed_ids = set()
    for d in destinations:
        processed_ids.add(d.get("entity_id", d["name"].lower()))
    save_graph_state(graph_builder, processed_ids)

    # Sync output to agent tools directory
    sync_to_agent_output()

    # ========================================
    # FINAL SUMMARY
    # ========================================
    elapsed = time.time() - start_time
    stats = graph_builder.get_stats()

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║     ✅ PIPELINE COMPLETE                               ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  ⏱️  Time elapsed: {elapsed:.1f} seconds")
    print(f"\n  📊 Final Graph:")
    print(f"     Total nodes: {stats['total_nodes']}")
    print(f"     Total edges: {stats['total_edges']}")
    print(f"\n  📊 Nodes by type:")
    for ntype, count in sorted(stats.get("nodes_by_type", {}).items()):
        print(f"     {ntype}: {count}")
    print(f"\n  📊 Edges by type:")
    for etype, count in sorted(stats.get("edges_by_type", {}).items()):
        print(f"     {etype}: {count}")
    print(f"\n  📁 Output files:")
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"     {f} ({size:,} bytes)")
    print()

    return graph_builder


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tourism Knowledge Graph Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                          # Full pipeline
  python main.py --add-data "data/Places.csv"             # Incremental (known CSV format)
  python main.py --add-data "data/Places.csv" --city-filter Chennai
  python main.py --smart-add "data/any_file.csv"          # LLM-powered (any CSV)
  python main.py --smart-add "data/any_file.csv" --city-filter Goa
        """
    )
    parser.add_argument(
        "--add-data",
        type=str,
        help="Path to a new CSV file to incrementally add (deterministic mode, known formats)",
    )
    parser.add_argument(
        "--smart-add",
        type=str,
        help="Path to ANY CSV file — LLM auto-detects column structure (universal mode)",
    )
    parser.add_argument(
        "--city-filter",
        type=str,
        default=None,
        help="Filter new data by city name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.smart_add:
        # Smart mode — LLM analyzes CSV structure
        print_banner(mode="incremental")
        from smart_csv_analyzer import run_smart_pipeline

        csv_path = args.smart_add
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path)

        run_smart_pipeline(csv_path, city_filter=args.city_filter)
        sync_to_agent_output()

    elif args.add_data:
        # Incremental mode — deterministic cleaners
        print_banner(mode="incremental")
        from incremental_loader import run_incremental_pipeline

        csv_path = args.add_data
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path)

        run_incremental_pipeline(csv_path, city_filter=args.city_filter)
        sync_to_agent_output()
    else:
        # Full pipeline
        main()
