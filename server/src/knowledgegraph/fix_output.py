"""
Post-Fix Script — Patches existing extracted_triples.json
Removes self-loops, fixes type conflicts, and regenerates clean output.
No LLM calls needed — runs instantly.

Usage:
    python fix_output.py
"""
import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
TRIPLES_FILE = os.path.join(OUTPUT_DIR, "extracted_triples.json")

def main():
    print("\n🔧 Post-Fix: Patching extracted_triples.json")
    print("=" * 50)

    # Load existing triples
    with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
        triples = json.load(f)

    original_count = len(triples)
    print(f"  📂 Loaded {original_count} triples")

    # Track what we remove
    self_loops_removed = 0
    type_conflicts_removed = 0

    # Build a type registry: first occurrence determines the canonical type
    node_types = {}
    for t in triples:
        key_s = t["subject"]
        key_o = t["object"]
        if key_s not in node_types:
            node_types[key_s] = t["subject_type"]
        if key_o not in node_types:
            node_types[key_o] = t["object_type"]

    cleaned = []
    for t in triples:
        subj = t["subject"]
        obj = t["object"]
        pred = t["predicate"]
        subj_type = t["subject_type"]

        # Fix 1: Remove self-loops
        if subj == obj:
            self_loops_removed += 1
            continue

        # Fix 2: Remove edges where PlaceType has Destination-like relationships
        if subj_type == "PlaceType" and pred in ("LOCATED_IN", "BEST_VISITED_IN", "EVOKES"):
            type_conflicts_removed += 1
            continue

        # Fix 2b: Remove edges where Destination is a target of LOCATED_IN
        obj_type = t["object_type"]
        if obj_type == "Destination" and pred == "LOCATED_IN":
            type_conflicts_removed += 1
            continue

        cleaned.append(t)

    # Save cleaned triples
    with open(TRIPLES_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Self-loops removed: {self_loops_removed}")
    print(f"  ✅ Type conflicts removed: {type_conflicts_removed}")
    print(f"  📊 Triples: {original_count} → {len(cleaned)} ({original_count - len(cleaned)} removed)")
    print(f"\n  💾 Saved to: {TRIPLES_FILE}")

    # Also regenerate the GraphML from cleaned triples
    try:
        import networkx as nx

        G = nx.DiGraph()
        for t in cleaned:
            subj = t["subject"]
            obj = t["object"]
            subj_type = t["subject_type"]
            obj_type = t["object_type"]
            pred = t["predicate"]

            if not G.has_node(subj):
                G.add_node(subj, node_type=subj_type)
            if not G.has_node(obj):
                G.add_node(obj, node_type=obj_type)
            G.add_edge(subj, obj, relationship=pred)

        graphml_path = os.path.join(OUTPUT_DIR, "tourism_knowledge_graph.graphml")
        nx.write_graphml(G, graphml_path)
        print(f"\n  🏗️  Regenerated GraphML: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"  💾 Saved to: {graphml_path}")
    except ImportError:
        print("\n  ⚠️  NetworkX not available — skipped GraphML regeneration")

    print(f"\n✅ Done! Open the graph viewer to see the fixed graph.")
    print(f"   python -m http.server 8888")
    print(f"   http://localhost:8888/output/graph_viewer.html")


if __name__ == "__main__":
    main()
