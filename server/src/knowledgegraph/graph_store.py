"""
Graph Store Module — Persistence layer for the Knowledge Graph
Saves and loads graph state so incremental updates can resume.
"""
import json
import os
import networkx as nx
try:
    from .config import OUTPUT_DIR, GRAPH_OUTPUT_FILE, TRIPLES_OUTPUT_FILE
except ImportError:
    from config import OUTPUT_DIR, GRAPH_OUTPUT_FILE, TRIPLES_OUTPUT_FILE


STATE_FILE = os.path.join(OUTPUT_DIR, "graph_state.json")


def save_graph_state(graph_builder, processed_ids=None):
    """
    Save full graph state to disk.
    Persists:
      - All triples (edges with node types)
      - All node metadata (properties)
      - Set of processed destination entity_ids
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect node data
    nodes = {}
    for node_name, attrs in graph_builder.graph.nodes(data=True):
        nodes[node_name] = dict(attrs)

    # Collect edge data
    triples = []
    for source, target, data in graph_builder.graph.edges(data=True):
        triples.append({
            "subject": source,
            "predicate": data.get("relationship", "UNKNOWN"),
            "object": target,
            "subject_type": graph_builder.graph.nodes[source].get("node_type", "Unknown"),
            "object_type": graph_builder.graph.nodes[target].get("node_type", "Unknown"),
        })

    state = {
        "nodes": nodes,
        "triples": triples,
        "processed_ids": list(processed_ids) if processed_ids else [],
    }

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    print(f"  💾 Graph state saved to: {STATE_FILE}")
    print(f"     Nodes: {len(nodes)}, Edges: {len(triples)}, Tracked IDs: {len(state['processed_ids'])}")

    return state


def load_graph_state():
    """
    Load graph state from disk.
    Returns: (nx.DiGraph, set_of_processed_ids)
    Returns (None, set()) if no state file exists.
    """
    if not os.path.exists(STATE_FILE):
        print("  ℹ️  No existing graph state found — starting fresh.")
        return None, set()

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    # Rebuild the NetworkX graph
    graph = nx.DiGraph()

    # Add nodes with all their metadata
    for node_name, attrs in state.get("nodes", {}).items():
        graph.add_node(node_name, **attrs)

    # Add edges
    for triple in state.get("triples", []):
        source = triple["subject"]
        target = triple["object"]
        rel_type = triple["predicate"]

        # Ensure nodes exist (safety)
        if not graph.has_node(source):
            graph.add_node(source, node_type=triple.get("subject_type", "Unknown"))
        if not graph.has_node(target):
            graph.add_node(target, node_type=triple.get("object_type", "Unknown"))

        graph.add_edge(source, target, relationship=rel_type)

    processed_ids = set(state.get("processed_ids", []))

    print(f"  ✅ Graph state loaded from: {STATE_FILE}")
    print(f"     Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}, Tracked IDs: {len(processed_ids)}")

    return graph, processed_ids


def build_state_from_existing_output():
    """
    Bootstrap graph state from existing extracted_triples.json.
    Use this when running incremental mode for the first time
    (state file doesn't exist yet but output files do).
    """
    if not os.path.exists(TRIPLES_OUTPUT_FILE):
        print("  ⚠️  No existing triples found — cannot bootstrap.")
        return None, set()

    with open(TRIPLES_OUTPUT_FILE, "r", encoding="utf-8") as f:
        triples = json.load(f)

    graph = nx.DiGraph()
    destination_names = set()

    for triple in triples:
        source = triple["subject"]
        target = triple["object"]
        rel_type = triple["predicate"]
        src_type = triple.get("subject_type", "Unknown")
        tgt_type = triple.get("object_type", "Unknown")

        if not graph.has_node(source):
            graph.add_node(source, node_type=src_type)
        if not graph.has_node(target):
            graph.add_node(target, node_type=tgt_type)

        graph.add_edge(source, target, relationship=rel_type)

        # Track Destination names for deduplication
        if src_type == "Destination":
            destination_names.add(source.lower())

    print(f"  🔄 Bootstrapped state from existing triples:")
    print(f"     Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    print(f"     Known destinations: {len(destination_names)}")

    return graph, destination_names


if __name__ == "__main__":
    # Quick test — try loading
    graph, ids = load_graph_state()
    if graph is None:
        print("\nAttempting bootstrap from existing output...")
        graph, ids = build_state_from_existing_output()
    if graph:
        print(f"\nGraph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
