"""
Graph Builder Module — Stage 3, 4, 5 of the KG Pipeline
Validates triples, builds NetworkX graph, and runs quality audit.
"""
import json
import os
import networkx as nx
try:
    from .config import (
        VALID_PREDICATES, VALID_NODE_TYPES, EMOTION_TAXONOMY,
        OUTPUT_DIR, GRAPH_OUTPUT_FILE, REPORT_OUTPUT_FILE, TRIPLES_OUTPUT_FILE,
    )
except ImportError:
    from config import (
        VALID_PREDICATES, VALID_NODE_TYPES, EMOTION_TAXONOMY,
        OUTPUT_DIR, GRAPH_OUTPUT_FILE, REPORT_OUTPUT_FILE, TRIPLES_OUTPUT_FILE,
    )


# ============================================================
# STAGE 3: TRIPLE VALIDATION
# ============================================================

class TripleValidator:
    """Validates every entity and relationship before graph insertion."""

    def __init__(self, source_destinations):
        """Initialize with source data for grounding checks."""
        self.source_names = set()
        self.source_countries = set()
        self.source_types = set()
        self.conflicts_log = []
        self.rejected_log = []
        self.stats = {
            "entities_validated": 0,
            "entities_rejected": 0,
            "relationships_validated": 0,
            "relationships_rejected": 0,
        }

        # Build source data index for grounding
        for dest in source_destinations:
            if dest.get("name"):
                self.source_names.add(dest["name"].lower().strip())
            if dest.get("country"):
                self.source_countries.add(dest["country"].lower().strip())
            if dest.get("type"):
                self.source_types.add(dest["type"].lower().strip())

    def validate_entity(self, entity):
        """
        Validate an entity against schema rules.
        Returns: (is_valid, reason)
        """
        name = entity.get("name", "").strip()
        etype = entity.get("type", "").strip()

        if not name:
            self.stats["entities_rejected"] += 1
            return False, "Empty name"

        if etype not in VALID_NODE_TYPES:
            self.stats["entities_rejected"] += 1
            reason = f"Invalid type '{etype}' (valid: {VALID_NODE_TYPES})"
            self.rejected_log.append(f"ENTITY REJECTED: {name} — {reason}")
            return False, reason

        # Emotion entities must be in our taxonomy
        if etype == "Emotion" and name.lower() not in EMOTION_TAXONOMY:
            self.stats["entities_rejected"] += 1
            reason = f"Emotion '{name}' not in fixed taxonomy"
            self.rejected_log.append(f"ENTITY REJECTED: {name} — {reason}")
            return False, reason

        self.stats["entities_validated"] += 1
        return True, "OK"

    def validate_relationship(self, relationship, existing_relationships=None):
        """
        Validate a relationship with 3 checks:
        1. Schema validation
        2. Source grounding
        3. Conflict detection
        Returns: (is_valid, reason)
        """
        source = relationship.get("source", "").strip()
        rel_type = relationship.get("type", "").strip()
        target = relationship.get("target", "").strip()

        # Check 1: Schema validation
        if not source or not target:
            self.stats["relationships_rejected"] += 1
            return False, "Empty source or target"

        # Check: Self-loop detection
        if source == target:
            self.stats["relationships_rejected"] += 1
            reason = f"Self-loop: '{source}' cannot relate to itself"
            self.rejected_log.append(f"REL REJECTED: {source} —[{rel_type}]→ {target} — {reason}")
            return False, reason

        if rel_type not in VALID_PREDICATES:
            self.stats["relationships_rejected"] += 1
            reason = f"Invalid relationship type '{rel_type}'"
            self.rejected_log.append(f"REL REJECTED: {source} —[{rel_type}]→ {target} — {reason}")
            return False, reason

        # Check 2: Conflict detection (e.g., same city in two different states)
        if existing_relationships:
            for existing in existing_relationships:
                if (existing["source"] == source and
                    existing["type"] == rel_type and
                    existing["target"] != target and
                    rel_type in ("LOCATED_IN", "BELONGS_TO", "PART_OF", "ON_CONTINENT")):
                    self.stats["relationships_rejected"] += 1
                    reason = (f"CONFLICT: {source} already {rel_type} "
                             f"{existing['target']}, cannot also be {target}")
                    self.conflicts_log.append(reason)
                    self.rejected_log.append(f"REL REJECTED: {reason}")
                    return False, reason

        self.stats["relationships_validated"] += 1
        return True, "OK"

    def get_report(self):
        """Generate validation report."""
        return {
            "stats": self.stats,
            "conflicts": self.conflicts_log,
            "rejected_count": len(self.rejected_log),
            "rejected_samples": self.rejected_log[:20],
        }


# ============================================================
# STAGE 4: GRAPH BUILDING
# ============================================================

class KnowledgeGraphBuilder:
    """Builds and manages the NetworkX knowledge graph."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_count_by_type = {}
        self.rel_count_by_type = {}

    def add_entity(self, entity):
        """Add a validated entity as a node."""
        name = entity["name"]
        etype = entity["type"]
        properties = entity.get("properties", {})

        # Clean properties (remove None values)
        clean_props = {k: v for k, v in properties.items() if v is not None}
        clean_props["node_type"] = etype

        if self.graph.has_node(name):
            # Merge properties (don't overwrite existing)
            existing = self.graph.nodes[name]
            for k, v in clean_props.items():
                if k not in existing or not existing[k]:
                    existing[k] = v
        else:
            self.graph.add_node(name, **clean_props)

        self.node_count_by_type[etype] = self.node_count_by_type.get(etype, 0) + 1

    def add_relationship(self, relationship):
        """Add a validated relationship as an edge."""
        source = relationship["source"]
        rel_type = relationship["type"]
        target = relationship["target"]

        # Prevent edges on nodes with wrong types
        if self.graph.has_node(source):
            src_type = self.graph.nodes[source].get("node_type", "")
            # A PlaceType node should not have Destination-like edges
            if src_type == "PlaceType" and rel_type in ("LOCATED_IN", "BEST_VISITED_IN", "EVOKES"):
                return
        if self.graph.has_node(target):
            tgt_type = self.graph.nodes[target].get("node_type", "")
            # A Destination should not be a target of LOCATED_IN
            if tgt_type == "Destination" and rel_type == "LOCATED_IN":
                return

        # Ensure both nodes exist
        if not self.graph.has_node(source):
            self.graph.add_node(source, node_type="Unknown")
        if not self.graph.has_node(target):
            self.graph.add_node(target, node_type="Unknown")

        # Add edge (or update if exists)
        self.graph.add_edge(source, target, relationship=rel_type)
        self.rel_count_by_type[rel_type] = self.rel_count_by_type.get(rel_type, 0) + 1

    def add_emotion_relationships(self, emotion_assignments):
        """Add Tier 3 emotion entities and EVOKES relationships."""
        for assignment in emotion_assignments:
            dest_name = assignment["destination_name"]
            emotions = assignment["emotions"]

            for emotion in emotions:
                # Create Emotion node if not exists
                if not self.graph.has_node(emotion.title()):
                    self.graph.add_node(
                        emotion.title(),
                        node_type="Emotion",
                        description=EMOTION_TAXONOMY.get(emotion, "")
                    )

                # Create EVOKES relationship
                if self.graph.has_node(dest_name):
                    self.graph.add_edge(dest_name, emotion.title(), relationship="EVOKES")
                    self.rel_count_by_type["EVOKES"] = self.rel_count_by_type.get("EVOKES", 0) + 1

    def get_stats(self):
        """Get graph statistics."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "nodes_by_type": self.node_count_by_type,
            "edges_by_type": self.rel_count_by_type,
        }

    def export_graphml(self, filepath=None):
        """Export graph to GraphML format."""
        if filepath is None:
            filepath = GRAPH_OUTPUT_FILE
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nx.write_graphml(self.graph, filepath)
        print(f"  💾 Graph exported to: {filepath}")

    def export_triples(self, filepath=None):
        """Export all triples as JSON."""
        if filepath is None:
            filepath = TRIPLES_OUTPUT_FILE
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        triples = []
        for source, target, data in self.graph.edges(data=True):
            triples.append({
                "subject": source,
                "predicate": data.get("relationship", "UNKNOWN"),
                "object": target,
                "subject_type": self.graph.nodes[source].get("node_type", "Unknown"),
                "object_type": self.graph.nodes[target].get("node_type", "Unknown"),
            })

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(triples, f, indent=2, ensure_ascii=False)
        print(f"  💾 Triples exported to: {filepath}")

        return triples


# ============================================================
# STAGE 5: QUALITY AUDIT
# ============================================================

class GraphAuditor:
    """Post-insertion quality audit for the knowledge graph."""

    def __init__(self, graph_builder):
        self.graph = graph_builder.graph
        self.issues = []

    def check_orphan_nodes(self):
        """Find nodes with no relationships."""
        orphans = []
        for node in self.graph.nodes():
            if (self.graph.in_degree(node) == 0 and
                self.graph.out_degree(node) == 0):
                orphans.append(node)

        if orphans:
            self.issues.append(f"⚠️  {len(orphans)} orphan nodes found (no relationships)")
            for orphan in orphans[:10]:
                node_type = self.graph.nodes[orphan].get("node_type", "Unknown")
                self.issues.append(f"   - {orphan} ({node_type})")

        return orphans

    def check_destinations_completeness(self):
        """Check that every Destination has required relationships."""
        incomplete = []
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") != "Destination":
                continue

            has_type = False
            has_emotion = False
            has_location = False

            for _, target, edge_data in self.graph.out_edges(node, data=True):
                rel = edge_data.get("relationship", "")
                if rel == "HAS_TYPE":
                    has_type = True
                elif rel == "EVOKES":
                    has_emotion = True
                elif rel in ("LOCATED_IN", "BELONGS_TO"):
                    has_location = True

            missing = []
            if not has_type:
                missing.append("HAS_TYPE")
            if not has_emotion:
                missing.append("EVOKES")
            if not has_location:
                missing.append("LOCATED_IN")

            if missing:
                incomplete.append((node, missing))

        if incomplete:
            self.issues.append(f"⚠️  {len(incomplete)} destinations missing required relationships")
            for dest, missing in incomplete[:10]:
                self.issues.append(f"   - {dest}: missing {', '.join(missing)}")

        return incomplete

    def check_tier_coverage(self):
        """Verify all 3 tiers have data."""
        tier1_types = {"Continent", "Country", "State", "City"}
        tier2_types = {"Destination", "PlaceType"}
        tier3_types = {"Emotion"}

        tier1_count = sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") in tier1_types)
        tier2_count = sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") in tier2_types)
        tier3_count = sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") in tier3_types)

        coverage = {
            "tier1_geography": tier1_count,
            "tier2_destinations": tier2_count,
            "tier3_emotions": tier3_count,
        }

        if tier1_count == 0:
            self.issues.append("❌ CRITICAL: Tier 1 (Geography) is EMPTY")
        if tier2_count == 0:
            self.issues.append("❌ CRITICAL: Tier 2 (Destinations/Types) is EMPTY")
        if tier3_count == 0:
            self.issues.append("❌ CRITICAL: Tier 3 (Emotions) is EMPTY")

        return coverage

    def check_data_quality_distribution(self):
        """Check distribution of data quality tags."""
        quality_dist = {}
        for _, data in self.graph.nodes(data=True):
            quality = data.get("data_quality", "untagged")
            quality_dist[quality] = quality_dist.get(quality, 0) + 1
        return quality_dist

    def run_full_audit(self):
        """Run all audit checks and generate report."""
        print("\n" + "=" * 60)
        print("🔍 STAGE 5: QUALITY AUDIT")
        print("=" * 60)

        # Run checks
        orphans = self.check_orphan_nodes()
        incomplete = self.check_destinations_completeness()
        coverage = self.check_tier_coverage()
        quality_dist = self.check_data_quality_distribution()

        # Print results
        print(f"\n  📊 Tier Coverage:")
        print(f"     Tier 1 (Geography): {coverage['tier1_geography']} nodes")
        print(f"     Tier 2 (Destinations): {coverage['tier2_destinations']} nodes")
        print(f"     Tier 3 (Emotions): {coverage['tier3_emotions']} nodes")

        print(f"\n  📊 Data Quality Distribution:")
        for quality, count in sorted(quality_dist.items()):
            print(f"     {quality}: {count}")

        print(f"\n  📊 Integrity Checks:")
        print(f"     Orphan nodes: {len(orphans)}")
        print(f"     Incomplete destinations: {len(incomplete)}")

        if self.issues:
            print(f"\n  ⚠️  Issues Found ({len(self.issues)}):")
            for issue in self.issues:
                print(f"     {issue}")
        else:
            print(f"\n  ✅ No issues found — graph is clean!")

        return {
            "orphan_count": len(orphans),
            "incomplete_count": len(incomplete),
            "tier_coverage": coverage,
            "quality_distribution": quality_dist,
            "issues": self.issues,
        }


def generate_quality_report(graph_stats, validation_report, audit_report, filepath=None):
    """Generate and save a human-readable quality report."""
    if filepath is None:
        filepath = REPORT_OUTPUT_FILE
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    lines = [
        "=" * 60,
        "TOURISM KNOWLEDGE GRAPH — QUALITY REPORT",
        "=" * 60,
        "",
        "GRAPH STATISTICS:",
        f"  Total nodes: {graph_stats['total_nodes']}",
        f"  Total edges: {graph_stats['total_edges']}",
        "",
        "  Nodes by type:",
    ]
    for ntype, count in sorted(graph_stats.get("nodes_by_type", {}).items()):
        lines.append(f"    {ntype}: {count}")

    lines.extend([
        "",
        "  Edges by type:",
    ])
    for etype, count in sorted(graph_stats.get("edges_by_type", {}).items()):
        lines.append(f"    {etype}: {count}")

    lines.extend([
        "",
        "VALIDATION RESULTS:",
        f"  Entities validated: {validation_report['stats']['entities_validated']}",
        f"  Entities rejected: {validation_report['stats']['entities_rejected']}",
        f"  Relationships validated: {validation_report['stats']['relationships_validated']}",
        f"  Relationships rejected: {validation_report['stats']['relationships_rejected']}",
        f"  Conflicts detected: {len(validation_report['conflicts'])}",
        "",
        "AUDIT RESULTS:",
        f"  Orphan nodes: {audit_report['orphan_count']}",
        f"  Incomplete destinations: {audit_report['incomplete_count']}",
        "",
        "TIER COVERAGE:",
        f"  Tier 1 (Geography): {audit_report['tier_coverage']['tier1_geography']}",
        f"  Tier 2 (Destinations/Types): {audit_report['tier_coverage']['tier2_destinations']}",
        f"  Tier 3 (Emotions): {audit_report['tier_coverage']['tier3_emotions']}",
        "",
    ])

    if audit_report["issues"]:
        lines.append("ISSUES:")
        for issue in audit_report["issues"]:
            lines.append(f"  {issue}")
    else:
        lines.append("✅ NO ISSUES — Graph passed all quality checks!")

    report_text = "\n".join(lines)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  📄 Quality report saved to: {filepath}")

    return report_text


# ============================================================
# BUILD & VALIDATE — Full pipeline stages 3-5
# ============================================================

def build_and_validate_graph(entities, relationships, emotion_assignments, source_destinations):
    """
    Stages 3-5: Validate → Build → Audit
    Returns: (graph_builder, quality_report_text)
    """
    # Stage 3: Validation
    print("\n" + "=" * 60)
    print("✅ STAGE 3: TRIPLE VALIDATION")
    print("=" * 60)

    validator = TripleValidator(source_destinations)

    valid_entities = []
    for entity in entities:
        is_valid, reason = validator.validate_entity(entity)
        if is_valid:
            valid_entities.append(entity)

    valid_relationships = []
    for rel in relationships:
        is_valid, reason = validator.validate_relationship(rel, valid_relationships)
        if is_valid:
            valid_relationships.append(rel)

    validation_report = validator.get_report()
    print(f"\n  📊 Validation Results:")
    print(f"     Entities: {validation_report['stats']['entities_validated']} valid, "
          f"{validation_report['stats']['entities_rejected']} rejected")
    print(f"     Relationships: {validation_report['stats']['relationships_validated']} valid, "
          f"{validation_report['stats']['relationships_rejected']} rejected")
    print(f"     Conflicts: {len(validation_report['conflicts'])}")

    # Stage 4: Graph Building
    print("\n" + "=" * 60)
    print("🏗️  STAGE 4: GRAPH CONSTRUCTION")
    print("=" * 60)

    builder = KnowledgeGraphBuilder()

    for entity in valid_entities:
        builder.add_entity(entity)

    for rel in valid_relationships:
        builder.add_relationship(rel)

    # Add Tier 3: Emotions
    builder.add_emotion_relationships(emotion_assignments)

    graph_stats = builder.get_stats()
    print(f"\n  📊 Graph Built:")
    print(f"     Nodes: {graph_stats['total_nodes']}")
    print(f"     Edges: {graph_stats['total_edges']}")

    # Export
    builder.export_graphml()
    triples = builder.export_triples()

    # Stage 5: Audit
    auditor = GraphAuditor(builder)
    audit_report = auditor.run_full_audit()

    # Generate report
    report_text = generate_quality_report(graph_stats, validation_report, audit_report)

    return builder, report_text
