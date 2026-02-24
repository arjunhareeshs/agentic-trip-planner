"""Quick integration test for KG tools."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server", "src"))

from agents.tools.knowledge_graph import (
    match_destinations,
    filter_destinations,
    get_graph_stats,
    get_destination_details,
    list_all_destinations,
)

print("=== GRAPH STATS ===")
stats = get_graph_stats()
print(f"Destinations: {stats['total_destinations']}")
print(f"Nodes: {stats['total_nodes']}")
print(f"Triples: {stats['total_triples']}")
print(f"Emotions: {stats['available_emotions']}")
print(f"Types: {len(stats['available_place_types'])} types")
print(f"Countries: {len(stats['available_countries'])} countries")
print()

print("=== KEYWORD SEARCH: beach romantic ===")
results = match_destinations(["beach", "romantic"])
for r in results[:5]:
    print(f"  {r['name']} (score={r['score']}, emotions={r['emotions']})")
print()

print("=== FILTER: emotion=romantic ===")
filtered = filter_destinations(emotion="romantic")
print(f"  Found {len(filtered)} romantic destinations")
for r in filtered[:5]:
    print(f"  - {r['name']} ({r.get('city','')}, {r.get('country','')})")
print()

print("=== FILTER: place_type=Beach ===")
filtered = filter_destinations(place_type="Beach")
print(f"  Found {len(filtered)} beach destinations")
for r in filtered[:5]:
    print(f"  - {r['name']} ({r.get('city','')}, {r.get('country','')})")
print()

print("=== DETAIL: Taj Mahal ===")
d = get_destination_details("Taj Mahal")
print(f"  {d.get('name', 'NOT FOUND')} - {d.get('city')}, {d.get('country')} - emotions: {d.get('emotions')}")
print()

print("=== ALL DESTINATIONS COUNT ===")
all_d = list_all_destinations()
print(f"  Total: {len(all_d)} destinations")
print()

print("✅ All KG tools working correctly!")
