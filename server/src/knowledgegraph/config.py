"""
Configuration for Tourism Knowledge Graph Pipeline
Uses DeepSeek V3.1 671B via Ollama (local)
"""
import os

# ============================================================
# OLLAMA LLM CONFIGURATION
# ============================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("KG_MODEL", "deepseek-v3.1:671b-cloud")

# LLM settings
LLM_TEMPERATURE = 0.1  # Low temperature for deterministic, structured output
LLM_MAX_TOKENS = 8192
LLM_RETRY_ATTEMPTS = 3
BATCH_SIZE = 10  # Number of CSV rows per LLM call (optimized)

# ============================================================
# DATA FILE PATHS
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Priority files (verified/useful data)
PRIORITY_FILES = {
    "indian_places": os.path.join(DATA_DIR, "Top Indian Places to Visit.csv"),
    "expanded_destinations": os.path.join(DATA_DIR, "archive (3)", "data", "Expanded_Destinations.csv"),
    "reviews": os.path.join(DATA_DIR, "archive (3)", "data", "Final_Updated_Expanded_Reviews.csv"),
    "tourist_destinations": os.path.join(DATA_DIR, "Tourist_Destinations.csv"),
    "travel_details": os.path.join(DATA_DIR, "archive (4)", "Travel details dataset.csv"),
    "cities": os.path.join(DATA_DIR, "archive (2)", "New folder", "City.csv"),
    "places": os.path.join(DATA_DIR, "archive (2)", "New folder", "Places.csv"),
}

# Blocked files (garbage data — DO NOT LOAD)
BLOCKED_FILES = [
    "tourism_dataset.csv",  # Random string location names — completely useless
]

# ============================================================
# OUTPUT PATHS
# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
GRAPH_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tourism_knowledge_graph.graphml")
REPORT_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "quality_report.txt")
TRIPLES_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "extracted_triples.json")

# Also export to agents/tools/kg_output for agent access
_AGENTS_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "agents", "tools", "kg_output")
AGENT_OUTPUT_DIR = os.path.normpath(_AGENTS_TOOLS_DIR)

# ============================================================
# FIXED EMOTION TAXONOMY (Tier 3)
# Only these 9 emotions are allowed in the graph
# ============================================================
EMOTION_TAXONOMY = {
    "peaceful": "Calm, serene, meditative experiences",
    "thrilling": "Adventure, adrenaline, excitement",
    "romantic": "Love, intimacy, couples experiences",
    "spiritual": "Divine, sacred, religious fulfillment",
    "nostalgic": "Historical wonder, connection to the past",
    "joyful": "Fun, happiness, family-friendly",
    "awe": "Grandeur, breathtaking, overwhelming beauty",
    "curious": "Learning, exploration, discovery",
    "relaxed": "Stress-free, vacation, unwinding",
}

# Default emotion mapping by place type (fallback when LLM/reviews unavailable)
DEFAULT_EMOTION_MAP = {
    # Nature
    "beach": ["relaxed", "romantic", "joyful"],
    "lake": ["peaceful", "romantic"],
    "waterfall": ["awe", "thrilling"],
    "hill station": ["peaceful", "romantic"],
    "national park": ["awe", "curious", "thrilling"],
    "wildlife sanctuary": ["awe", "curious", "thrilling"],
    "garden": ["peaceful", "relaxed"],
    "island": ["relaxed", "romantic", "awe"],
    "valley": ["peaceful", "awe"],
    "forest": ["peaceful", "curious"],
    "mountain": ["awe", "thrilling"],
    "nature": ["peaceful", "awe", "relaxed"],
    "nature reserve": ["peaceful", "curious"],

    # Historical & Cultural
    "fort": ["nostalgic", "curious", "awe"],
    "palace": ["awe", "nostalgic", "curious"],
    "monument": ["nostalgic", "awe", "curious"],
    "historical": ["nostalgic", "curious"],
    "heritage site": ["nostalgic", "awe"],
    "museum": ["curious", "nostalgic"],
    "archaeological site": ["curious", "nostalgic"],
    "ruins": ["nostalgic", "curious"],
    "cave": ["curious", "awe", "thrilling"],
    "cultural": ["curious", "joyful"],

    # Religious & Spiritual
    "temple": ["spiritual", "peaceful"],
    "church": ["spiritual", "peaceful"],
    "mosque": ["spiritual", "peaceful"],
    "gurudwara": ["spiritual", "peaceful"],
    "monastery": ["spiritual", "peaceful"],
    "religious": ["spiritual", "peaceful"],
    "pilgrimage": ["spiritual"],

    # Urban & Entertainment
    "city": ["curious", "joyful"],
    "urban": ["curious", "joyful"],
    "market": ["joyful", "curious"],
    "zoo": ["joyful", "curious"],
    "amusement park": ["joyful", "thrilling"],
    "theme park": ["joyful", "thrilling"],
    "shopping": ["joyful"],

    # Adventure
    "adventure": ["thrilling", "awe"],
    "trekking": ["thrilling", "awe"],
    "diving": ["thrilling", "awe"],
    "rafting": ["thrilling", "joyful"],
    "skiing": ["thrilling", "joyful"],

    # Default fallback
    "other": ["curious"],
}

# ============================================================
# COUNTRY ALIAS MAP — Canonical names
# ============================================================
COUNTRY_ALIASES = {
    # Americas
    "usa": "United States", "us": "United States",
    "united states of america": "United States",
    "u.s.a.": "United States", "u.s.": "United States",
    "america": "United States",

    # Europe
    "uk": "United Kingdom", "u.k.": "United Kingdom",
    "england": "United Kingdom", "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "holland": "Netherlands",
    "czech republic": "Czechia",

    # Asia
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "south korea": "South Korea", "korea": "South Korea",
    "republic of korea": "South Korea",
    "burma": "Myanmar",

    # Common variations
    "people's republic of china": "China",
    "prc": "China",
    "russian federation": "Russia",
    "republic of india": "India",
    "bharat": "India",
}

# Continent mapping for countries
CONTINENT_MAP = {
    "India": "Asia", "China": "Asia", "Japan": "Asia", "Thailand": "Asia",
    "Indonesia": "Asia", "Vietnam": "Asia", "Malaysia": "Asia",
    "South Korea": "Asia", "Nepal": "Asia", "Sri Lanka": "Asia",
    "Myanmar": "Asia", "Cambodia": "Asia", "Philippines": "Asia",
    "Singapore": "Asia", "Bangladesh": "Asia", "Pakistan": "Asia",
    "United Arab Emirates": "Asia", "Turkey": "Asia", "Israel": "Asia",
    "Jordan": "Asia", "Saudi Arabia": "Asia", "Oman": "Asia",
    "Maldives": "Asia", "Bhutan": "Asia", "Laos": "Asia",
    "Mongolia": "Asia", "Taiwan": "Asia",

    "United States": "North America", "Canada": "North America",
    "Mexico": "North America", "Costa Rica": "North America",
    "Cuba": "North America", "Jamaica": "North America",
    "Panama": "North America", "Guatemala": "North America",

    "Brazil": "South America", "Argentina": "South America",
    "Peru": "South America", "Colombia": "South America",
    "Chile": "South America", "Ecuador": "South America",
    "Bolivia": "South America", "Uruguay": "South America",

    "France": "Europe", "Italy": "Europe", "Spain": "Europe",
    "Germany": "Europe", "United Kingdom": "Europe", "Greece": "Europe",
    "Portugal": "Europe", "Netherlands": "Europe", "Switzerland": "Europe",
    "Austria": "Europe", "Belgium": "Europe", "Sweden": "Europe",
    "Norway": "Europe", "Denmark": "Europe", "Finland": "Europe",
    "Ireland": "Europe", "Poland": "Europe", "Czechia": "Europe",
    "Hungary": "Europe", "Croatia": "Europe", "Romania": "Europe",
    "Russia": "Europe", "Iceland": "Europe", "Scotland": "Europe",

    "Egypt": "Africa", "South Africa": "Africa", "Morocco": "Africa",
    "Kenya": "Africa", "Tanzania": "Africa", "Nigeria": "Africa",
    "Ethiopia": "Africa", "Ghana": "Africa", "Tunisia": "Africa",
    "Madagascar": "Africa", "Namibia": "Africa", "Rwanda": "Africa",
    "Uganda": "Africa", "Zimbabwe": "Africa", "Botswana": "Africa",
    "Mauritius": "Africa", "Senegal": "Africa", "Mozambique": "Africa",

    "Australia": "Oceania", "New Zealand": "Oceania",
    "Fiji": "Oceania", "Papua New Guinea": "Oceania",
}

# ============================================================
# VALID RELATIONSHIP TYPES (Schema enforcement)
# ============================================================
VALID_PREDICATES = [
    "LOCATED_IN",        # Destination → City/State/Country
    "BELONGS_TO",        # City → State, State → Country
    "PART_OF",           # State → Country
    "ON_CONTINENT",      # Country → Continent
    "HAS_TYPE",          # Destination → PlaceType
    "EVOKES",            # Destination → Emotion
    "BEST_VISITED_IN",   # Destination → Season
    "SIMILAR_TO",        # Destination ↔ Destination
    "HAS_DESTINATION",   # Country/City → Destination
    "NEAR",              # Destination → Destination (geographic proximity)
]

# Valid node types
VALID_NODE_TYPES = [
    "Continent", "Country", "State", "City",
    "Destination", "PlaceType", "Emotion", "Season",
]

# ============================================================
# SYNTHETIC NAME PATTERNS (to detect fake data)
# ============================================================
SYNTHETIC_NAME_PATTERNS = [
    "Hidden", "Serene", "Mystic", "Secret", "Enchanted",
    "Unknown", "Magical", "Whispering", "Dreamy", "Lost",
    "Forgotten", "Ancient", "Eternal", "Crystal", "Golden",
    "Silver", "Shadow", "Phantom", "Legendary", "Celestial",
]

# Fuzzy matching threshold (0-100)
FUZZY_MATCH_THRESHOLD = 85

# ============================================================
# SEASONS
# ============================================================
VALID_SEASONS = ["Spring", "Summer", "Monsoon", "Autumn", "Winter", "Year-round"]

SEASON_ALIASES = {
    "spring": "Spring", "mar-may": "Spring", "march-may": "Spring",
    "summer": "Summer", "jun-aug": "Summer", "june-august": "Summer",
    "monsoon": "Monsoon", "rainy": "Monsoon", "jul-sep": "Monsoon",
    "autumn": "Autumn", "fall": "Autumn", "sep-nov": "Autumn",
    "oct-dec": "Autumn",
    "winter": "Winter", "dec-feb": "Winter", "november-february": "Winter",
    "oct-mar": "Winter", "october-march": "Winter",
    "all": "Year-round", "year-round": "Year-round",
    "all year": "Year-round", "any": "Year-round",
}
