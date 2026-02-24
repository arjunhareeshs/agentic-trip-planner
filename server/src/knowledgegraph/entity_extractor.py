"""
Entity Extractor Module — Stage 2a of the KG Pipeline
Uses DeepSeek V3.1 671B via Ollama for entity extraction.
"""
import json
import re
import time
import requests
try:
    from .config import (
        OLLAMA_BASE_URL, OLLAMA_MODEL,
        LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_RETRY_ATTEMPTS, BATCH_SIZE,
        VALID_NODE_TYPES, VALID_PREDICATES, CONTINENT_MAP, DEFAULT_EMOTION_MAP,
    )
except ImportError:
    from config import (
        OLLAMA_BASE_URL, OLLAMA_MODEL,
        LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_RETRY_ATTEMPTS, BATCH_SIZE,
        VALID_NODE_TYPES, VALID_PREDICATES, CONTINENT_MAP, DEFAULT_EMOTION_MAP,
    )


def call_ollama_llm(prompt, system_prompt="", temperature=None):
    """Call DeepSeek V3.1 via Ollama's OpenAI-compatible endpoint with retry logic."""
    if temperature is None:
        temperature = LLM_TEMPERATURE

    # Ollama exposes OpenAI-compatible API at /v1/chat/completions
    url = f"{OLLAMA_BASE_URL}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": LLM_MAX_TOKENS,
    }

    for attempt in range(LLM_RETRY_ATTEMPTS):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=300,  # Longer timeout for local LLM
            )

            if response.status_code == 429:
                wait_time = (attempt + 1) * 5
                print(f"    ⏳ Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Strip <think>...</think> reasoning tags (DeepSeek)
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

            # Strip markdown code fences if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            return content

        except requests.exceptions.RequestException as e:
            print(f"    ❌ Ollama call failed (attempt {attempt + 1}/{LLM_RETRY_ATTEMPTS}): {e}")
            if attempt < LLM_RETRY_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
            else:
                raise

    return None


def parse_json_response(response_text):
    """Parse LLM response as JSON with error handling."""
    if not response_text:
        return None
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to find JSON object within the response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start >= 0 and end > start:
            try:
                return json.loads(response_text[start:end])
            except json.JSONDecodeError:
                pass
        # Try to find JSON array
        start = response_text.find('[')
        end = response_text.rfind(']') + 1
        if start >= 0 and end > start:
            try:
                return json.loads(response_text[start:end])
            except json.JSONDecodeError:
                pass
    print(f"    ❌ Failed to parse JSON. Response preview: {response_text[:200]}...")
    return None


ENTITY_SYSTEM_PROMPT = """You are a knowledge graph entity extractor for a tourism trip planner.
You extract entities and relationships from structured destination data.

RULES:
1. Output ONLY valid JSON — no explanations, no markdown.
2. Every entity must have: name, type, and properties.
3. Every relationship must be one of these types: LOCATED_IN, BELONGS_TO, PART_OF, ON_CONTINENT, HAS_TYPE, EVOKES, BEST_VISITED_IN, HAS_DESTINATION, SIMILAR_TO, NEAR
4. Valid entity types: Continent, Country, State, City, Destination, PlaceType, Emotion, Season
5. Do NOT invent facts not present in the input data.
6. Normalize all names to Title Case.
7. Use the EXACT country and city names from the input — do not change them."""


def build_extraction_prompt(destinations_batch):
    """Build prompt for entity/relationship extraction from a batch of destinations."""
    # Format the batch as a structured list
    data_text = "DESTINATION DATA:\n"
    for i, dest in enumerate(destinations_batch, 1):
        data_text += f"\n--- Destination {i} ---\n"
        for key, value in dest.items():
            if value and str(value).strip() and key not in ("entity_id", "is_synthetic", "source_file"):
                data_text += f"  {key}: {value}\n"

    prompt = f"""{data_text}

TASK: Extract ALL entities and relationships from the destinations above.

For each destination, create:
1. A Destination node
2. A Country node (if not already created)
3. A Continent node (if not already created)
4. A State node (if state info exists)
5. A City node (if city info exists)
6. A PlaceType node for the destination type
7. A Season node for best_season (if exists)
8. LOCATED_IN relationships connecting Destination → City → State → Country
9. ON_CONTINENT relationship for Country → Continent
10. HAS_TYPE relationship for Destination → PlaceType
11. BEST_VISITED_IN relationship for Destination → Season

OUTPUT FORMAT (valid JSON only):
{{
  "entities": [
    {{"name": "Taj Mahal", "type": "Destination", "properties": {{"rating": 4.5, "data_quality": "verified", "avg_cost_usd": 50}}}},
    {{"name": "India", "type": "Country", "properties": {{}}}},
    {{"name": "Monument", "type": "PlaceType", "properties": {{}}}}
  ],
  "relationships": [
    {{"source": "Taj Mahal", "type": "LOCATED_IN", "target": "Agra"}},
    {{"source": "Taj Mahal", "type": "HAS_TYPE", "target": "Monument"}},
    {{"source": "India", "type": "ON_CONTINENT", "target": "Asia"}}
  ]
}}"""
    return prompt


def extract_entities_from_batch(destinations_batch, batch_num, total_batches):
    """Extract entities and relationships from a batch using LLM."""
    print(f"\n  📤 Processing batch {batch_num}/{total_batches} ({len(destinations_batch)} destinations)...")

    prompt = build_extraction_prompt(destinations_batch)
    response = call_ollama_llm(prompt, system_prompt=ENTITY_SYSTEM_PROMPT)
    parsed = parse_json_response(response)

    if not parsed:
        print(f"    ❌ Batch {batch_num} failed — using fallback extraction")
        return fallback_extract(destinations_batch)

    entities = parsed.get("entities", [])
    relationships = parsed.get("relationships", [])
    print(f"    ✅ Extracted {len(entities)} entities, {len(relationships)} relationships")

    return entities, relationships


def fallback_extract(destinations_batch):
    """Deterministic fallback when LLM fails — extract directly from data."""
    entities = []
    relationships = []
    seen_entities = set()

    for dest in destinations_batch:
        name = dest.get("name", "")
        country = dest.get("country", "")
        continent = dest.get("continent", CONTINENT_MAP.get(country, "Unknown"))
        state = dest.get("state", "")
        city = dest.get("city", "")
        place_type = dest.get("type", "Other")
        best_season = dest.get("best_season", "")

        # Destination entity
        if name and f"Destination:{name}" not in seen_entities:
            properties = {
                "rating": dest.get("rating"),
                "data_quality": dest.get("data_quality", "unverified"),
            }
            if dest.get("avg_cost_usd"):
                properties["avg_cost_usd"] = dest["avg_cost_usd"]
            if dest.get("entrance_fee"):
                properties["entrance_fee"] = dest["entrance_fee"]
                properties["entrance_fee_currency"] = dest.get("entrance_fee_currency", "INR")
            if dest.get("visit_time_hrs"):
                properties["visit_time_hrs"] = dest["visit_time_hrs"]
            if dest.get("significance"):
                properties["significance"] = dest["significance"]
            if dest.get("description"):
                properties["description"] = dest["description"]

            entities.append({"name": name, "type": "Destination", "properties": properties})
            seen_entities.add(f"Destination:{name}")

        # Country
        if country and f"Country:{country}" not in seen_entities:
            entities.append({"name": country, "type": "Country", "properties": {}})
            seen_entities.add(f"Country:{country}")

        # Continent
        if continent and f"Continent:{continent}" not in seen_entities:
            entities.append({"name": continent, "type": "Continent", "properties": {}})
            seen_entities.add(f"Continent:{continent}")

        # State
        if state and f"State:{state}" not in seen_entities:
            entities.append({"name": state, "type": "State", "properties": {}})
            seen_entities.add(f"State:{state}")

        # City
        if city and f"City:{city}" not in seen_entities:
            entities.append({"name": city, "type": "City", "properties": {}})
            seen_entities.add(f"City:{city}")

        # PlaceType
        if place_type and f"PlaceType:{place_type}" not in seen_entities:
            entities.append({"name": place_type, "type": "PlaceType", "properties": {}})
            seen_entities.add(f"PlaceType:{place_type}")

        # Season
        if best_season and f"Season:{best_season}" not in seen_entities:
            entities.append({"name": best_season, "type": "Season", "properties": {}})
            seen_entities.add(f"Season:{best_season}")

        # Relationships
        if name and city:
            relationships.append({"source": name, "type": "LOCATED_IN", "target": city})
        if name and not city and country:
            relationships.append({"source": name, "type": "LOCATED_IN", "target": country})
        if city and state:
            relationships.append({"source": city, "type": "BELONGS_TO", "target": state})
        if state and country:
            relationships.append({"source": state, "type": "PART_OF", "target": country})
        if city and not state and country:
            relationships.append({"source": city, "type": "BELONGS_TO", "target": country})
        if country and continent:
            relationships.append({"source": country, "type": "ON_CONTINENT", "target": continent})
        if name and place_type:
            relationships.append({"source": name, "type": "HAS_TYPE", "target": place_type})
        if name and best_season:
            relationships.append({"source": name, "type": "BEST_VISITED_IN", "target": best_season})

    return entities, relationships


def extract_all_entities(destinations):
    """
    Stage 2a: Extract entities and relationships from all destinations.
    Processes in batches to stay within LLM context limits.
    """
    print("\n" + "=" * 60)
    print("🔍 STAGE 2a: ENTITY EXTRACTION (DeepSeek V3.1 via Ollama)")
    print("=" * 60)
    print(f"   Model: {OLLAMA_MODEL}")
    print(f"   Total destinations: {len(destinations)}")
    print(f"   Batch size: {BATCH_SIZE}")

    all_entities = []
    all_relationships = []

    # Process in batches
    total_batches = (len(destinations) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(destinations), BATCH_SIZE):
        batch = destinations[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        try:
            entities, relationships = extract_entities_from_batch(batch, batch_num, total_batches)
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        except Exception as e:
            print(f"    ❌ Batch {batch_num} error: {e}")
            print(f"    🔄 Using fallback extraction...")
            entities, relationships = fallback_extract(batch)
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        # Brief pause between batches
        if batch_num < total_batches:
            time.sleep(0.5)

    print(f"\n📊 ENTITY EXTRACTION COMPLETE:")
    print(f"   Total entities: {len(all_entities)}")
    print(f"   Total relationships: {len(all_relationships)}")

    return all_entities, all_relationships


if __name__ == "__main__":
    # Quick test
    test_destinations = [
        {
            "name": "Taj Mahal", "country": "India", "continent": "Asia",
            "state": "Uttar Pradesh", "city": "Agra", "type": "Monument",
            "rating": 4.5, "data_quality": "verified",
        }
    ]
    entities, rels = extract_all_entities(test_destinations)
    print(json.dumps({"entities": entities, "relationships": rels}, indent=2))
