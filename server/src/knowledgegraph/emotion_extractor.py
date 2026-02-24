"""
Emotion Extractor Module — Stage 2b of the KG Pipeline
Uses DeepSeek V3.1 via Ollama to extract Tier 3 emotions.
Also applies rule-based emotion mapping from place types and significance.
"""
import json
try:
    from .config import (
        EMOTION_TAXONOMY, DEFAULT_EMOTION_MAP,
        BATCH_SIZE, OLLAMA_MODEL,
    )
    from .entity_extractor import call_ollama_llm, parse_json_response
except ImportError:
    from config import (
        EMOTION_TAXONOMY, DEFAULT_EMOTION_MAP,
        BATCH_SIZE, OLLAMA_MODEL,
    )
    from entity_extractor import call_ollama_llm, parse_json_response


def validate_emotion(emotion):
    """Only allow emotions from the fixed 9-emotion taxonomy."""
    if not isinstance(emotion, str):
        return None
    emotion = emotion.lower().strip()
    if emotion in EMOTION_TAXONOMY:
        return emotion
    # Fuzzy map common synonyms
    SYNONYM_MAP = {
        "calm": "peaceful", "serene": "peaceful", "tranquil": "peaceful",
        "meditative": "peaceful", "quiet": "peaceful", "soothing": "peaceful",
        "exciting": "thrilling", "adventurous": "thrilling", "exhilarating": "thrilling",
        "adrenaline": "thrilling", "daring": "thrilling", "wild": "thrilling",
        "love": "romantic", "intimate": "romantic", "passionate": "romantic",
        "dreamy": "romantic", "couple": "romantic",
        "divine": "spiritual", "sacred": "spiritual", "holy": "spiritual",
        "religious": "spiritual", "blessed": "spiritual", "prayerful": "spiritual",
        "historic": "nostalgic", "historical": "nostalgic", "ancient": "nostalgic",
        "heritage": "nostalgic", "old-world": "nostalgic", "timeless": "nostalgic",
        "happy": "joyful", "fun": "joyful", "cheerful": "joyful",
        "pleasant": "joyful", "delightful": "joyful", "playful": "joyful",
        "family-friendly": "joyful", "festive": "joyful",
        "breathtaking": "awe", "stunning": "awe", "magnificent": "awe",
        "grand": "awe", "majestic": "awe", "spectacular": "awe",
        "impressive": "awe", "overwhelming": "awe", "wonder": "awe",
        "interesting": "curious", "educational": "curious", "fascinating": "curious",
        "exploratory": "curious", "learning": "curious", "discovery": "curious",
        "informative": "curious",
        "chill": "relaxed", "laid-back": "relaxed", "leisurely": "relaxed",
        "vacation": "relaxed", "unwinding": "relaxed", "comfortable": "relaxed",
        "restful": "relaxed", "casual": "relaxed",
    }
    if emotion in SYNONYM_MAP:
        return SYNONYM_MAP[emotion]
    # Try partial match
    for synonym, canonical in SYNONYM_MAP.items():
        if synonym in emotion or emotion in synonym:
            return canonical
    return None


def get_emotions_from_type(place_type):
    """Rule-based emotion mapping from place type."""
    if not place_type:
        return ["curious"]
    key = place_type.lower().strip()
    # Direct match
    if key in DEFAULT_EMOTION_MAP:
        return DEFAULT_EMOTION_MAP[key]
    # Partial match
    for type_key, emotions in DEFAULT_EMOTION_MAP.items():
        if type_key in key or key in type_key:
            return emotions
    return DEFAULT_EMOTION_MAP.get("other", ["curious"])


def get_emotions_from_significance(significance):
    """Map significance column to emotions (Indian Places dataset)."""
    if not significance or not isinstance(significance, str):
        return []
    sig = significance.lower().strip()
    mapping = {
        "historical": ["nostalgic", "curious"],
        "religious": ["spiritual", "peaceful"],
        "scenic": ["awe", "peaceful", "relaxed"],
        "cultural": ["curious", "joyful"],
        "architectural": ["awe", "nostalgic"],
        "natural": ["peaceful", "awe"],
        "spiritual": ["spiritual", "peaceful"],
        "adventure": ["thrilling"],
        "educational": ["curious"],
        "recreational": ["joyful", "relaxed"],
    }
    for key, emotions in mapping.items():
        if key in sig:
            return emotions
    return []


EMOTION_SYSTEM_PROMPT = """You are an emotion classifier for a tourism knowledge graph.
Given review text or destination descriptions, extract the emotions a visitor would feel.

RULES:
1. Output ONLY valid JSON — no explanations, no markdown.
2. You must ONLY use emotions from this fixed list:
   peaceful, thrilling, romantic, spiritual, nostalgic, joyful, awe, curious, relaxed
3. For each review/description, return 1-3 most relevant emotions.
4. Base your classification on the actual text content, not assumptions.
5. If the text is too vague, return ["curious"] as default."""


def extract_emotions_from_reviews(reviews, batch_size=20):
    """Use LLM to extract emotions from review texts."""
    print("\n" + "=" * 60)
    print("🎭 STAGE 2b: EMOTION EXTRACTION (DeepSeek V3.1 via Ollama)")
    print("=" * 60)
    print(f"   Model: {OLLAMA_MODEL}")
    print(f"   Total reviews: {len(reviews)}")

    # Group reviews by destination_id
    dest_reviews = {}
    for review in reviews:
        did = str(review.get("destination_id", ""))
        if did not in dest_reviews:
            dest_reviews[did] = []
        dest_reviews[did].append(review["review_text"])

    print(f"   Unique destinations with reviews: {len(dest_reviews)}")

    destination_emotions = {}
    dest_ids = list(dest_reviews.keys())
    total_batches = (len(dest_ids) + batch_size - 1) // batch_size

    for i in range(0, len(dest_ids), batch_size):
        batch_ids = dest_ids[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"\n  📤 Processing review batch {batch_num}/{total_batches}...")

        # Build prompt with sample reviews for each destination
        review_data = ""
        for did in batch_ids:
            texts = dest_reviews[did][:3]  # Max 3 reviews per destination
            combined = " | ".join(texts[:200] for texts in texts)  # Truncate
            review_data += f'\nDestination ID {did}: "{combined}"\n'

        prompt = f"""REVIEW TEXTS:
{review_data}

For each Destination ID, classify the emotions a visitor would feel based on the review text.
Use ONLY these emotions: peaceful, thrilling, romantic, spiritual, nostalgic, joyful, awe, curious, relaxed

OUTPUT FORMAT (valid JSON only):
{{
  "destination_emotions": {{
    "1": ["awe", "romantic"],
    "2": ["peaceful", "spiritual"]
  }}
}}"""

        try:
            response = call_ollama_llm(prompt, system_prompt=EMOTION_SYSTEM_PROMPT)
            parsed = parse_json_response(response)

            if parsed and "destination_emotions" in parsed:
                for did, emotions in parsed["destination_emotions"].items():
                    validated = []
                    for emo in emotions:
                        valid = validate_emotion(emo)
                        if valid:
                            validated.append(valid)
                    if validated:
                        destination_emotions[str(did)] = list(set(validated))
                print(f"    ✅ Extracted emotions for {len(parsed['destination_emotions'])} destinations")
            else:
                print(f"    ⚠️  Batch {batch_num}: invalid response, using fallback")
        except Exception as e:
            print(f"    ❌ Batch {batch_num} error: {e}")

    print(f"\n📊 EMOTION EXTRACTION COMPLETE:")
    print(f"   Destinations with emotions: {len(destination_emotions)}")

    return destination_emotions


def assign_emotions_to_destinations(destinations, review_emotions):
    """
    Assign emotions to each destination using multi-source approach:
    1. Review-based emotions (highest priority)
    2. Significance-based emotions
    3. Type-based default emotions (fallback)
    """
    print("\n  🎯 Assigning emotions to destinations...")

    results = []
    for dest in destinations:
        dest_id = str(dest.get("entity_id", ""))
        name = dest.get("name", "")
        place_type = dest.get("type", "Other")
        significance = dest.get("significance", "")

        # Source 1: Review-based (highest priority)
        emotions = review_emotions.get(dest_id, [])
        emotion_source = "reviews" if emotions else ""

        # Source 2: Significance-based
        if not emotions and significance:
            emotions = get_emotions_from_significance(significance)
            emotion_source = "significance" if emotions else ""

        # Source 3: Type-based default (fallback)
        if not emotions:
            emotions = get_emotions_from_type(place_type)
            emotion_source = "type_default"

        # Validate all emotions
        validated = []
        for emo in emotions:
            valid = validate_emotion(emo)
            if valid:
                validated.append(valid)

        if not validated:
            validated = ["curious"]
            emotion_source = "fallback"

        # Ensure at least 2 emotions per destination
        if len(validated) < 2:
            type_defaults = get_emotions_from_type(place_type)
            for emo in type_defaults:
                valid = validate_emotion(emo)
                if valid and valid not in validated:
                    validated.append(valid)
                if len(validated) >= 2:
                    break
            # Ultimate fallback
            if len(validated) < 2 and "curious" not in validated:
                validated.append("curious")

        results.append({
            "destination_name": name,
            "emotions": list(set(validated)),
            "emotion_source": emotion_source,
        })

    # Summary
    sources = {}
    for r in results:
        src = r["emotion_source"]
        sources[src] = sources.get(src, 0) + 1

    print(f"   Emotion source breakdown:")
    for src, count in sorted(sources.items()):
        print(f"     {src}: {count} destinations")

    return results


if __name__ == "__main__":
    # Quick test
    test_reviews = [
        {"destination_id": "1", "review_text": "Absolutely breathtaking view! The palace was magnificent and romantic."},
        {"destination_id": "1", "review_text": "Such a peaceful place, felt very spiritual."},
        {"destination_id": "2", "review_text": "Great adventure! Thrilling rafting experience."},
    ]
    emotions = extract_emotions_from_reviews(test_reviews)
    print(json.dumps(emotions, indent=2))
