"""
agents/orchestrator.py — Architecture reference.

The root agent (trip_planner_orchestrator) is defined in agent.py.
This file documents the full multi-agent orchestration architecture.

═══════════════════════════════════════════════════════════════
SYSTEM ARCHITECTURE
═══════════════════════════════════════════════════════════════

  User
   │
   ▼
 trip_planner_orchestrator   (agent.py — Google ADK LlmAgent)
   │  Model: REASONING_MODEL (deepseek-v3.1:671b-cloud)
   │  Direct tools: geocode, get_weather_forecast,
   │                match_destinations, filter_destinations,
   │                get_destination_details, list_all_destinations,
   │                get_graph_stats, web_search
   │
   ├──► preference_agent         subagents/preference_agent.py
   │      Model: DEFAULT_MODEL
   │      Trigger: vibe + budget + duration collected
   │      Tools: match_destinations, filter_destinations,
   │             get_destination_details, list_all_destinations,
   │             get_graph_stats, web_search
   │      Output key: preference_results
   │
   ├──► itinerary_agent          subagents/itinerary_agent.py
   │      Model: REASONING_MODEL
   │      Trigger: destination + duration + budget + travel mode confirmed
   │      Tools: geocode, get_weather_forecast, search_places,
   │             get_place_details, get_route, web_search
   │      Output key: itinerary_result
   │
   ├──► booking_agent            subagents/booking_agent.py
   │      Model: REASONING_MODEL
   │      Trigger: user explicitly asks to book
   │      Tools: search_flights, geocode, search_places,
   │             get_place_details, web_search
   │      Output key: booking_result
   │
   ├──► data_visualization_agent subagents/data_visualization_agent.py
   │      Model: DEFAULT_MODEL
   │      Trigger: user requests charts or visual comparisons
   │      Tools: BuiltInCodeExecutor (matplotlib)
   │      Output key: visualization_result
   │
   └──► image_analysis_agent     subagents/image_analysis_agent.py
          Model: llava:latest (VLM)
          Trigger: user shares a travel image
          Tools: web_search
          Output key: image_analysis_result

═══════════════════════════════════════════════════════════════
DATA SOURCES
═══════════════════════════════════════════════════════════════

  Knowledge Graph (tools/kg_output/graph_state.json)
    600+ destinations across 50+ countries
    Built from: Top Indian Places to Visit.csv,
                Tourist_Destinations.csv,
                Expanded_Destinations.csv,
                Final_Updated_Expanded_Reviews.csv
    Node types: Continent, Country, State, City, Destination,
                PlaceType, Emotion, Season
    Emotions: peaceful, thrilling, romantic, spiritual,
              nostalgic, joyful, awe, curious, relaxed

  Live APIs
    Geoapify     — geocoding, venue search, routing
    OpenWeather  — weather forecasts (global)
    AviationStack — flight data (worldwide IATA)
    DuckDuckGo   — web search (no API key required)

  RAG Pipeline (src/RAG/)
    Dual-modal text + vision RAG (Phase 2 — not yet connected)

═══════════════════════════════════════════════════════════════
CONVERSATION FLOW
═══════════════════════════════════════════════════════════════

  Turn 1  Greeting  → ask about travel mood/vibe
  Turn 2  Companions + budget tier
  Turn 3  Duration + region preference
  Turn 4  Constraints → delegate to preference_agent
  Turn 5  Present destinations → user picks one
  Turn 6  Confirm details → delegate to itinerary_agent
  Turn 7  Present itinerary → offer booking
  Turn 8  Delegate to booking_agent if requested

═══════════════════════════════════════════════════════════════
HOW TO RUN
═══════════════════════════════════════════════════════════════

  cd "c:\\agentic ai\\agent-trip-planner\\server\\src"
  adk web
  → Select "agents" from the dropdown

"""
