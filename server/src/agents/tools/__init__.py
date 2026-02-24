# Tools package
#
# ============================================================
# TOOL MANIFEST
# ============================================================
# All callable tools available across agents in this system.
#
# KNOWLEDGE GRAPH TOOLS  (server/src/agents/tools/knowledge_graph.py)
#   match_destinations(keywords: list[str]) -> list[dict]
#       Keyword search across 600+ worldwide destinations.
#       Returns up to 10 results with score, emotions, types, seasons, location.
#
#   filter_destinations(emotion=None, place_type=None, season=None,
#                       city=None, state=None, country=None) -> list[dict]
#       Precise multi-criteria AND-filter. All args optional strings.
#       Supports any country in the knowledge graph (50+ countries).
#       Returns up to 20 results.
#
#   get_destination_details(destination_name: str) -> dict
#       Full destination profile: description, significance, entrance_fee,
#       visit_time, emotions, place_types, seasons, full location hierarchy.
#
#   list_all_destinations() -> list[dict]
#       Lightweight catalog of every destination in the KG.
#
#   get_graph_stats() -> dict
#       Available emotions, place_types, seasons, countries in the KG.
#       Call first to map user preferences to valid KG facets.
#
# LIVE-DATA TOOLS
#   geocode(place_name: str) -> dict               [geoapify.py]
#       Place name -> lat/lng. Works for any city worldwide.
#
#   get_weather_forecast(lat, lng, days=5) -> list  [openweather.py]
#       Daily weather forecast for any lat/lng on earth.
#
#   search_places(query, lat, lng, radius_m, limit) -> list[dict]  [geoapify.py]
#       Venue/POI search (hotels, restaurants, attractions) worldwide.
#
#   get_place_details(place_id: str) -> dict        [geoapify.py]
#       Full details for a Geoapify place_id.
#
#   get_route(origin_lat, origin_lng, dest_lat, dest_lng, mode) -> dict  [geoapify.py]
#       Distance and travel time. mode: drive|walk|bicycle|transit.
#
#   search_flights(departure_iata, arrival_iata, date="") -> list[dict]  [aviationstack.py]
#       Live flight search between two airports by IATA code.
#
#   web_search(query: str, max_results=5) -> list[dict]  [web_search.py]
#       DuckDuckGo search. Returns title, url, snippet.
#       Last-resort fallback and real-time data retrieval.
#
#   search_place_images(query, budget=None, max_results=6) -> list[dict]  [image_search.py]
#       DuckDuckGo IMAGE search. Returns direct image URLs for any destination.
#       Each result: {title, image_url, thumbnail, source_url, width, height}
#       Embed results inline: ![alt](image_url)
#       Use whenever a place is mentioned or user wants visual references.
#
#   extract_web_content(url: str, prompt: str) -> str    [web_scraper.py]
#       Scrapes a webpage using a visible Chromium browser and extracts structured 
#       data locally via Ollama 120b using LLMExtractionStrategy. 
#       Use to extract reviews, exact offers, and specific content from URLs.
#
#   deep_web_scrape(url: str, prompt: str) -> str        [web_scraper.py]
#       Advanced adaptive multi-page crawler. Provide a URL and a prompt
#       (e.g., "Find bookings or reviews"). Navigates deep into a site via local
#       embedding-based link selection and extracts data.
#
# ============================================================
# AGENT → TOOL ASSIGNMENT
# ============================================================
#
#   trip_planner_orchestrator  match_destinations, filter_destinations,
#                              get_destination_details, list_all_destinations,
#                              get_graph_stats, geocode, get_weather_forecast,
#                              web_search, search_place_images
#
#   preference_agent           match_destinations, filter_destinations,
#                              get_destination_details, list_all_destinations,
#                              get_graph_stats, web_search
#
#   itinerary_agent            geocode, get_weather_forecast, search_places,
#                              get_place_details, get_route, web_search
#
#   booking_agent              search_flights, geocode, search_places,
#                              get_place_details, web_search
#
#   image_analysis_agent       web_search, search_place_images
#
#   web_automation_agent       web_search, extract_web_content, deep_web_scrape
# ============================================================
