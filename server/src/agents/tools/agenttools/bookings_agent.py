"""Bookings Agent — AgentTool wrapper.

A specialised LlmAgent that searches flights (AviationStack), finds venues
(Foursquare), and uses Google Search for booking links.  Wrapped as
AgentTool for the booking sub-agent.
"""

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool, google_search

from agents.tools.api_connectors.aviationstack import search_flights
from agents.tools.api_connectors.foursquare import search_venues, get_venue_details

# ── Inner agent ─────────────────────────────────────────────────────────────
_bookings_impl = LlmAgent(
    name="bookings_agent",
    model=os.getenv("REASONING_MODEL", "deepseek-v3.1:671b-cloud"),
    instruction="""You are a travel bookings specialist.

Your job is to find flights, hotels, and booking information.

When asked:
1. **Flights**: Use search_flights() with IATA codes.
   Common Indian airports: DEL (Delhi), BOM (Mumbai), MAA (Chennai),
   BLR (Bangalore), CCU (Kolkata), GOI (Goa), HYD (Hyderabad),
   IXC (Chandigarh), JAI (Jaipur), COK (Kochi), IXZ (Port Blair).

2. **Hotels/Venues**: Use search_venues() to find accommodation nearby,
   then get_venue_details() for ratings, tips, hours, and pricing.

3. **Booking Links**: Use google_search to find direct booking URLs
   on platforms like MakeMyTrip, Goibibo, Booking.com, OYO, IRCTC, Redbus.

Always return:
- Multiple options at different price points
- Direct booking links where available
- Key details (timing, price, rating)
- Cancellation / refund policy notes if available
""",
    tools=[search_flights, search_venues, get_venue_details, google_search],
)

# ── Exported AgentTool ──────────────────────────────────────────────────────
bookings_agent_tool = AgentTool(agent=_bookings_impl)
