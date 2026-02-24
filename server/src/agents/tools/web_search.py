"""Web search tool using DuckDuckGo (ddgs).

Free web search that works with any model (Ollama, LiteLLM, etc.)
— no API key required.
"""

from __future__ import annotations

from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using DuckDuckGo and return top results.

    Args:
        query: the search query string, e.g. "best hotels in Goa 2026".
        max_results: maximum number of results to return (default 5).

    Returns:
        A list of dicts, each with 'title', 'url', and 'snippet'.
    """
    try:
        results = DDGS(timeout=15).text(query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": f"Web search failed: {str(e)}"}]
