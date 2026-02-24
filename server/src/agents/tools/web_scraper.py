import asyncio
import concurrent.futures
import sys
from typing import Dict, Any

from web_automation.scraper import scrape_and_extract
from web_automation.main import run_adaptive_scraper

# Thread pool for running async scraper in a fresh event loop
# (Playwright requires SelectorEventLoop on Windows, which conflicts with
# uvicorn's ProactorEventLoop)
_scraper_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _run_async_in_thread(coro):
    """Run an async coroutine in a new SelectorEventLoop inside a thread.
    
    On Windows, Playwright needs SelectorEventLoop (not ProactorEventLoop).
    We force SelectorEventLoop here so the browser can actually launch.
    """
    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def extract_web_content(url: str, prompt: str) -> str:
    """
    Scrape a webpage and use a local LLM to extract directly structured data according to the prompt.
    For example, reviews, offers, specific content, or insights.

    Args:
        url (str): The URL of the webpage to scrape.
        prompt (str): Detailed instruction to the LLM on what to extract from the page.

    Returns:
        str: The extracted content formatted organically, or JSON as requested by the prompt.
    """
    try:
        future = _scraper_pool.submit(
            _run_async_in_thread, scrape_and_extract(url, prompt)
        )
        result = future.result(timeout=120)
        if "extracted_content" in result and result["extracted_content"]:
            return result["extracted_content"]
        else:
            return f"Failed to extract content or no content matched the prompt from {url}."
    except Exception as e:
        return f"Error occurred during web scraping and extraction: {str(e)}"


def deep_web_scrape(url: str, prompt: str) -> str:
    """
    Advanced adaptive scraper. Give it a URL and a strategic prompt (like "Find reviews" or "Search for booking pages").
    It navigates deep into the site by analyzing links locally and pulling the relevant data across multiple pages.
    Returns highly detailed summaries.
    """
    try:
        future = _scraper_pool.submit(
            _run_async_in_thread, run_adaptive_scraper(url, prompt)
        )
        results = future.result(timeout=180)
        if not results:
            return f"Failed to extract any content across multiple pages from {url}."

        # Combine the results
        output = [f"Found {len(results)} pages matching the data."]
        for i, res in enumerate(results):
            if res.get("extracted_content"):
                output.append(f"\n--- Page {i+1}: {res['url']} ---\n{res['extracted_content']}")

        return "\n".join(output)
    except Exception as e:
        return f"Error occurred during advanced adaptive web scraping: {str(e)}"
