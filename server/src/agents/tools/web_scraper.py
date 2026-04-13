import asyncio
import concurrent.futures
import subprocess
import sys
from typing import Dict, Any

from web_automation.scraper import scrape_and_extract
from web_automation.main import run_adaptive_scraper

# Thread pool for running async scraper in a fresh event loop
_scraper_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _run_async_in_thread(coro):
    """Run an async coroutine in a new event loop inside a thread.
    
    On Windows we need ProactorEventLoop for subprocess support (Playwright
    launches a Chromium subprocess). We create a fresh ProactorEventLoop in
    a thread to avoid conflicting with uvicorn's event loop.
    """
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _fast_scrape(url: str) -> dict:
    """Scrape a page and return markdown only (no LLM extraction)."""
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
    from web_automation.config import get_browser_config

    browser_config = get_browser_config()
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        scan_full_page=True,
        remove_overlay_elements=True,
        simulate_user=True,
        override_navigator=True,
        magic=True,
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=crawler_config)
        md = ""
        if hasattr(result, "markdown") and result.markdown:
            md = getattr(result.markdown, "raw_markdown", None) or str(result.markdown)
        return {"url": url, "markdown": md}


def scrape_page(url: str) -> str:
    """
    Quickly scrape a webpage and return its text content as markdown.
    This is FAST — it just fetches the page in a visible Chromium browser and
    converts to markdown. No slow LLM extraction step. Use when you want to
    read a page yourself and pick out the relevant information.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        str: The page content as markdown text (truncated to ~8000 chars).
    """
    try:
        future = _scraper_pool.submit(_run_async_in_thread, _fast_scrape(url))
        result = future.result(timeout=60)
        md = result.get("markdown", "")
        if md:
            return md[:8000]
        return f"Page at {url} returned no readable content."
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


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
        if result.get("extracted_content"):
            return result["extracted_content"]
        # Fallback: return raw markdown content (truncated) so the LLM still gets data
        md = result.get("markdown", "")
        if md:
            truncated = md[:6000]
            return f"[Raw page content from {url} — LLM extraction skipped]\n\n{truncated}"
        return f"Page at {url} returned no usable content."
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


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
