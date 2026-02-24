import asyncio
from crawl4ai import AsyncWebCrawler
from web_automation.config import get_browser_config, get_extraction_strategy
from web_automation.crawler_config import get_crawler_config
from web_automation.utils import format_output

async def scrape_and_extract(url: str, prompt: str) -> dict:
    """
    Scrape a webpage and use an LLM (via local Ollama) to extract 
    structured data according to the given prompt. Uses infinite scroll and lazy loading config.
    """
    browser_config = get_browser_config()
    extraction_strategy = get_extraction_strategy(prompt)
    crawler_config = get_crawler_config()
    
    # Inject the LLM strategy into the crawler config
    crawler_config.extraction_strategy = extraction_strategy

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=crawler_config,
        )

        return format_output(result)

if __name__ == "__main__":
    # Test script for the scraper
    url = input("🌐 Enter URL: ").strip()
    prompt = input("📝 Enter extraction prompt: ").strip()

    print(f"\n🔥 Starting LLM Web Scraper...\n")
    result = asyncio.run(scrape_and_extract(url, prompt))
    print(f"\n✅ Result: \n{result}")
