import asyncio

from web_automation.main import run_adaptive_scraper

def test_adaptive():
    url = "https://example.com"
    prompt = "Extract main info."
    print("Testing adaptive scraper logic...")
    results = asyncio.run(run_adaptive_scraper(url, prompt))
    print(f"Results len: {len(results)}")

if __name__ == "__main__":
    test_adaptive()
