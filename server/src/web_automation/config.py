import os
from crawl4ai import BrowserConfig, LLMConfig, LLMExtractionStrategy
from dotenv import load_dotenv

load_dotenv()

# Read from .env
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", os.getenv("REASONING_MODEL", "deepseek-v3.1:671b-cloud"))

def get_browser_config() -> BrowserConfig:
    """Returns the browser configuration for crawl4ai."""
    return BrowserConfig(
        headless=False,                # Live visible browser for real-time scraping
        browser_type="chromium",
        enable_stealth=True,           # Stealth mode
        verbose=True,
        extra_args=[
            "--disable-blink-features=AutomationControlled",
            "--start-maximized",
        ],
    )

def get_llm_config() -> LLMConfig:
    """Returns an LLMConfig pointing to local Ollama."""
    model = os.getenv("OLLAMA_MODEL", os.getenv("REASONING_MODEL", "deepseek-v3.1:671b-cloud"))
    return LLMConfig(
        provider=f"ollama/{model}",
        api_token="ollama",  # Ollama doesn't need a real key
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        max_tokens=8000,
    )

def get_extraction_strategy(prompt: str) -> LLMExtractionStrategy:
    """
    Returns a configured LLMExtractionStrategy using local Ollama
    to extract structured data from every crawled page.
    """
    return LLMExtractionStrategy(
        llm_config=get_llm_config(),
        instruction=prompt,
        chunk_token_threshold=2048,
        overlap_rate=0.1,
        input_format="markdown",
        verbose=True,
    )
