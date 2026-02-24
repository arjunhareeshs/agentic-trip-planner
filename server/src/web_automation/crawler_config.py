from crawl4ai import CrawlerRunConfig, CacheMode
from web_automation.scroll import VIRTUAL_SCROLL_JS

def get_crawler_config() -> CrawlerRunConfig:
    """Returns the crawler run config for advanced behavior."""
    return CrawlerRunConfig(
        word_count_threshold=50,
        screenshot=True,
        cache_mode=CacheMode.BYPASS,
        wait_for=None,
        js_code=VIRTUAL_SCROLL_JS,
        remove_overlay_elements=True,
        scan_full_page=True,
        simulate_user=True,
        override_navigator=True,
        magic=True,
        scroll_delay=1.0,
        max_scroll_steps=50,
        verbose=True,
    )
