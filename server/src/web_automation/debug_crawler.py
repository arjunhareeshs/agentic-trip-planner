"""
Custom AdaptiveCrawler that:
1. Injects screenshots, magic mode, virtual scrolling, LLM extraction per page
2. Generates query variations LOCALLY (no API needed)
3. Uses local sentence-transformers for all embeddings
4. Logs real-time navigation and saves screenshots
5. Makes LLM extraction gracefully optional — errors don't break the crawl
6. Properly preserves links for multi-page crawling (no head_data gating)
"""

import asyncio
import os
import json
import base64
import random
from typing import Optional, List, Tuple, Any

from crawl4ai import (
    AsyncWebCrawler,
    AdaptiveCrawler,
    CrawlerRunConfig,
    CacheMode,
    LLMExtractionStrategy,
)
from crawl4ai.async_configs import LinkPreviewConfig
from crawl4ai.models import Link, CrawlResult

from web_automation.scroll import VIRTUAL_SCROLL_JS


class DebugAdaptiveCrawler(AdaptiveCrawler):
    """
    Extends AdaptiveCrawler to:
    1. Use our full CrawlerRunConfig on every page (screenshots, magic, scroll)
    2. Optionally inject LLMExtractionStrategy for structured extraction
    3. Generate query variations LOCALLY without LLM API
    4. Print real-time navigation logs
    5. Save screenshots per page as they're captured
    6. KEEP ALL discovered links (no head_data gating) for multi-page crawling
    """

    def __init__(self, *args,
                 screenshot_dir: str = "./output/screenshots",
                 extraction_strategy: Optional[LLMExtractionStrategy] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.screenshot_dir = screenshot_dir
        self.extraction_strategy = extraction_strategy
        self._page_counter = 0
        self._extraction_successes = 0
        self._extraction_failures = 0
        os.makedirs(self.screenshot_dir, exist_ok=True)

        # Override the strategy's query expansion to be fully local
        self.strategy.map_query_semantic_space = self._local_query_expansion

    async def _local_query_expansion(self, query: str, n_synthetic: int = 10) -> Any:
        """
        Generate query variations LOCALLY without any LLM API call.
        Uses simple text manipulation + local sentence-transformers embeddings.
        """
        import numpy as np

        print(f"  📝 Generating {n_synthetic} query variations locally...")

        # Generate variations using text manipulation (no API needed)
        variations = self._generate_variations(query, n_synthetic)

        # Split into train and validation
        random.shuffle(variations)
        n_validation = max(2, int(len(variations) * 0.2))
        val_queries = variations[-n_validation:]
        train_queries = [query] + variations[:-n_validation]

        # Embed using LOCAL sentence-transformers
        train_embeddings = await self.strategy._get_embeddings(train_queries)

        # Store validation queries
        self.strategy._validation_queries = val_queries

        print(f"  ✅ Generated {len(train_queries)} training + {len(val_queries)} validation queries")
        for i, q in enumerate(train_queries[:5]):
            print(f"     [{i+1}] {q}")
        if len(train_queries) > 5:
            print(f"     ... and {len(train_queries) - 5} more")

        return train_embeddings, train_queries

    def _generate_variations(self, query: str, n: int) -> list:
        """Generate query variations using text manipulation patterns."""
        variations = []
        words = query.split()

        # Pattern 1: Rephrasings
        prefixes = [
            "find information about", "search for", "extract data about",
            "look for content related to", "get details on",
            "retrieve information on", "scrape content about",
            "collect data related to", "gather information about",
            "find all content about", "explore topics about",
            "discover content related to", "locate information on",
        ]
        for prefix in prefixes[:n]:
            core = " ".join(w for w in words if w.lower() not in
                          {"scrape", "extract", "find", "get", "search", "the", "all", "alone", "only", "contents", "content"})
            if core.strip():
                variations.append(f"{prefix} {core.strip()}")

        # Pattern 2: Aspect-focused queries
        aspects = [
            f"what are the main topics in {query}",
            f"key concepts related to {query}",
            f"detailed content about {query}",
            f"comprehensive guide on {query}",
            f"tutorial content for {query}",
            f"examples and explanations of {query}",
            f"fundamentals of {query}",
            f"advanced topics in {query}",
        ]
        variations.extend(aspects)

        # Pattern 3: Word reordering and subsets
        if len(words) > 2:
            for _ in range(min(5, n)):
                subset = random.sample(words, max(2, len(words) - 1))
                variations.append(" ".join(subset))

        # Deduplicate and limit
        seen = set()
        unique = []
        for v in variations:
            v_lower = v.lower().strip()
            if v_lower not in seen and v_lower != query.lower().strip():
                seen.add(v_lower)
                unique.append(v)

        # Ensure we have enough — pad with slight modifications
        while len(unique) < int(n * 1.3):
            base = random.choice(unique) if unique else query
            unique.append(f"{base} overview")

        return unique[:int(n * 1.3)]

    def _build_crawl_config(self, query: str) -> CrawlerRunConfig:
        """Build a full CrawlerRunConfig with all features + optional LLM extraction."""
        return CrawlerRunConfig(
            # LLM Extraction Strategy (optional — gracefully skipped if API down)
            extraction_strategy=self.extraction_strategy,

            # Screenshots
            screenshot=True,

            # Anti-detection & simulation
            magic=True,
            simulate_user=True,
            override_navigator=True,

            # Full page content
            scan_full_page=True,
            scroll_delay=0.8,
            max_scroll_steps=30,
            remove_overlay_elements=True,

            # Virtual scrolling JS
            js_code=VIRTUAL_SCROLL_JS,

            # Cache
            cache_mode=CacheMode.BYPASS,

            # Content
            word_count_threshold=30,

            # Link preview for adaptive crawling
            # Lower timeout, higher max_links for better link discovery
            link_preview_config=LinkPreviewConfig(
                include_internal=True,
                include_external=False,
                query=query,
                concurrency=10,
                timeout=3,
                max_links=100,
                verbose=False,
            ),
            score_links=True,

            # Verbose
            verbose=True,
        )

    def _is_extraction_error(self, content: str) -> bool:
        """Check if the extracted content is an LLM error response."""
        if not content:
            return False
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return any(item.get("error") for item in data if isinstance(item, dict))
            if isinstance(data, dict):
                return data.get("error", False)
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return False

    async def _crawl_with_preview(self, url: str, query: str) -> Optional[CrawlResult]:
        """Override: crawl with our full config + debug logging.

        KEY FIX: Do NOT filter out links without head_data.
        The original code removed all links that didn't have head_data,
        which killed multi-page crawling when link previews timed out.
        """
        self._page_counter += 1
        page_num = self._page_counter

        print(f"\n{'─'*60}")
        print(f"  🌐 [{page_num}] NAVIGATING TO: {url}")
        print(f"{'─'*60}")

        config = self._build_crawl_config(query)

        try:
            result = await self.crawler.arun(url=url, config=config)

            # Extract actual CrawlResult from container
            if hasattr(result, '_results') and result._results:
                result = result._results[0]

            if result and hasattr(result, 'success') and result.success:
                # Log content info
                content_len = 0
                if hasattr(result, 'markdown') and result.markdown:
                    if hasattr(result.markdown, 'raw_markdown'):
                        content_len = len(result.markdown.raw_markdown or "")
                    else:
                        content_len = len(str(result.markdown))

                # Count links — DO NOT FILTER by head_data
                total_internal = 0
                links_with_preview = 0
                if hasattr(result, 'links') and result.links:
                    if isinstance(result.links, dict):
                        internal_links = result.links.get('internal', [])
                        total_internal = len(internal_links)
                        links_with_preview = sum(
                            1 for link in internal_links
                            if isinstance(link, dict) and link.get('head_data')
                        )
                    else:
                        internal_links = getattr(result.links, 'internal', [])
                        total_internal = len(internal_links)

                print(f"  ✅ [{page_num}] SUCCESS: {url}")
                print(f"     📝 Content: {content_len:,} chars extracted")
                print(f"     🔗 Links found: {total_internal} internal ({links_with_preview} with preview)")

                # Check LLM extraction result — detect errors gracefully
                if hasattr(result, 'extracted_content') and result.extracted_content:
                    ext_content = str(result.extracted_content)
                    if self._is_extraction_error(ext_content):
                        # LLM extraction failed — log warning, clear the error content
                        self._extraction_failures += 1
                        print(f"     ⚠️  LLM Extraction FAILED (error from model)")
                        try:
                            err_data = json.loads(ext_content)
                            if isinstance(err_data, list) and err_data:
                                err_msg = err_data[0].get("content", "Unknown error")
                            elif isinstance(err_data, dict):
                                err_msg = err_data.get("content", "Unknown error")
                            else:
                                err_msg = ext_content[:150]
                            print(f"        Error: {err_msg[:150]}")
                        except Exception:
                            print(f"        Error: {ext_content[:150]}")
                        # Clear the error so it doesn't get saved as "extracted content"
                        result.extracted_content = None
                    else:
                        self._extraction_successes += 1
                        ext_len = len(ext_content)
                        print(f"     🧠 LLM Extraction: {ext_len:,} chars")
                        preview = ext_content[:200]
                        for line in preview.split('\n')[:3]:
                            print(f"        {line.strip()}")
                        if ext_len > 200:
                            print(f"        ... ({ext_len:,} total chars)")
                else:
                    print(f"     🧠 LLM Extraction: (skipped or unavailable)")

                # Brief pause so user can see the page in the browser
                await asyncio.sleep(1)

                # Save screenshot immediately
                if hasattr(result, 'screenshot') and result.screenshot:
                    self._save_screenshot(result.screenshot, url, page_num)
                else:
                    print(f"     📸 No screenshot captured")

                # ─────────────────────────────────────────────────────
                # KEY FIX: Do NOT filter links by head_data!
                # The old code did:
                #   result.links['internal'] = [l for l in ... if l.get('head_data')]
                # This killed ALL links when previews timed out, stopping multi-page crawl.
                # We keep ALL links and let the ranking strategy decide relevance.
                # ─────────────────────────────────────────────────────

            else:
                print(f"  ❌ [{page_num}] FAILED: {url}")

            return result

        except Exception as e:
            print(f"  ❌ [{page_num}] ERROR: {url}")
            print(f"     {type(e).__name__}: {e}")
            return None

    async def _crawl_batch(self, links_with_scores: List[Tuple[Link, float]], query: str) -> List[CrawlResult]:
        """Override: crawl batch sequentially so user can watch each navigation."""
        valid_results = []
        for link, score in links_with_scores:
            print(f"\n  📊 Link score: {score:.3f} → {link.href}")
            result = await self._crawl_with_preview(link.href, query)
            if result and isinstance(result, CrawlResult):
                if hasattr(result, 'success') and result.success:
                    valid_results.append(result)

        return valid_results

    def _save_screenshot(self, screenshot_data, url: str, page_num: int):
        """Save a screenshot to disk immediately."""
        safe_name = url.replace("https://", "").replace("http://", "")
        safe_name = safe_name.replace("/", "_").replace("?", "_").replace(":", "_")[:80]
        screenshot_path = os.path.join(self.screenshot_dir, f"{page_num}_{safe_name}.png")

        try:
            if isinstance(screenshot_data, str):
                if "base64," in screenshot_data:
                    screenshot_data = screenshot_data.split("base64,")[1]
                img_bytes = base64.b64decode(screenshot_data)
                with open(screenshot_path, "wb") as f:
                    f.write(img_bytes)
            elif isinstance(screenshot_data, bytes):
                with open(screenshot_path, "wb") as f:
                    f.write(screenshot_data)

            size_kb = os.path.getsize(screenshot_path) / 1024
            print(f"     📸 Screenshot saved: {screenshot_path} ({size_kb:.0f} KB)")
        except Exception as e:
            print(f"     ⚠️  Screenshot save failed: {e}")

    def get_extraction_stats(self) -> dict:
        """Return extraction success/failure counts."""
        return {
            "successes": self._extraction_successes,
            "failures": self._extraction_failures,
            "total_pages": self._page_counter,
        }
