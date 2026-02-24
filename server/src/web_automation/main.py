import asyncio
import json
import os
from datetime import datetime
from crawl4ai import (
    AsyncWebCrawler,
    AdaptiveConfig,
)
from web_automation.config import get_browser_config, get_extraction_strategy
from web_automation.debug_crawler import DebugAdaptiveCrawler
from web_automation.utils import format_output


OUTPUT_DIR = "./output"


def ensure_output_dir():
    """Create output directory with timestamp subfolder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "screenshots"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "markdown"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "extracted"), exist_ok=True)
    return run_dir


def save_results(results, run_dir):
    """Save all extraction results to files."""

    summary = []
    for i, r in enumerate(results):
        entry = {
            "page": i + 1,
            "url": r["url"],
            "markdown_length": r.get("markdown_length", 0),
            "extracted_content_length": len(r.get("extracted_content", "")),
            "word_count": r.get("word_count", 0),
            "has_screenshot": r.get("screenshot") is not None,
            "has_extraction": bool(r.get("extracted_content")),
        }
        summary.append(entry)

        # Safe filename
        safe_name = r["url"].replace("https://", "").replace("http://", "")
        safe_name = safe_name.replace("/", "_").replace("?", "_").replace(":", "_")[:80]

        # Save individual markdown files
        md_path = os.path.join(run_dir, "markdown", f"{i+1}_{safe_name}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Page {i+1}: {r['url']}\n\n")
            f.write(r.get("markdown", "(markdown content not available in formatted output)"))

        # Save LLM extracted content (only if successful — errors are already filtered)
        if r.get("extracted_content"):
            ext_path = os.path.join(run_dir, "extracted", f"{i+1}_{safe_name}.txt")
            with open(ext_path, "w", encoding="utf-8") as f:
                f.write(f"# LLM Extraction — Page {i+1}: {r['url']}\n\n")
                f.write(r["extracted_content"])

    # Save summary JSON
    summary_path = os.path.join(run_dir, "results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save all markdown combined
    combined_path = os.path.join(run_dir, "all_content.md")
    with open(combined_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(results):
            f.write(f"\n{'='*60}\n")
            f.write(f"# Page {i+1}: {r['url']}\n")
            f.write(f"{'='*60}\n\n")
            f.write("## Raw Markdown\n\n")
            f.write(r.get("markdown", "(no content)"))
            f.write("\n\n")
            if r.get("extracted_content"):
                f.write("## 🧠 LLM Extracted Content\n\n")
                f.write(r["extracted_content"])
                f.write("\n\n")

    # Save all extractions combined
    extraction_path = os.path.join(run_dir, "all_extractions.md")
    with open(extraction_path, "w", encoding="utf-8") as f:
        f.write("# 🧠 LLM Extractions — All Pages\n\n")
        for i, r in enumerate(results):
            if r.get("extracted_content"):
                f.write(f"\n{'─'*60}\n")
                f.write(f"## Page {i+1}: {r['url']}\n")
                f.write(f"{'─'*60}\n\n")
                f.write(r["extracted_content"])
                f.write("\n\n")

    return summary_path, combined_path, extraction_path


async def run_adaptive_scraper(url, prompt):
    """Adaptive crawling with LLM-controlled link selection + structured extraction."""

    browser_config = get_browser_config()
    run_dir = ensure_output_dir()
    screenshot_dir = os.path.join(run_dir, "screenshots")

    # Create LLM extraction strategy for every page (local Ollama)
    extraction_strategy = get_extraction_strategy(prompt)

    # Read model name from env for display
    from web_automation.config import OLLAMA_MODEL

    print(f"📁 Output directory: {os.path.abspath(run_dir)}\n")

    # Create the base crawler with visible browser
    async with AsyncWebCrawler(config=browser_config) as crawler:

        # Adaptive config with STATISTICAL strategy
        adaptive_config = AdaptiveConfig(
            max_pages=20,
            max_depth=3,
            confidence_threshold=0.5,
            min_gain_threshold=0.1,
            strategy="statistical",  # ← Statistical-based link selection + confidence
        )

        # Use our custom debug crawler with LLM extraction
        adaptive = DebugAdaptiveCrawler(
            crawler=crawler,
            config=adaptive_config,
            screenshot_dir=screenshot_dir,
            extraction_strategy=extraction_strategy,
        )

        print(f"🚀 Starting adaptive crawl from: {url}")
        print(f"📝 Query: {prompt}")
        print(f"⚙️  Strategy: STATISTICAL")
        print(f"🧠 LLM: Ollama {OLLAMA_MODEL} (local)")
        print(f"{'='*60}\n")

        # Run adaptive digest
        start_time = datetime.now()
        state = await adaptive.digest(
            start_url=url,
            query=prompt,
        )
        elapsed = (datetime.now() - start_time).total_seconds()

        # Process results
        results = []
        for result in state.knowledge_base:
            output = format_output(result)
            results.append(output)

        # Get extraction stats
        ext_stats = adaptive.get_extraction_stats()

        # Summary
        print(f"\n{'='*60}")
        print(f"  🏁 ADAPTIVE CRAWL COMPLETE")
        print(f"{'='*60}")
        print(f"  ⏱️  Time elapsed: {elapsed:.1f}s")
        print(f"  📊 Total pages crawled: {len(state.crawled_urls)}")
        print(f"  🔍 Confidence: {adaptive.confidence:.2%}")
        print(f"  📈 Depth reached: {state.metrics.get('depth_reached', 0)}")
        print(f"  📸 Screenshots saved: {adaptive._page_counter}")
        pages_with_extraction = sum(1 for r in results if r.get("extracted_content"))
        print(f"  🧠 LLM extraction: {ext_stats['successes']} succeeded, {ext_stats['failures']} failed")
        print(f"  📄 Pages with content: {pages_with_extraction}/{len(results)}")
        total_words = sum(r.get("word_count", 0) for r in results)
        print(f"  📝 Total words extracted: {total_words:,}")
        print(f"{'='*60}\n")

        adaptive.print_stats()

        # Save results
        if results:
            summary_path, combined_path, extraction_path = save_results(results, run_dir)

            print(f"\n📄 Output files:")
            print(f"   📋 Summary:       {os.path.abspath(summary_path)}")
            print(f"   📝 Combined MD:   {os.path.abspath(combined_path)}")
            print(f"   🧠 Extractions:   {os.path.abspath(extraction_path)}")
            print(f"   📂 Markdown:      {os.path.abspath(os.path.join(run_dir, 'markdown'))}")
            print(f"   📂 Extracted:     {os.path.abspath(os.path.join(run_dir, 'extracted'))}")
            print(f"   📸 Screenshots:   {os.path.abspath(screenshot_dir)}")

            # Print URLs
            print(f"\n🗺️  Pages visited:")
            for i, r in enumerate(results):
                has_ss = "📸" if r.get("screenshot") else "  "
                has_llm = "🧠" if r.get("extracted_content") else "  "
                wc = r.get("word_count", 0)
                md_len = r.get("markdown_length", len(r.get("markdown", "")))
                print(f"   {has_ss}{has_llm} [{i+1}] {r['url']}  ({md_len:,} chars, {wc:,} words)")
        else:
            print("\n⚠️  No results extracted.")

        return results


if __name__ == "__main__":
    url = input("🌐 Enter URL: ").strip()
    prompt = input("📝 Enter extraction prompt: ").strip()

    print(f"\n🔥 Starting LLM-Controlled Adaptive Crawl...\n")
    results = asyncio.run(run_adaptive_scraper(url, prompt))
    print(f"\n✅ Done! Extracted data from {len(results)} pages.")
