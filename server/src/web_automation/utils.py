import json

def _is_extraction_error(content: str) -> bool:
    """Check if extracted content is an LLM error response."""
    if not content:
        return True
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return any(item.get("error") for item in data if isinstance(item, dict))
        if isinstance(data, dict):
            return data.get("error", False)
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return False

def format_output(result) -> dict:
    """Format a CrawlResult into a clean dict with both raw and LLM content."""
    markdown = ""
    if hasattr(result, "markdown") and result.markdown:
        if hasattr(result.markdown, "raw_markdown"):
            markdown = result.markdown.raw_markdown or ""
        else:
            markdown = str(result.markdown)

    extracted = ""
    if hasattr(result, "extracted_content") and result.extracted_content:
        raw_extracted = str(result.extracted_content)
        if _is_extraction_error(raw_extracted):
            extracted = ""
        else:
            extracted = raw_extracted

    word_count = len(markdown.split()) if markdown else 0
    url = getattr(result, "url", "")

    return {
        "url": url,
        "markdown": markdown,
        "markdown_length": len(markdown),
        "word_count": word_count,
        "extracted_content": extracted,
    }
