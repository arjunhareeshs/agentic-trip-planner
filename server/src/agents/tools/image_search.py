"""Image search tool using DuckDuckGo image search.

Returns real image URLs for any travel destination or place query so the
front-end can render inline reference images — exactly like ChatGPT's image
references in travel responses.

No API key required.
"""

from __future__ import annotations

from ddgs import DDGS


def search_place_images(
    query: str,
    budget: str | None = None,
    max_results: int = 3,
) -> list[dict]:
    """Search for travel place images and return direct image URLs.

    Use this tool whenever the user asks about a destination, shares a photo
    and wants to see similar places, or requests visual references for any
    travel location. Returns direct image URLs that can be embedded inline
    in Markdown as  ![alt text](image_url).

    IMPORTANT: After receiving results, embed each image in your response using:
      ![title](image_url)

    Args:
        query: descriptive place search query, e.g.
               "Santorini Greece blue dome sunset",
               "Machu Picchu aerial view morning mist",
               "Kyoto Japan cherry blossom temples".
               Be specific — include city/country + a visual keyword.
        budget: optional budget hint to refine query context, e.g.
               "budget", "mid-range", "luxury". Appended to query
               automatically when provided.
        max_results: number of image results to return (default 3, max 6).

    Returns:
        A list of dicts, each containing:
          - title       (str)  : descriptive image title
          - image_url   (str)  : direct URL to the full image (embed-ready)
          - source_url  (str)  : webpage where the image was found
    """
    # Build the actual search string
    search_str = query.strip()
    if budget:
        search_str = f"{search_str} {budget} travel"

    try:
        results = DDGS(timeout=15).images(
            keywords=search_str,
            max_results=min(max_results, 6),
        )
        output = []
        for r in results:
            img_url = r.get("image", "")
            if not img_url:
                continue
            title   = r.get("title", search_str)
            src_url = r.get("url", "")
            output.append({
                "title":       title,
                "image_url":   img_url,
                "source_url":  src_url,
            })
        if not output:
            return [{"error": f"No images found for: {search_str}"}]
        return output

    except Exception as e:
        return [{"error": f"Image search failed: {str(e)}"}]


def format_image_gallery(
    images: list[dict],
    heading: str = "Visual References",
    max_show: int = 4,
) -> str:
    """Convert search_place_images results into a Markdown image gallery.

    Args:
        images:   list returned by search_place_images.
        heading:  optional section heading text.
        max_show: maximum images to include in the gallery (default 4).

    Returns:
        A Markdown string with inline images and source links.
    """
    if not images or "error" in images[0]:
        return ""

    lines = [f"### {heading}"]
    for img in images[:max_show]:
        img_url = img.get("image_url", "")
        title   = img.get("title", "Travel photo")
        src_url = img.get("source_url", "")
        if not img_url:
            continue
        # Inline image with clickable source link
        lines.append(f"![{title}]({img_url})")
        if src_url:
            lines.append(f"*[{title}]({src_url})*")
        lines.append("")  # blank line between images
    return "\n".join(lines)
