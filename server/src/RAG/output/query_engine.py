"""
query_engine.py — Interactive REPL for querying the RAG pipeline.

After PDF ingestion, starts an interactive prompt where users can
type natural-language queries and see retrieved text chunks, image
paths, and relevance scores.

USAGE (called by test_metrics.py --pdf --interactive, or directly):
    from RAG.output.query_engine import InteractiveQueryEngine
    engine = InteractiveQueryEngine(pipeline)
    engine.run()

Commands inside the REPL:
    <any text>         → run RAG query
    :quit / :q / exit  → exit the REPL
    :health            → print model health check
    :help              → show available commands
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline import RAGPipeline


class InteractiveQueryEngine:
    """
    Interactive question-answering loop over an initialized RAGPipeline.

    Args:
        pipeline: A fully initialized RAGPipeline with data already ingested.
    """

    def __init__(self, pipeline: "RAGPipeline"):
        self._pipeline = pipeline

    def run(self) -> None:
        """Start the interactive query REPL. Blocks until user exits."""
        self._print_banner()

        while True:
            try:
                user_input = input("\n  [QUERY] >>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!\n")
                break

            if not user_input:
                continue

            # Handle commands
            lower = user_input.lower()
            if lower in (":quit", ":q", "exit", "quit"):
                print("  Goodbye!\n")
                break
            if lower == ":help":
                self._print_help()
                continue
            if lower == ":health":
                self._print_health()
                continue

            # Run RAG query
            self._execute_query(user_input)

    # ── Private helpers ───────────────────────────────────────

    def _execute_query(self, query_text: str) -> None:
        """Execute a single query and print formatted results."""
        result = self._pipeline.query(query_text)

        if not result.success:
            print(f"\n  ERROR: {result.error}")
            return

        context = result.data  # RAGContext

        print(f"  {'-'*60}")
        print(f"  QUERY:  {query_text}")
        print(f"  {'-'*60}")

        # Text results
        text_nodes = context.retrieved_text_nodes
        if text_nodes:
            print(f"\n  -- Text Results ({len(text_nodes)}) --\n")
            for i, node in enumerate(text_nodes, 1):
                meta = node.metadata or {}
                source = meta.get("source_pdf", "unknown")
                page = meta.get("page_number", "?")
                score = f"{node.similarity_score:.4f}" if node.similarity_score else "N/A"
                etype = meta.get("element_type", "")

                print(f"  [{i}] Score: {score}  |  Page: {page}  |  Type: {etype}")
                print(f"      Source: {source}")
                # Truncate long content for readability
                content = node.content.strip()
                wrapped = textwrap.fill(
                    content[:500] + ("..." if len(content) > 500 else ""),
                    width=72, initial_indent="      ", subsequent_indent="      ",
                )
                print(wrapped)
                print()
        else:
            print("\n  No text results found.")

        # Image results
        image_nodes = context.retrieved_image_nodes
        if image_nodes:
            print(f"  -- Image Results ({len(image_nodes)}) --\n")
            for i, node in enumerate(image_nodes, 1):
                meta = node.metadata or {}
                page = meta.get("page_number", "?")
                caption = meta.get("caption", node.content)[:120]
                img_path = meta.get("image_path", "")
                score = f"{node.similarity_score:.4f}" if node.similarity_score else "N/A"

                print(f"  [{i}] Score: {score}  |  Page: {page}")
                if caption:
                    print(f"      Caption: {caption}")
                if img_path:
                    print(f"      Path:    {img_path}")
                print()

        # Image paths summary
        if context.image_paths:
            print(f"  -- Image Paths --")
            for p in context.image_paths:
                print(f"    {p}")
            print()

        # Assembled prompt preview
        prompt = context.assembled_prompt
        if prompt:
            preview = prompt[:300] + ("..." if len(prompt) > 300 else "")
            print(f"  -- Assembled Prompt (preview, ~{context.token_count} tokens) --")
            print(textwrap.fill(preview, width=72, initial_indent="    ", subsequent_indent="    "))
            print()

        print(f"  {'-'*60}")

    def _print_health(self) -> None:
        """Print model health check status."""
        health = self._pipeline.health_check()
        print(f"\n  -- Model Health Check --")
        for key, val in health.items():
            status = "OK" if val else "FAIL"
            print(f"    {key:<30} {status}")
        print()

    @staticmethod
    def _print_banner() -> None:
        print(f"\n{'='*60}")
        print("  RAG Interactive Query Engine")
        print(f"{'='*60}")
        print("  Type a question about your documents.")
        print("  Commands:  :help  :health  :quit")
        print(f"{'='*60}")

    @staticmethod
    def _print_help() -> None:
        print(textwrap.dedent("""
          -- Commands ----------------------------
          <any text>    Ask a question about your PDFs
          :health       Show model health check
          :help         Show this help
          :quit / :q    Exit the query engine
        """))
