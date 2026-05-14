"""Wiki page extraction: two-step CoT (analyze -> generate).

Step 1 (Analyze): LLM reads Q&A + index + SCHEMA, outputs analysis
  (topics, actions, related pages, contradictions).

Step 2 (Generate): LLM reads analysis + existing pages, outputs wiki page content.
  System writes to disk, updates SQLite index, rebuilds index.md.
"""

import logging
import os
from datetime import datetime

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.database import get_connection
from storage.models import (
    upsert_page,
    update_page_relations,
    get_page_by_title,
    find_similar_pages,
    save_page_version,
)
from storage.wiki_storage import (
    ensure_dirs,
    title_to_filename,
    read_schema,
    read_page,
    write_page,
    build_frontmatter,
    extract_wikilinks,
    compute_checksum,
)
from storage.wiki_index import rebuild_index, get_index_for_prompt

logger = logging.getLogger(__name__)


# ── Pydantic models for Step 1 (Analysis) ──

class AnalysisAction(BaseModel):
    topic: str = Field(description="Topic extracted from the Q&A")
    action: str = Field(description="'create' for new page, 'update' for existing")
    target: str = Field(description="Existing page title to update, or empty string for new")


class AnalysisOutput(BaseModel):
    topics: list[str] = Field(description="Topics covered in this Q&A")
    actions: list[AnalysisAction] = Field(
        description="Per-topic actions: create new page or update existing"
    )
    related_pages: list[str] = Field(
        description="Titles of existing pages related to this content"
    )
    contradictions: list[str] = Field(
        description="Contradictions between new content and existing knowledge"
    )


# ── Pydantic model for Step 2 (Generation) ──

class WikiPageOutput(BaseModel):
    title: str = Field(description="Page title")
    content: str = Field(
        description="Full page markdown content (body only, no frontmatter)"
    )
    tags: list[str] = Field(description="Tags for this page")
    sources: list[str] = Field(description="Source conversation IDs")


class WikiBatchOutput(BaseModel):
    pages: list[WikiPageOutput] = Field(
        description="All wiki pages to create or update (one per topic action)"
    )


# ── Prompt templates ──

def _find_similar_existing_pages(source_text: str, max_pages: int = 5) -> list[dict]:
    """Find semantically similar existing pages to help prevent duplicates."""
    similar = find_similar_pages(source_text, threshold=0.5, limit=max_pages)
    result = []
    for p in similar:
        file_page = read_page(p["file_path"])
        if file_page:
            body = file_page["body"]
            preview = body[:200] + "..." if len(body) > 200 else body
            result.append({
                "title": p["title"],
                "body_preview": preview,
                "distance": p.get("distance", 0),
            })
    return result


def _build_analysis_prompt(
    source_text: str,
    source_label: str,
    similar_pages: list[dict] | None = None,
) -> str:
    schema_content = read_schema()
    page_index = get_index_for_prompt()

    similar_text = ""
    if similar_pages:
        similar_text = "## Similar Existing Pages\n\n"
        similar_text += (
            "The following pages may already cover topics from the source content. "
            "If the topic already exists, prefer to UPDATE the existing page "
            "rather than creating a duplicate.\n\n"
        )
        for sp in similar_pages:
            similar_text += f"### {sp['title']}\n\n{sp['body_preview']}\n\n---\n\n"

    return (
        f"{schema_content}\n\n"
        f"## Current Wiki Page Index\n\n"
        f"{page_index}\n\n"
        f"{similar_text}"
        f"## Source Content to Analyze\n\n"
        f"Context: {source_label}\n\n"
        f"{source_text}\n\n"
        f"## Analysis Requirements\n\n"
        f"1. Identify all topics covered in this content\n"
        f"2. For each topic, decide whether to create a new page or update an existing one\n"
        f"3. List existing pages related to this content\n"
        f"4. If the new content contradicts existing knowledge, note it\n\n"
        f"Note: Different topics may need different actions. "
        f"For example, one Q&A might update an existing 'Django' page "
        f"while creating a new 'ORM Optimization' page.\n"
        f"Important: If the source content covers the same topic as an existing page, "
        f"always choose 'update' rather than creating a duplicate."
    )


def _build_generation_prompt(
    analysis: AnalysisOutput,
    source_text: str,
    source_label: str,
    existing_page_contents: list[dict],
) -> str:
    schema_content = read_schema()

    existing_text = ""
    if existing_page_contents:
        existing_text = "## Existing Page Content (for update)\n\n"
        for p in existing_page_contents:
            existing_text += f"### Page: {p['title']}\n\n"
            if p.get("file_path"):
                existing_text += f"Current path: {p['file_path']}\n\n"
            existing_text += f"{p.get('body', '')}\n\n---\n\n"

    actions_text_lines = []
    for a in analysis.actions:
        if a.action == "create":
            actions_text_lines.append(f"- {a.topic}: Create new page")
        else:
            actions_text_lines.append(f'- {a.topic}: Update "{a.target}"')
    actions_text = "\n".join(actions_text_lines)

    return (
        f"{schema_content}\n\n"
        f"## Analysis Report\n\n"
        f"Topics: {', '.join(analysis.topics)}\n"
        f"Actions:\n{actions_text}\n"
        f"Related pages: {', '.join(analysis.related_pages) if analysis.related_pages else 'None'}\n"
        f"Contradictions: {', '.join(analysis.contradictions) if analysis.contradictions else 'None found'}\n\n"
        f"{existing_text}"
        f"## Original Content\n\n"
        f"Context: {source_label}\n\n"
        f"{source_text}\n\n"
        f"## Generation Requirements\n\n"
        f"Based on the analysis above, generate wiki page content:\n"
        f"1. Content field must contain only the body (NO frontmatter)\n"
        f"2. Use Chinese for explanations, keep English for technical terms\n"
        f"3. Cross-reference other pages using [[page title]] syntax\n"
        f"4. tags should be content labels, not categories\n"
        f"5. If updating an existing page, output the COMPLETE updated content (not just the diff)"
    )


def _get_source_id() -> str:
    """Generate a source conversation ID from timestamp."""
    return f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _read_existing_pages(actions: list[AnalysisAction]) -> list[dict]:
    """Read full content of pages that need updating."""
    existing = []
    for a in actions:
        if a.action != "update" or not a.target:
            continue
        page = get_page_by_title(a.target)
        if page:
            file_page = read_page(page["file_path"])
            if file_page:
                existing.append({
                    "title": page["title"],
                    "file_path": page["file_path"],
                    "body": file_page["body"],
                })
    return existing


def extract_to_wiki(
    source_text: str,
    source_id: str,
    source_label: str,
) -> dict:
    """Two-step CoT: analyze -> generate -> write -> index.

    Args:
        source_text: Full text to analyze (answer text or file text).
        source_id: Unique identifier for this extraction.
        source_label: Short descriptor for prompt context.

    Returns:
        dict with "page_ids" (list[int]) and "logic_chain" (list[dict]).
    """
    ensure_dirs()

    # ── Step 0: Find similar existing pages (dedup) ──
    similar_pages = _find_similar_existing_pages(source_text)
    if similar_pages:
        logger.info("Found %d similar existing pages for dedup context: %s",
                    len(similar_pages), [s["title"] for s in similar_pages])

    # ── Step 1: Analysis ──
    logger.info("Step 1: Analyzing content for wiki extraction...")
    analysis_prompt = _build_analysis_prompt(source_text, source_label, similar_pages)
    analysis = LLM.generate_structured(analysis_prompt, AnalysisOutput, use_language=False)
    if analysis is None:
        logger.error("Analysis LLM returned None")
        return {}
    logger.info("Analysis complete: %d topics, %d actions",
                len(analysis.topics), len(analysis.actions))

    # ── Step 2: Generation ──
    existing_contents = _read_existing_pages(analysis.actions)
    logger.info("Step 2: Generating wiki page(s)...")
    gen_prompt = _build_generation_prompt(analysis, source_text, source_label, existing_contents)
    batch = LLM.generate_structured(gen_prompt, WikiBatchOutput, use_language=False)
    if batch is None or not batch.pages:
        logger.error("Generation LLM returned None or empty pages")
        return {}

    # ── Write to filesystem + update SQLite ──
    now = datetime.now().strftime("%Y-%m-%d")
    saved_ids = []

    for wp in batch.pages:
        tags = wp.tags
        filename = title_to_filename(wp.title)
        file_path = os.path.join("pages", filename)

        existing_page_data = read_page(file_path)
        created_str = now
        if existing_page_data and existing_page_data.get("created"):
            created_str = existing_page_data["created"]

        # Override sources: use real source_id, merge with existing on update
        sources = [source_id]
        if existing_page_data:
            old_sources = existing_page_data.get("sources", [])
            if isinstance(old_sources, list):
                for s in old_sources:
                    if s not in sources:
                        sources.append(s)

        frontmatter = build_frontmatter(
            title=wp.title,
            tags=tags,
            sources=sources,
            created=created_str,
            updated=now,
        )
        full_content = frontmatter + "\n\n" + wp.content.strip()

        # Save current disk content as version before overwriting
        if existing_page_data:
            disk_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "wiki", file_path,
            )
            if os.path.exists(disk_path):
                with open(disk_path, encoding="utf-8") as _f:
                    old_raw = _f.read()
                _pid_row = get_connection().execute(
                    "SELECT id FROM pages WHERE title = ?", (wp.title,)
                ).fetchone()
                if _pid_row:
                    save_page_version(
                        page_id=_pid_row["id"],
                        title=wp.title,
                        content=old_raw,
                        checksum=compute_checksum(old_raw),
                        source_id=source_id,
                        source_question=source_label.replace("Question: ", ""),
                    )
                get_connection().close()

        checksum = write_page(file_path, full_content)

        pid = upsert_page(
            title=wp.title,
            file_path=file_path,
            tags=tags,
            sources=sources,
            checksum=checksum,
            content=full_content,
        )
        saved_ids.append(pid)

        links = extract_wikilinks(full_content)
        if links:
            update_page_relations(pid, links)

    rebuild_index()
    logger.info("Stored %d wiki pages", len(saved_ids))

    return {
        "page_ids": saved_ids,
        "logic_chain": [{
            "node": "store",
            "action": f"Wiki: stored {len(saved_ids)} pages",
            "reasoning": (
                f"Pages: {[wp.title for wp in batch.pages]}, "
                f"Actions: {[a.action for a in analysis.actions]}"
            ),
        }],
    }


def _fast_skip_check(answer: str) -> tuple[bool, str]:
    """Fast-path heuristic: skip LLM extraction if answer doesn't warrant a wiki page.

    Uses lightweight rules + vector DB query (no LLM calls).
    Returns (should_skip, reason).
    """
    if len(answer) < 30:
        return True, "answer too short (<30 chars)"

    if len(answer) < 200:
        similar = find_similar_pages(answer, threshold=0.75, limit=3)
        if similar and similar[0].get("distance", 1) >= 0.82:
            return True, (
                f"already covered by '{similar[0]['title']}' "
                f"(dist={similar[0].get('distance', 0):.2f})"
            )

    return False, ""


def store(state: dict) -> dict:
    """Two-step CoT extraction: analyze -> generate -> write."""
    if not state.get("needs_store", True):
        logger.info("Skipping store: needs_store=False")
        return {}

    if not state.get("answer"):
        logger.info("Skipping store: no answer")
        return {}

    if state.get("contradiction_found"):
        logger.info("Skipping store: contradiction detected")
        return {}

    # Fast-path: skip LLM for trivial or already-covered answers
    should_skip, skip_reason = _fast_skip_check(state["answer"])
    if should_skip:
        logger.info("Fast-path skip store: %s", skip_reason)
        return {}

    source_id = _get_source_id()
    source_label = f"Question: {state['user_message']}"
    result = extract_to_wiki(state["answer"], source_id, source_label)

    if not result.get("page_ids"):
        return {}

    return {
        "stored_knowledge_ids": result["page_ids"],
        "wiki_page_ids": result["page_ids"],
        "logic_chain": result.get("logic_chain", []),
    }
