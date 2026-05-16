"""Deep pipeline test: run multi-turn queries, inspect wiki/DB, report findings."""
import json, os, re, sqlite3
from datetime import datetime

os.environ["OUTPUT_LANGUAGE"] = "zh"  # ensure Chinese output

import logging
logging.basicConfig(level=logging.WARNING)
for name in ["httpx", "httpcore", "urllib3", "fastembed", "jieba"]:
    logging.getLogger(name).setLevel(logging.ERROR)

from storage.database import init_db, get_connection
from storage.models import get_all_pages_index, find_similar_pages, get_page_by_title
from storage.wiki_storage import read_page, WIKI_DIR, write_page, parse_frontmatter
from storage.wiki_storage import extract_wikilinks
from storage.profile import load_profile
from agent.graph import build_graph

REPORT = []
def log(msg):
    REPORT.append(msg)
    print(msg)

init_db()

# ── Phase 0: Snapshot current state ──
existing = get_all_pages_index()
log(f"=== Phase 0: {len(existing)} pages before test ===")

# ── Phase 1: Test queries ──
queries = [
    # Q1: Follow-up to existing topic (Redis was created last session)
    ("Redis的RDB和AOF各有什么优缺点", "follow-up to Redis持久化"),
    # Q2: New topic (should create new pages unrelated to existing)
    ("Java中HashMap和Hashtable的区别是什么", "new topic"),
    # Q3: Narrow follow-up (should update existing page)
    ("请详细解释一下GIL在多线程编程中的实际影响", "follow-up to GIL"),
]

graph = build_graph()
for i, (q, reason) in enumerate(queries):
    log(f"\n{'='*60}")
    log(f">>> Q{i+1}: {q}")
    log(f"    type: {reason}")

    start = datetime.now()
    result = graph.invoke({
        "user_message": q,
        "user_id": "deep_tester",
        "session_id": str(900 + i),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "user_profile": load_profile(),
    })
    elapsed = (datetime.now() - start).total_seconds()

    answer = result.get("final_response", "")
    wiki_ids = result.get("wiki_page_ids", [])
    stored_ids = result.get("stored_knowledge_ids", [])
    logic = result.get("logic_chain", [])

    log(f"    time: {elapsed:.1f}s, answer: {len(answer)} chars")
    log(f"    wiki_page_ids: {wiki_ids}")
    log(f"    stored_knowledge_ids: {stored_ids}")
    for entry in logic:
        if "store" in entry.get("node", "") or "Wiki" in entry.get("action", ""):
            log(f"    store action: {entry.get('action', '')}")

# ── Phase 2: Full inspection ──
log(f"\n{'='*60}")
log("PHASE 2: COMPREHENSIVE INSPECTION")
log(f"{'='*60}")

conn = get_connection()
conn.row_factory = sqlite3.Row

# 2a. List ALL pages with full metadata
log("\n--- All pages in DB ---")
all_pages = conn.execute("SELECT id, title, file_path, sources, tags, status FROM pages WHERE status='active' ORDER BY id").fetchall()
for p in all_pages:
    sources = json.loads(p["sources"]) if p["sources"] else []
    tags = json.loads(p["tags"]) if p["tags"] else []
    log(f"  [{p['id']:2d}] {p['title']:30s} | tags={tags[:3]} | sources={sources[:2]}")

# 2b. Check new pages (ids >= 28) in detail
log("\n--- New pages detail (id >= 28) ---")
new_pages = [p for p in all_pages if p["id"] >= 28]
if not new_pages:
    log("  (no new pages created)")
else:
    for p in new_pages:
        fp = p["file_path"]
        full_path = os.path.join(WIKI_DIR, fp)
        body = ""
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                body = f.read()
        meta, content = parse_frontmatter(body)

        log(f"\n  [{p['id']}] {meta.get('title', '?')}")
        log(f"        file: {fp} ({len(body)} chars)")
        log(f"        created: {meta.get('created', 'MISSING!')}")
        log(f"        updated: {meta.get('updated', 'MISSING!')}")
        log(f"        sources: {meta.get('sources', [])}")
        log(f"        tags: {meta.get('tags', [])}")
        # Check cross-references
        wikilinks = extract_wikilinks(content)
        if wikilinks:
            log(f"        wikilinks: {wikilinks[:5]}")
        else:
            log(f"        wikilinks: (none)")

# 2c. Dedup check
log("\n--- Dedup check: new pages overlap with existing ---")
for p in new_pages:
    similar = find_similar_pages(p["title"], threshold=0.5, limit=5)
    for s in similar:
        if s["id"] != p["id"] and s.get("distance", 1) < 0.5:
            log(f"  {p['title']} ~ {s['title']} (id={s['id']}, dist={s.get('distance',0):.2f})")

# 2d. Sources integrity check
log("\n--- Sources integrity ---")
for p in all_pages:
    sources = json.loads(p["sources"]) if p["sources"] else []
    if len(sources) > 2:
        log(f"  [WARN] [{p['id']}] {p['title']}: {len(sources)} sources: {sources}")
    file_sources = [s for s in sources if s.startswith("file_") or s.endswith(".pdf")]
    if len(file_sources) > 1:
        log(f"  [WARN] [{p['id']}] {p['title']}: duplicate file sources: {file_sources}")

# 2e. File consistency
log("\n--- File path consistency ---")
for p in all_pages:
    fp = p["file_path"]
    full_path = os.path.join(WIKI_DIR, fp)
    exists = os.path.exists(full_path)
    if not exists:
        log(f"  [WARN] [{p['id']}] {p['title']}: file MISSING at {full_path}")
    else:
        fsize = os.path.getsize(full_path)
        if fsize < 30:
            log(f"  [WARN] [{p['id']}] {p['title']}: too small ({fsize} bytes)")

# 2f. Cross-page content overlap within new pages
log("\n--- Content overlap among new pages ---")
for p1 in new_pages:
    for p2 in new_pages:
        if p1["id"] >= p2["id"]:
            continue
        content1 = (p1["title"] or "")
        similar = find_similar_pages(content1, threshold=0.2, limit=10)
        for s in similar:
            if s["id"] == p2["id"]:
                log(f"  {p1['title']} ~ {p2['title']} (dist={s.get('distance',0):.2f})")

conn.close()
log(f"\n{'='*60}")
log("END OF REPORT")
log(f"{'='*60}")

# Save report
with open("deep_test_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(REPORT))
print("\nReport saved to deep_test_report.txt")
