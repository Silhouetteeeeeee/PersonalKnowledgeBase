"""Comprehensive system test."""
import json, os, re, sqlite3, tempfile
from datetime import datetime

os.environ["OUTPUT_LANGUAGE"] = "zh"
import logging
logging.basicConfig(level=logging.WARNING)
for name in ["httpx", "httpcore", "urllib3", "fastembed", "jieba"]:
    logging.getLogger(name).setLevel(logging.ERROR)

from storage.database import init_db, get_connection
from storage.wiki_storage import read_page, WKI_DIR, parse_frontmatter
from storage.profile import load_profile
from agent.graph import build_graph
from server.bot import _process_and_store_file

REPORT, PASS, FAIL = [], 0, 0

def log(msg):
    REPORT.append(msg); print(msg)

def check(cond, desc):
    global PASS, FAIL
    if cond:
        log(f"  [OK] {desc}"); PASS += 1
    else:
        log(f"  [FAIL] {desc}"); FAIL += 1

init_db()

conn = get_connection()
existing = conn.execute("SELECT COUNT(*) FROM pages WHERE status='active'").fetchone()[0]
conn.close()
log(f"=== 测试开始：{existing} 个未谁面＝ ===")

graph = build_graph()

def ask(msg, sid="0"):
    log(f"\n\\    >>> Q: {msg}")
    start = datetime.now()
    r = graph.invoke(dict(user_message=msg, user_id="tester",
        session_id=f"gs_{sid}",
        timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        user_profile=load_profile()))
    ans = r.get("final_response","")
    wids = r.get("wiki_page_ids",[])
    sids = r.get("stored_knowledge_ids",[])
    logic = r.get("logic_chain",[])
    hs = any("store" in e.get("node","") or "Wiki" in e.get("action","") for e in logic)
    el = (datetime.now()-start).total_seconds()
    log(f"    time={el:.1f}s, ans={len(ans)}c, wiki={wids}, store={hs}")
    return dict(answer=ans, wiki_ids=wids, stored_ids=sids, logic=logic)