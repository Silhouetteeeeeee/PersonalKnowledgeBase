import os, sys, tempfile, json
sys.path.insert(0, 'D:/Programming/LangChain-Learning')

# Setup temp paths
tmp = tempfile.mkdtemp()
db_dir = os.path.join(tmp, "data")
wiki_dir = os.path.join(tmp, "wiki")
pages_dir = os.path.join(wiki_dir, "pages")
os.makedirs(db_dir)
os.makedirs(pages_dir)

# Patch DB
import storage.database
storage.database.DB_DIR = db_dir
storage.database.DB_PATH = os.path.join(db_dir, "knowledge.db")
storage.database.init_db()

# Create page file
page_file = os.path.join(pages_dir, "test.md")
with open(page_file, "w", encoding="utf-8") as f:
    f.write("---\ntitle: Test Page\ntags: []\nsources: []\ncreated: 2026-05-13\nupdated: 2026-05-13\n---\n\nTest content body")
print("Page file exists:", os.path.exists(page_file))

# Insert DB record
from storage.database import get_connection
conn = get_connection()
conn.execute(
    "INSERT INTO pages (id, title, file_path, tags, sources, created_at, updated_at) "
    "VALUES (1, 'Test Page', 'pages/test.md', '[]', '[]', datetime('now','localtime','-1 day'), datetime('now','localtime','-1 day'))"
)
conn.commit()
conn.close()

# Now test _get_yesterday_pages with WIKI_DIR patched
print("Patching WIKI_DIR to:", wiki_dir)
import storage.wiki_storage
storage.wiki_storage.WIKI_DIR = wiki_dir

import server.daily_summary
print("daily_summary.WIKI_DIR:", repr(server.daily_summary.WIKI_DIR))

try:
    pages = server.daily_summary._get_yesterday_pages()
    print("Pages:", pages)
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
