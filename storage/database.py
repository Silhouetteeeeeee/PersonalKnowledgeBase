import sqlite3
import os

import sqlite_vec

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "knowledge.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS file_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_hash TEXT,
            extracted_text TEXT NOT NULL,
            knowledge_ids TEXT NOT NULL DEFAULT '[]',
            source_user_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
        CREATE TABLE IF NOT EXISTS error_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            wrong_answer TEXT NOT NULL,
            correct_answer TEXT DEFAULT '',
            category TEXT DEFAULT '',
            contradiction_details TEXT DEFAULT '',
            error_type TEXT DEFAULT 'unknown',
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS error_vectors USING vec0(
            embedding float[512] distance_metric=cosine
        );
        -- Wiki pages (index-only, content stored as markdown files)
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL UNIQUE,
            file_path TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]',
            sources TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'active',
            checksum TEXT DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS page_vectors USING vec0(
            embedding float[512] distance_metric=cosine
        );
        CREATE TABLE IF NOT EXISTS page_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            target_title TEXT NOT NULL,
            relation_type TEXT NOT NULL DEFAULT 'wikilink',
            FOREIGN KEY (source_id) REFERENCES pages(id)
        );
    """)
    conn.commit()
    conn.close()
    from memory.models import init_memory_tables
    init_memory_tables()
