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
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            parent_id INTEGER REFERENCES categories(id),
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS knowledge_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            knowledge_text TEXT NOT NULL,
            source_question TEXT NOT NULL,
            category TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'active',
            corrected_text TEXT DEFAULT '',
            reasoning_log_path TEXT DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vectors USING vec0(
            embedding float[512] distance_metric=cosine
        );
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
    """)
    # Schema migration for existing databases
    try:
        conn.execute("ALTER TABLE knowledge_points ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE knowledge_points ADD COLUMN corrected_text TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE knowledge_points ADD COLUMN reasoning_log_path TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()
    from memory.models import init_memory_tables
    init_memory_tables()
