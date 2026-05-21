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
        CREATE TABLE IF NOT EXISTS page_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id INTEGER NOT NULL,
            version INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            checksum TEXT NOT NULL,
            source_id TEXT DEFAULT '',
            source_question TEXT DEFAULT '',
            change_summary TEXT DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (page_id) REFERENCES pages(id),
            UNIQUE(page_id, version)
        );
        CREATE TABLE IF NOT EXISTS review_schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id INTEGER NOT NULL UNIQUE,
            easiness_factor REAL NOT NULL DEFAULT 2.5,
            interval_days INTEGER NOT NULL DEFAULT 1,
            repetitions INTEGER NOT NULL DEFAULT 0,
            next_review_at TEXT NOT NULL,
            last_reviewed_at TEXT DEFAULT '',
            last_quality INTEGER DEFAULT -1,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS sent_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schedule_id INTEGER NOT NULL,
            page_id INTEGER NOT NULL,
            marker_id TEXT NOT NULL UNIQUE,
            sent_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            status TEXT NOT NULL DEFAULT 'pending',
            FOREIGN KEY (schedule_id) REFERENCES review_schedule(id) ON DELETE CASCADE,
            FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE
        );
        DROP TABLE IF EXISTS source_questions;
        -- Fund bot tables
        CREATE TABLE IF NOT EXISTS user_portfolio (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            fund_code   TEXT NOT NULL,
            fund_name   TEXT,
            shares      REAL,
            cost_price  REAL,
            notes       TEXT,
            added_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            updated_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            UNIQUE(user_id, fund_code)
        );
        CREATE TABLE IF NOT EXISTS fund_info (
            code             TEXT PRIMARY KEY,
            name             TEXT,
            fund_type        TEXT,
            company          TEXT,
            established_date TEXT,
            fund_size        REAL,
            manager          TEXT,
            nav              REAL,
            total_nav        REAL,
            nav_date         TEXT,
            updated_at       TEXT
        );
        CREATE TABLE IF NOT EXISTS fund_nav_cache (
            fund_code    TEXT,
            date         TEXT,
            nav          REAL,
            total_nav    REAL,
            daily_return REAL,
            PRIMARY KEY (fund_code, date)
        );
        CREATE TABLE IF NOT EXISTS fund_holdings_cache (
            fund_code     TEXT,
            report_date   TEXT,
            holdings_json TEXT,
            sectors_json  TEXT,
            PRIMARY KEY (fund_code, report_date)
        );
        CREATE TABLE IF NOT EXISTS fund_decisions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT,
            fund_code       TEXT,
            decision_date   TEXT,
            rating          TEXT,
            decision_text   TEXT,
            nav_at_decision REAL,
            raw_return      REAL,
            alpha_return    REAL,
            reflection      TEXT,
            status          TEXT DEFAULT 'pending',
            created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            resolved_at     TEXT
        );
    """)
    conn.commit()
    conn.close()
    from memory.models import init_memory_tables
    init_memory_tables()
