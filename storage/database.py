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
        -- 用户基金持仓表
        CREATE TABLE IF NOT EXISTS user_portfolio (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,       -- 微信用户ID
            fund_code   TEXT NOT NULL,       -- 基金代码(6位数字)
            fund_name   TEXT,                -- 基金名称
            shares      REAL,                -- 持有份额
            cost_price  REAL,                -- 成本单价(单位净值)
            notes       TEXT,                -- 备注(如"定投账户"/"养老账户")
            added_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),  -- 添加时间
            updated_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),  -- 更新时间
            UNIQUE(user_id, fund_code)
        );
        -- 基金基本信息缓存(24h TTL)
        CREATE TABLE IF NOT EXISTS fund_info (
            code             TEXT PRIMARY KEY,   -- 基金代码
            name             TEXT,               -- 基金简称
            fund_type        TEXT,               -- 基金类型(股票型/混合型/债券型等)
            company          TEXT,               -- 基金管理人
            established_date TEXT,               -- 成立日期
            fund_size        REAL,               -- 份额规模(亿份)
            manager          TEXT,               -- 基金经理
            nav              REAL,               -- 最新单位净值
            total_nav        REAL,               -- 最新累计净值
            nav_date         TEXT,               -- 净值日期
            updated_at       TEXT                -- 缓存更新时间
        );
        -- 基金单位净值历史缓存(1h TTL)
        CREATE TABLE IF NOT EXISTS fund_nav_cache (
            fund_code    TEXT,       -- 基金代码
            date         TEXT,       -- 净值日期
            nav          REAL,       -- 单位净值
            total_nav    REAL,       -- 累计净值
            daily_return REAL,       -- 日涨跌幅(%)
            updated_at   TEXT,       -- 缓存更新时间
            PRIMARY KEY (fund_code, date)
        );
        -- 基金持仓快照缓存(7d TTL, 按季披露)
        CREATE TABLE IF NOT EXISTS fund_holdings_cache (
            fund_code     TEXT,       -- 基金代码
            report_date   TEXT,       -- 报告期(如"2024")
            holdings_json TEXT,       -- 前十大持仓JSON
            sectors_json  TEXT,       -- 行业分布JSON
            updated_at    TEXT,       -- 缓存更新时间
            PRIMARY KEY (fund_code, report_date)
        );
        -- 基金决策日志(TradingAgents风格延迟反思)
        CREATE TABLE IF NOT EXISTS fund_decisions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT,           -- 微信用户ID
            fund_code       TEXT,           -- 基金代码
            decision_date   TEXT,           -- 决策日期
            rating          TEXT,           -- 评级(Strong Buy/Buy/Hold/Reduce/Sell)
            decision_text   TEXT,           -- 完整决策分析文本
            nav_at_decision REAL,           -- 决策时的单位净值
            raw_return      REAL,           -- 实际收益率(延迟填充, Phase B)
            alpha_return    REAL,           -- 超额收益率(延迟填充, Phase B)
            reflection      TEXT,           -- 决策反思(延迟填充, Phase B)
            status          TEXT DEFAULT 'pending',  -- pending(待反思) / resolved(已反思)
            created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),  -- 创建时间
            resolved_at     TEXT            -- 反思时间
        );
    """)
    conn.commit()
    conn.close()
    from memory.models import init_memory_tables
    init_memory_tables()
