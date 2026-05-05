"""Tests for file processing module."""

import os

import pytest
from storage.database import init_db, get_connection


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    init_db()


def test_extract_text_from_txt(tmp_path):
    from storage.file_processor import extract_text_from_file

    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world\nline 2", encoding="utf-8")
    text = extract_text_from_file(str(file_path))
    assert text == "hello world\nline 2"


def test_extract_text_from_txt_gbk(tmp_path):
    from storage.file_processor import extract_text_from_file

    file_path = tmp_path / "test_gbk.txt"
    content = "中文测试内容"
    file_path.write_text(content, encoding="gbk")
    text = extract_text_from_file(str(file_path))
    assert text == content


def test_extract_text_from_md(tmp_path):
    from storage.file_processor import extract_text_from_file

    file_path = tmp_path / "test.md"
    file_path.write_text("# Title\n\nSome content", encoding="utf-8")
    text = extract_text_from_file(str(file_path))
    assert "# Title" in text


def test_extract_text_unsupported_type(tmp_path):
    from storage.file_processor import extract_text_from_file

    file_path = tmp_path / "test.xyz"
    file_path.write_text("some content", encoding="utf-8")
    text = extract_text_from_file(str(file_path))
    assert text == ""


def test_extract_text_from_xmind(tmp_path):
    import xmind
    from storage.file_processor import extract_text_from_file

    # 创建测试用的 .xmind 文件
    xmind_path = tmp_path / "test.xmind"
    wb = xmind.load(str(xmind_path))
    sheet = wb.getPrimarySheet()
    sheet.setTitle("测试导图")
    root = sheet.getRootTopic()
    root.setTitle("根节点")
    t1 = root.addSubTopic()
    t1.setTitle("子节点A")
    t1.setPlainNotes("备注内容")
    t2 = root.addSubTopic()
    t2.setTitle("子节点B")
    t2_1 = t2.addSubTopic()
    t2_1.setTitle("孙节点")
    xmind.save(wb, str(xmind_path))

    text = extract_text_from_file(str(xmind_path))

    assert "测试导图" in text
    assert "根节点" in text
    assert "子节点A" in text
    assert "子节点B" in text
    assert "孙节点" in text
    assert "备注内容" in text

    from xmindparser import xmind_to_markdown
    t = xmind_to_markdown("D:\\dudu\\Documents\\JVM.xmind")
    print(t)


def test_compute_file_hash():
    from storage.file_processor import compute_file_hash

    h = compute_file_hash(b"hello")
    assert len(h) == 32
    assert h == "5d41402abc4b2a76b9719d911017c592"


def test_save_and_get_file_record():
    from storage.models import save_file_record, get_file_record_by_hash, get_file_records

    rid = save_file_record(
        file_name="test.txt",
        file_type=".txt",
        file_hash="abc123",
        extracted_text="hello world",
        knowledge_ids=[1, 2],
        source_user_id="user1",
    )
    assert rid > 0

    record = get_file_record_by_hash("abc123")
    assert record is not None
    assert record["file_name"] == "test.txt"
    assert record["file_hash"] == "abc123"

    records = get_file_records(limit=5)
    assert len(records) == 1


def test_get_file_record_not_found():
    from storage.models import get_file_record_by_hash

    assert get_file_record_by_hash("nonexistent") is None


def test_file_record_has_correct_schema():
    """Verify the file_records table exists and has the right columns."""
    conn = get_connection()
    cur = conn.execute("PRAGMA table_info(file_records)")
    columns = {row["name"] for row in cur.fetchall()}
    conn.close()
    assert "id" in columns
    assert "file_name" in columns
    assert "file_type" in columns
    assert "file_hash" in columns
    assert "extracted_text" in columns
    assert "knowledge_ids" in columns
    assert "source_user_id" in columns
    assert "created_at" in columns
