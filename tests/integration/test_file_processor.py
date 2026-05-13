"""Integration tests for file processing: wiki extraction from files."""
import pytest
from storage.database import init_db


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    monkeypatch.setattr("storage.wiki_storage.WIKI_DIR", str(tmp_path / "wiki"))
    init_db()


def test_process_file_creates_wiki_pages(monkeypatch, tmp_path):
    """Upload a .txt file -> text extracted -> wiki pages created."""
    from server.bot import _process_and_store_file

    content = b"Python dict is a key-value store. List is an ordered collection."
    result = _process_and_store_file(content, "test.txt", "user1")
    assert "wiki" in result.lower() or "页面" in result


def test_process_file_too_large(monkeypatch, tmp_path):
    """File with >8000 chars -> rejected with length warning."""
    from server.bot import _process_and_store_file, MAX_FILE_CHARS

    large_content = b"A" * (MAX_FILE_CHARS + 1)
    result = _process_and_store_file(large_content, "large.txt", "user1")
    assert "过长" in result
