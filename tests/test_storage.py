import pytest
from storage.database import init_db


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    monkeypatch.setattr("storage.database.DB_DIR", str(db_dir))
    monkeypatch.setattr("storage.database.DB_PATH", str(db_dir / "knowledge.db"))
    init_db()


