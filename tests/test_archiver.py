from __future__ import annotations

from pathlib import Path

from fullstackautoquant.model.archive import LocalArchive


def test_local_archive(tmp_path: Path) -> None:
    source = tmp_path / "ranked.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    archive_root = tmp_path / "archive"

    strategy = LocalArchive(archive_root)
    target = strategy.persist(source, "2025-10-17")

    assert target.exists()
    assert target.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
