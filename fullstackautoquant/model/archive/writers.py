from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path


class ArchiveStrategy(ABC):
    @abstractmethod
    def persist(self, source: Path, used_date: str) -> Path:
        """Persist the generated signal file and return the target path."""


class LocalArchive(ArchiveStrategy):
    def __init__(self, root: Path) -> None:
        self._root = root

    def persist(self, source: Path, used_date: str) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        target = self._root / f"ranked_scores_{used_date}.csv"
        shutil.copy2(source, target)
        return target
