"""Archive strategy module for persisting inference outputs."""

from .writers import ArchiveStrategy, LocalArchive

__all__ = ["ArchiveStrategy", "LocalArchive"]
