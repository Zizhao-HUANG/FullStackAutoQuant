"""Database initialization script."""

from __future__ import annotations

from pathlib import Path

from .services import bootstrap_services


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    services = bootstrap_services(base_dir)
    services.db.create_schema()
    print("Database initialized:", services.db.engine.url)


if __name__ == "__main__":
    main()
