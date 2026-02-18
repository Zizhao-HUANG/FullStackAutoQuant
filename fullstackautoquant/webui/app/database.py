"""Database access layer (SQLite)."""

from __future__ import annotations

import contextlib
import datetime as dt
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import Date, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


@dataclass
class DatabaseConfig:
    url: str
    echo: bool = False


class Base(DeclarativeBase):
    pass


class Position(Base):
    __tablename__ = "positions"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    qty: Mapped[float] = mapped_column(Float, default=0.0)
    cost_price: Mapped[float] = mapped_column(Float, default=0.0)
    market_value: Mapped[float] = mapped_column(Float, default=0.0)
    update_time: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class Plan(Base):
    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String)
    target_qty: Mapped[float] = mapped_column(Float)
    delta_qty: Mapped[float] = mapped_column(Float)
    current_qty: Mapped[float] = mapped_column(Float, default=0.0)
    last_price: Mapped[float] = mapped_column(Float, default=0.0)
    delta_value: Mapped[float] = mapped_column(Float, default=0.0)
    target_value: Mapped[float] = mapped_column(Float, default=0.0)
    current_value: Mapped[float] = mapped_column(Float, default=0.0)
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    gen_date: Mapped[dt.date] = mapped_column(Date, default=dt.date.today)


class EditLog(Base):
    __tablename__ = "edits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String)
    before_qty: Mapped[float] = mapped_column(Float)
    after_qty: Mapped[float] = mapped_column(Float)
    modify_time: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class Snapshot(Base):
    __tablename__ = "snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[dt.date] = mapped_column(Date)
    symbol: Mapped[str] = mapped_column(String)
    qty: Mapped[float] = mapped_column(Float)
    market_value: Mapped[float] = mapped_column(Float)


class ConfigEntry(Base):
    __tablename__ = "config"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class HistoricalPosition(Base):
    __tablename__ = "historical_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    qty: Mapped[float] = mapped_column(Float, default=0.0)
    cost_price: Mapped[float] = mapped_column(Float, default=0.0)
    record_date: Mapped[dt.date] = mapped_column(Date, index=True)
    note: Mapped[str] = mapped_column(String, default="")
    inserted_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=dt.datetime.now(dt.timezone.utc))


class Database:
    def __init__(self, config: DatabaseConfig) -> None:
        self.engine = create_engine(config.url, echo=config.echo, future=True)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextlib.contextmanager
    def session(self) -> Iterator[Session]:
        with Session(self.engine) as session:
            yield session

    # Positions
    def get_positions(self) -> list[Position]:
        with self.session() as sess:
            return list(sess.query(Position).order_by(Position.symbol).all())

    def upsert_position(self, symbol: str, qty: float, cost_price: float, market_value: float) -> None:
        now = dt.datetime.utcnow()
        with self.session() as sess:
            obj = sess.get(Position, symbol)
            if obj is None:
                obj = Position(symbol=symbol)
                sess.add(obj)
            obj.qty = qty
            obj.cost_price = cost_price
            obj.market_value = market_value
            obj.update_time = now
            sess.commit()

    def record_edit(self, symbol: str, before_qty: float, after_qty: float) -> None:
        if before_qty == after_qty:
            return
        with self.session() as sess:
            log = EditLog(symbol=symbol, before_qty=before_qty, after_qty=after_qty)
            sess.add(log)
            sess.commit()

    def replace_positions(self, rows: Sequence[dict[str, Any]]) -> None:
        with self.session() as sess:
            now = dt.datetime.now(dt.timezone.utc)
        with self.session() as sess:
            now = dt.datetime.now(dt.timezone.utc)
            existing = {pos.symbol: pos for pos in sess.query(Position).all()}
            incoming_symbols = set()
            for row in rows:
                symbol = row.get("symbol")
                if not symbol:
                    continue
                incoming_symbols.add(symbol)
                qty = float(row.get("qty", 0.0) or 0.0)
                cost_price = float(row.get("cost_price", 0.0) or 0.0)
                market_value = float(row.get("market_value", 0.0) or 0.0)
                obj = existing.get(symbol)
                before_qty = obj.qty if obj else 0.0
                if obj is None:
                    obj = Position(symbol=symbol)
                    sess.add(obj)
                obj.qty = qty
                obj.cost_price = cost_price
                obj.market_value = market_value
                obj.update_time = now
                if before_qty != qty:
                    sess.add(EditLog(symbol=symbol, before_qty=before_qty, after_qty=qty, modify_time=now))

            to_remove = set(existing.keys()) - incoming_symbols
            if to_remove:
                sess.query(Position).filter(Position.symbol.in_(to_remove)).delete(synchronize_session=False)
                for symbol in to_remove:
                    sess.add(EditLog(symbol=symbol, before_qty=existing[symbol].qty, after_qty=0.0, modify_time=now))

            sess.commit()

    def get_edit_logs(self, limit: int = 200) -> list[EditLog]:
        with self.session() as sess:
            return list(
                sess.query(EditLog)
                .order_by(EditLog.modify_time.desc())
                .limit(limit)
                .all()
            )

    # Plans
    def upsert_plans(self, rows: Sequence[dict[str, Any]], date: dt.date | None = None) -> None:
        with self.session() as sess:
            date = date or dt.date.today()
            sess.query(Plan).filter(Plan.gen_date == date).delete(synchronize_session=False)
            for row in rows:
                symbol = row.get("symbol")
                if not symbol:
                    continue
                sess.add(Plan(
                    symbol=symbol,
                    target_qty=float(row.get("target_qty", 0.0) or 0.0),
                    delta_qty=float(row.get("delta_qty", 0.0) or 0.0),
                    current_qty=float(row.get("qty_current", 0.0) or 0.0),
                    last_price=float(row.get("last_price", 0.0) or 0.0),
                    delta_value=float(row.get("delta_value", 0.0) or 0.0),
                    target_value=float(row.get("target_value", 0.0) or 0.0),
                    current_value=float(row.get("current_value", 0.0) or 0.0),
                    weight=float(row.get("weight", 0.0) or 0.0),
                    gen_date=date,
                ))
            sess.commit()

    def get_plans(self, date: dt.date | None = None) -> list[Plan]:
        with self.session() as sess:
            query = sess.query(Plan)
            if date is not None:
                query = query.filter(Plan.gen_date == date)
            return list(query.order_by(Plan.symbol).all())

    # Snapshots
    def insert_snapshot(self, rows: list[dict[str, Any]], date: dt.date | None = None) -> None:
        with self.session() as sess:
            date = date or dt.date.today()
            for row in rows:
                sess.add(Snapshot(
                    date=date,
                    symbol=row.get("symbol"),
                    qty=row.get("qty", 0.0),
                    market_value=row.get("market_value", 0.0),
                ))
            sess.commit()

    def get_snapshots(self, date: dt.date | None = None) -> list[Snapshot]:
        with self.session() as sess:
            query = sess.query(Snapshot)
            if date is not None:
                query = query.filter(Snapshot.date == date)
            return list(query.order_by(Snapshot.date.desc(), Snapshot.symbol).all())

    # Config
    def set_config(self, key: str, value: str) -> None:
        with self.session() as sess:
            entry = sess.get(ConfigEntry, key)
            if entry is None:
                entry = ConfigEntry(key=key)
                sess.add(entry)
            entry.value = value
            sess.commit()

    def get_config(self, key: str, default: str | None = None) -> str | None:
        with self.session() as sess:
            entry = sess.get(ConfigEntry, key)
            return entry.value if entry else default

    def get_all_config(self) -> dict[str, str]:
        with self.session() as sess:
            entries = sess.query(ConfigEntry).all()
            return {entry.key: entry.value for entry in entries}

    def append_historical_positions(self, rows: Sequence[dict[str, Any]], date: dt.date | None = None, note: str = "") -> None:
        with self.session() as sess:
            record_date = date or dt.date.today()
            now = dt.datetime.utcnow()
            for row in rows:
                symbol = row.get("symbol")
                if not symbol:
                    continue
                sess.add(HistoricalPosition(
                    symbol=str(symbol),
                    qty=float(row.get("qty", 0.0) or 0.0),
                    cost_price=float(row.get("cost_price", 0.0) or 0.0),
                    record_date=record_date,
                    note=note,
                    inserted_at=now,
                ))
            sess.commit()

    def get_historical_positions(self, limit: int = 200) -> list[HistoricalPosition]:
        with self.session() as sess:
            query = sess.query(HistoricalPosition).order_by(HistoricalPosition.record_date.desc(), HistoricalPosition.id.desc())
            if limit > 0:
                query = query.limit(limit)
            return list(query.all())


def build_database_from_config(config: dict[str, Any]) -> Database:
    db_path = config.get("paths", {}).get("database", "../data/trade.db")
    resolved = Path(__file__).resolve().parents[1] / db_path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{resolved}"
    cfg = DatabaseConfig(url=url)
    db = Database(cfg)
    db.create_schema()
    return db

