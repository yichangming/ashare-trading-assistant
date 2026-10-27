from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

_SAFE = re.compile(r"[^a-zA-Z0-9_\-]")


def _safe(name: str) -> str:
    return _SAFE.sub("_", name)


class ParquetStore:
    def __init__(
        self,
        raw_dir: Path | str,
        processed_dir: Path | str,
        regime_dir: Path | str,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.regime_dir = Path(regime_dir)
        for directory in (self.raw_dir, self.processed_dir, self.regime_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def write_raw(self, endpoint: str, key: str, df: pl.DataFrame) -> Path:
        path = self.raw_dir / _safe(endpoint) / f"{_safe(key)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return path

    def read_raw(self, endpoint: str, key: str) -> pl.DataFrame:
        path = self.raw_dir / _safe(endpoint) / f"{_safe(key)}.parquet"
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(path)

    def write_reference(self, name: str, df: pl.DataFrame) -> Path:
        path = self.raw_dir / "reference" / f"{_safe(name)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return path

    def read_reference(self, name: str) -> pl.DataFrame:
        path = self.raw_dir / "reference" / f"{_safe(name)}.parquet"
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(path)

    def write_processed(self, dataset: str, trade_date: date, df: pl.DataFrame) -> Path:
        path = (
            self.processed_dir
            / _safe(dataset)
            / f"trade_date={trade_date.isoformat()}"
            / "data.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return path

    def read_processed(self, dataset: str, start: date, end: date) -> pl.DataFrame:
        root = self.processed_dir / _safe(dataset)
        if not any(root.glob("trade_date=*/data.parquet")):
            return pl.DataFrame()
        pattern = str(root / "trade_date=*" / "data.parquet").replace("'", "''")
        connection = duckdb.connect()
        try:
            return connection.execute(
                f"SELECT * FROM read_parquet('{pattern}') "
                "WHERE trade_date BETWEEN ? AND ?",
                [start, end],
            ).pl()
        finally:
            connection.close()

    def write_regime(self, trade_date: date, payload: dict) -> Path:
        path = self.regime_dir / f"trade_date={trade_date.isoformat()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )
        return path
