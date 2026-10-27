from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

_SAFE = re.compile(r"[^a-zA-Z0-9_\-]")


def _safe(name: str) -> str:
    return _SAFE.sub("_", name)


def _atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        write(temp_path)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _validate_partition_date(frame: pl.DataFrame, trade_date: date) -> None:
    if "trade_date" not in frame.columns:
        raise ValueError("processed data must contain a trade_date column")
    column = frame.get_column("trade_date")
    if column.dtype != pl.Date:
        raise ValueError(f"trade_date column must have dtype Date, got {column.dtype}")
    if column.null_count() > 0 or not (column == trade_date).all():
        raise ValueError(
            "all trade_date values must match partition date "
            f"{trade_date.isoformat()}"
        )


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
        _atomic_write(path, df.write_parquet)
        return path

    def read_raw(self, endpoint: str, key: str) -> pl.DataFrame:
        path = self.raw_dir / _safe(endpoint) / f"{_safe(key)}.parquet"
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(path)

    def write_reference(self, name: str, df: pl.DataFrame) -> Path:
        path = self.raw_dir / "reference" / f"{_safe(name)}.parquet"
        _atomic_write(path, df.write_parquet)
        return path

    def read_reference(self, name: str) -> pl.DataFrame:
        path = self.raw_dir / "reference" / f"{_safe(name)}.parquet"
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(path)

    def write_processed(self, dataset: str, trade_date: date, df: pl.DataFrame) -> Path:
        _validate_partition_date(df, trade_date)
        path = (
            self.processed_dir
            / _safe(dataset)
            / f"trade_date={trade_date.isoformat()}"
            / "data.parquet"
        )
        _atomic_write(path, df.write_parquet)
        return path

    def read_processed(self, dataset: str, start: date, end: date) -> pl.DataFrame:
        root = self.processed_dir / _safe(dataset)
        if not any(root.glob("trade_date=*/data.parquet")):
            return pl.DataFrame()
        pattern = str(root / "trade_date=*" / "data.parquet").replace("'", "''")
        connection = duckdb.connect()
        try:
            return connection.execute(
                f"SELECT * FROM read_parquet('{pattern}', "
                "hive_partitioning = true, union_by_name = true) "
                "WHERE trade_date BETWEEN ? AND ?",
                [start, end],
            ).pl()
        finally:
            connection.close()

    def write_regime(self, trade_date: date, payload: dict) -> Path:
        path = self.regime_dir / f"trade_date={trade_date.isoformat()}.json"
        content = json.dumps(payload, ensure_ascii=False, default=str, indent=2)
        _atomic_write(
            path,
            lambda temp_path: temp_path.write_text(content, encoding="utf-8"),
        )
        return path
