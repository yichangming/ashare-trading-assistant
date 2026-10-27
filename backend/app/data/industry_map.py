from __future__ import annotations

from datetime import date, datetime

import polars as pl


def _parse_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue
    return None


def _require_columns(
    frame: pl.DataFrame, columns: tuple[str, ...], label: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{label} missing required columns: {', '.join(missing)}"
        )


def _stock_industry(row: dict) -> str:
    return row.get("industry") or "未知"


class IndustryMap:
    def __init__(self, mapping: dict[str, str], source: str, degraded: bool):
        self.mapping = mapping
        self.source = source
        self.degraded = degraded

    def to_frame(self) -> pl.DataFrame:
        schema = {"ts_code": pl.String, "sector": pl.String}
        if not self.mapping:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(
            {
                "ts_code": list(self.mapping.keys()),
                "sector": list(self.mapping.values()),
            },
            schema=schema,
        )

    @classmethod
    def from_sources(
        cls,
        stock_basic: pl.DataFrame,
        index_member: pl.DataFrame | None = None,
        index_classify: pl.DataFrame | None = None,
        as_of: date | None = None,
        source: str = "auto",
    ) -> IndustryMap:
        if not stock_basic.is_empty():
            _require_columns(stock_basic, ("ts_code",), "stock_basic")

        if (
            source in ("auto", "sw")
            and index_member is not None
            and not index_member.is_empty()
        ):
            if "con_code" not in index_member.columns and (
                "ts_code" not in index_member.columns
            ):
                raise ValueError(
                    "index_member missing required columns: con_code or ts_code"
                )
            _require_columns(index_member, ("index_code",), "index_member")

            names: dict[str, str] = {}
            if index_classify is not None and not index_classify.is_empty():
                _require_columns(
                    index_classify, ("index_code",), "index_classify"
                )
                for row in index_classify.iter_rows(named=True):
                    index_code = row.get("index_code")
                    if index_code is None:
                        continue
                    names[index_code] = (
                        row.get("industry_name") or index_code
                    )

            mapping: dict[str, str] = {}
            for row in index_member.iter_rows(named=True):
                code = row.get("con_code") or row.get("ts_code")
                index_code = row.get("index_code")
                if code is None or index_code is None:
                    continue
                in_date = _parse_date(row.get("in_date"))
                out_date = _parse_date(row.get("out_date"))
                if as_of is not None and in_date is not None and in_date > as_of:
                    continue
                if as_of is not None and out_date is not None and out_date < as_of:
                    continue
                mapping[code] = names.get(index_code, index_code)
            if mapping:
                if not stock_basic.is_empty():
                    for row in stock_basic.iter_rows(named=True):
                        mapping.setdefault(
                            row["ts_code"], _stock_industry(row)
                        )
                return cls(mapping, "sw", False)

        mapping = {}
        if not stock_basic.is_empty():
            for row in stock_basic.iter_rows(named=True):
                mapping[row["ts_code"]] = _stock_industry(row)
        return cls(mapping, "stock_basic", True)
