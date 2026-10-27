from __future__ import annotations

from datetime import date, datetime

import polars as pl

_SOURCE_PREFERENCE = ("SW2021", "SW")
_MEMBERSHIP_COLUMNS = ("index_code", "in_date", "out_date")


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


def _select_source(frame: pl.DataFrame) -> str | None:
    """Pick one classification source deterministically.

    Tushare publishes SW versions through ``src`` (for example ``SW2021``
    and the older ``SW``); prefer the newest SW version, then fall back to
    the lexicographically first non-null source.
    """
    values = frame.get_column("src").drop_nulls().unique().to_list()
    sources = sorted(
        {str(value).strip() for value in values if str(value).strip()}
    )
    if not sources:
        return None
    for preferred in _SOURCE_PREFERENCE:
        if preferred in sources:
            return preferred
    return sources[0]


def _classify_names(
    classify: pl.DataFrame,
) -> tuple[dict[str, str], set[str] | None]:
    """Return L1/source-filtered names and the allowed index codes."""
    frame = classify
    allowed: set[str] | None = None
    if "level" in frame.columns:
        level = (
            pl.col("level").cast(pl.String).str.strip_chars().str.to_uppercase()
        )
        frame = frame.filter(level == "L1")
        allowed = set(frame["index_code"].drop_nulls().to_list())
    if "src" in frame.columns:
        selected = _select_source(frame)
        if selected is not None:
            frame = frame.filter(
                pl.col("src").cast(pl.String).str.strip_chars() == selected
            )
            source_codes = set(frame["index_code"].drop_nulls().to_list())
            allowed = (
                source_codes
                if allowed is None
                else allowed & source_codes
            )
    names: dict[str, str] = {}
    for row in frame.iter_rows(named=True):
        index_code = row.get("index_code")
        if index_code is None:
            continue
        names[index_code] = row.get("industry_name") or index_code
    return names, allowed


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
            _require_columns(
                index_member, _MEMBERSHIP_COLUMNS, "index_member"
            )

            names: dict[str, str] = {}
            allowed_codes: set[str] | None = None
            if index_classify is not None and not index_classify.is_empty():
                _require_columns(
                    index_classify, ("index_code",), "index_classify"
                )
                names, allowed_codes = _classify_names(index_classify)

            candidates = index_member
            if allowed_codes is not None:
                candidates = candidates.filter(
                    pl.col("index_code").is_in(list(allowed_codes))
                )
            candidate_count = candidates.height

            # One active membership per stock: latest in_date wins, then the
            # lexicographically smallest index_code for a deterministic tie.
            chosen: dict[str, tuple[date, str, str]] = {}
            unusable = 0
            for row in candidates.iter_rows(named=True):
                code = row.get("con_code") or row.get("ts_code")
                index_code = row.get("index_code")
                raw_out = row.get("out_date")
                in_date = _parse_date(row.get("in_date"))
                out_date = _parse_date(raw_out)
                if (
                    code is None
                    or index_code is None
                    or in_date is None
                    or (raw_out is not None and out_date is None)
                ):
                    unusable += 1
                    continue
                if as_of is not None and in_date > as_of:
                    continue
                if as_of is not None and out_date is not None and out_date < as_of:
                    continue
                candidate = (
                    in_date,
                    str(index_code),
                    names.get(index_code, index_code),
                )
                current = chosen.get(code)
                if current is None or candidate[0] > current[0] or (
                    candidate[0] == current[0]
                    and candidate[1] < current[1]
                ):
                    chosen[code] = candidate

            if chosen:
                mapping = {
                    code: candidate[2] for code, candidate in chosen.items()
                }
                if not stock_basic.is_empty():
                    for row in stock_basic.iter_rows(named=True):
                        mapping.setdefault(
                            row["ts_code"], _stock_industry(row)
                        )
                return cls(mapping, "sw", unusable > 0)
            if candidate_count > 0 and unusable == candidate_count:
                raise ValueError(
                    "index_member has no usable membership rows "
                    "(invalid or missing in_date/out_date)"
                )

        mapping = {}
        if not stock_basic.is_empty():
            for row in stock_basic.iter_rows(named=True):
                mapping[row["ts_code"]] = _stock_industry(row)
        return cls(mapping, "stock_basic", True)
