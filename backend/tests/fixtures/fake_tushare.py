from __future__ import annotations

import polars as pl


class FakePro:
    def __init__(
        self,
        fail_endpoints: set[str] | None = None,
        empty_endpoints: set[str] | None = None,
    ):
        self.fail_endpoints = fail_endpoints or set()
        self.empty_endpoints = empty_endpoints or set()
        self.calls: list[tuple[str, dict]] = []

    def _maybe_fail(self, endpoint: str) -> None:
        if endpoint in self.fail_endpoints:
            raise RuntimeError(f"{endpoint} permission denied")

    def _frame(self, endpoint: str, frame: pl.DataFrame):
        if endpoint in self.empty_endpoints:
            frame = frame.head(0)
        return frame.to_pandas()

    def trade_cal(self, **kwargs):
        self.calls.append(("trade_cal", kwargs))
        self._maybe_fail("trade_cal")
        return self._frame(
            "trade_cal",
            pl.DataFrame(
                {
                    "exchange": ["SSE"] * 5,
                    "cal_date": [
                        "20240101",
                        "20240102",
                        "20240103",
                        "20240104",
                        "20240105",
                    ],
                    "is_open": [0, 1, 1, 1, 1],
                    "pretrade_date": [
                        "20231229",
                        "20231229",
                        "20240102",
                        "20240103",
                        "20240104",
                    ],
                }
            ),
        )

    def stock_basic(self, **kwargs):
        self.calls.append(("stock_basic", kwargs))
        self._maybe_fail("stock_basic")
        return self._frame(
            "stock_basic",
            pl.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "symbol": ["000001", "000002"],
                    "name": ["平安银行", "万科A"],
                    "area": ["深圳", "深圳"],
                    "industry": ["银行", "房地产"],
                    "market": ["主板", "主板"],
                    "list_date": ["20000407", "19910129"],
                    "delist_date": [None, None],
                    "is_hs": ["S", "S"],
                }
            ),
        )

    def namechange(self, **kwargs):
        self.calls.append(("namechange", kwargs))
        self._maybe_fail("namechange")
        return self._frame(
            "namechange",
            pl.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "name": ["平安银行"],
                    "start_date": ["20000407"],
                    "end_date": [None],
                }
            ),
        )

    def _daily_frame(self, **kwargs) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [kwargs.get("trade_date", "20240102")] * 2,
                "open": [10.0, 20.0],
                "high": [10.5, 20.5],
                "low": [9.8, 19.5],
                "close": [10.2, 20.2],
                "pre_close": [10.0, 20.0],
                "change": [0.2, 0.2],
                "pct_chg": [2.0, 1.0],
                "vol": [100000.0, 200000.0],
                "amount": [1020000.0, 4040000.0],
            }
        )

    def daily(self, **kwargs):
        self.calls.append(("daily", kwargs))
        self._maybe_fail("daily")
        return self._frame("daily", self._daily_frame(**kwargs))

    def adj_factor(self, **kwargs):
        self.calls.append(("adj_factor", kwargs))
        self._maybe_fail("adj_factor")
        return self._frame(
            "adj_factor",
            pl.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "trade_date": [kwargs.get("trade_date", "20240102")] * 2,
                    "adj_factor": [1.0, 2.0],
                }
            ),
        )

    def daily_basic(self, **kwargs):
        self.calls.append(("daily_basic", kwargs))
        self._maybe_fail("daily_basic")
        return self._frame(
            "daily_basic",
            pl.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "trade_date": [kwargs.get("trade_date", "20240102")] * 2,
                    "turnover_rate": [1.5, 2.0],
                    "circ_mv": [1000000.0, 2000000.0],
                }
            ),
        )

    def stk_limit(self, **kwargs):
        self.calls.append(("stk_limit", kwargs))
        self._maybe_fail("stk_limit")
        return self._frame(
            "stk_limit",
            pl.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "trade_date": [kwargs.get("trade_date", "20240102")] * 2,
                    "up_limit": [11.0, 22.0],
                    "down_limit": [9.0, 18.0],
                }
            ),
        )

    def suspend_d(self, **kwargs):
        self.calls.append(("suspend_d", kwargs))
        self._maybe_fail("suspend_d")
        return self._frame(
            "suspend_d",
            pl.DataFrame(
                {"ts_code": [], "trade_date": [], "suspend_type": []}
            ),
        )

    def index_daily(self, **kwargs):
        self.calls.append(("index_daily", kwargs))
        self._maybe_fail("index_daily")
        return self._frame(
            "index_daily",
            pl.DataFrame(
                {
                    "ts_code": ["000300.SH"],
                    "trade_date": [kwargs.get("trade_date", "20240102")],
                    "open": [4000.0],
                    "high": [4050.0],
                    "low": [3980.0],
                    "close": [4020.0],
                    "vol": [1000000.0],
                    "amount": [2000000.0],
                }
            ),
        )

    def limit_list_d(self, **kwargs):
        self.calls.append(("limit_list_d", kwargs))
        self._maybe_fail("limit_list_d")
        return self._frame(
            "limit_list_d",
            pl.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": [kwargs.get("trade_date", "20240102")],
                    "first_time": ["09:35:00"],
                    "last_time": ["14:50:00"],
                    "open_times": [1],
                    "limit_amount": [1000000.0],
                }
            ),
        )

    def index_classify(self, **kwargs):
        self.calls.append(("index_classify", kwargs))
        self._maybe_fail("index_classify")
        return self._frame(
            "index_classify",
            pl.DataFrame(
                {"index_code": ["801010.SI"], "industry_name": ["银行"]}
            ),
        )

    def index_member(self, **kwargs):
        self.calls.append(("index_member", kwargs))
        self._maybe_fail("index_member")
        return self._frame(
            "index_member",
            pl.DataFrame(
                {
                    "index_code": ["801010.SI"],
                    "con_code": ["000001.SZ"],
                    "in_date": ["20000407"],
                    "out_date": [None],
                    "is_new": ["Y"],
                }
            ),
        )
