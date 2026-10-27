from __future__ import annotations

import time

import polars as pl


class TushareClient:
    def __init__(
        self,
        token: str | None = None,
        *,
        pro=None,
        rate_limit_per_min: int = 180,
        max_retries: int = 3,
        sleep=time.sleep,
    ):
        if pro is None:
            import tushare as ts

            if not token:
                raise ValueError("Tushare token is required when pro is not injected")
            pro = ts.pro_api(token)
        self._pro = pro
        self._min_interval = 60.0 / max(rate_limit_per_min, 1)
        self._last_call = 0.0
        self._max_retries = max_retries
        self._sleep = sleep

    def fetch(self, endpoint: str, **params) -> pl.DataFrame:
        method = getattr(self._pro, endpoint)
        for attempt in range(self._max_retries):
            self._throttle()
            try:
                frame = method(**params)
                if frame is None:
                    return pl.DataFrame()
                if isinstance(frame, pl.DataFrame):
                    return frame
                return pl.from_pandas(frame)
            except Exception:
                if attempt == self._max_retries - 1:
                    raise
                self._sleep(2**attempt)
        raise RuntimeError("unreachable")

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            self._sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()
