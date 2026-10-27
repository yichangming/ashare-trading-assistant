from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime

from pydantic import BaseModel, ConfigDict, Field


class DataManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trade_date: date
    row_counts: dict[str, int]
    checksums: dict[str, str]
    reference_checksums: dict[str, str] = Field(default_factory=dict)
    degraded: list[str] = Field(default_factory=list)
    reference_degraded: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data_version: str | None = None

    def version(self) -> str:
        payload = self.model_dump(mode="json", exclude={"created_at", "data_version"})
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()[:12]
