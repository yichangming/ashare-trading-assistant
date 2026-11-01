"""Consistent SQLite backup with simple retention.

Usage (from ``backend/``)::

    uv run python scripts/backup_db.py
    uv run python scripts/backup_db.py --database data/app.db \
        --out data/backups --retain 30

The script uses sqlite3's online backup API, so it is safe while the web app
is running.  It does not delete the source database.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_DATABASE = Path("data/app.db")
DEFAULT_OUTPUT = Path("data/backups")


def backup(database: Path, output_dir: Path, retain: int) -> Path:
    if not database.is_file():
        raise FileNotFoundError(f"database not found: {database}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = output_dir / f"app-{stamp}.db"

    source = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    destination = sqlite3.connect(target)
    try:
        source.backup(destination)
        destination.commit()
    finally:
        destination.close()
        source.close()

    backups = sorted(output_dir.glob("app-*.db"), key=lambda path: path.name)
    for stale in backups[:-retain] if retain > 0 else []:
        stale.unlink(missing_ok=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retain", type=int, default=30)
    args = parser.parse_args()
    try:
        target = backup(args.database, args.out, args.retain)
    except (OSError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
