from __future__ import annotations

import sqlite3

from scripts.backup_db import backup


def test_sqlite_backup_is_readable_and_retains_limit(tmp_path):
    database = tmp_path / "app.db"
    source = sqlite3.connect(database)
    source.execute("create table sample (value integer)")
    source.execute("insert into sample values (42)")
    source.commit()
    source.close()

    out = tmp_path / "backups"
    first = backup(database, out, retain=1)
    copied = sqlite3.connect(first)
    assert copied.execute("select value from sample").fetchone()[0] == 42
    copied.close()

    second = backup(database, out, retain=1)
    remaining = sorted(out.glob("app-*.db"))
    assert remaining == [second]
