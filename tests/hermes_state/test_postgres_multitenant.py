"""Many Hermes instances sharing one PostgreSQL database, one schema each."""

import os
import sqlite3
import uuid

import pytest

from hermes_state import SessionDB


@pytest.fixture
def tenant_homes(tmp_path, monkeypatch):
    """Two profile homes on the same database, each bound to its own schema."""
    if os.environ.get("HERMES_TEST_POSTGRES") != "1":
        pytest.skip("set HERMES_TEST_POSTGRES=1 to run PostgreSQL integration tests")
    import psycopg
    from psycopg import sql

    database_url = os.environ.get(
        "HERMES_TEST_POSTGRES_URL", "postgresql://hermes:hermes-test@127.0.0.1:55433/hermes_test")
    monkeypatch.delenv("HERMES_DATABASE_URL", raising=False)
    homes = []
    for name in ("a", "b"):
        home = tmp_path / name
        home.mkdir()
        (home / "config.yaml").write_text(
            f"database:\n  backend: postgres\n  schema: tenant_{name}_{uuid.uuid4().hex[:12]}\n", encoding="utf-8")
        (home / ".env").write_text(f"HERMES_DATABASE_URL={database_url}\n", encoding="utf-8")
        homes.append(home)
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    try:
        yield homes
    finally:
        with psycopg.connect(database_url, autocommit=True) as conn:
            for home in homes:
                schema = (home / "config.yaml").read_text().split("schema: ")[1].strip()
                conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


def test_tenants_never_see_each_others_sessions(tenant_homes):
    home_a, home_b = tenant_homes
    with SessionDB(home_a / "state.db") as a, SessionDB(home_b / "state.db") as b:
        assert a.schema != b.schema
        # Same session id in both tenants: independent rows, no collision.
        for db, text in ((a, "tenant a pineapple"), (b, "tenant b mango")):
            db.create_session("shared-id", "cli")
            db.append_message("shared-id", "user", text)

        assert [m["content"] for m in a.get_messages("shared-id")] == ["tenant a pineapple"]
        assert [m["content"] for m in b.get_messages("shared-id")] == ["tenant b mango"]
        assert a.search_messages("mango") == []
        assert b.search_messages("pineapple") == []
        assert [s["id"] for s in b.list_sessions_rich()] == ["shared-id"]


def _sqlite_columns(path):
    conn = sqlite3.connect(path)
    try:
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%fts%'")]
        columns = {table: {row[1] for row in conn.execute(f"PRAGMA table_info('{table}')")} for table in tables}
        indexes = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL")}
    finally:
        conn.close()
    return columns, indexes


def test_postgres_schema_covers_every_sqlite_column_and_index(postgres_db, tmp_path):
    """Upstream grows the SQLite schema constantly; a column missing here fails every insert."""
    with SessionDB(tmp_path / "sqlite.db", database_settings=_sqlite_settings()):
        pass
    sqlite_columns, sqlite_indexes = _sqlite_columns(tmp_path / "sqlite.db")
    with postgres_db._read_ctx() as conn:
        pg_columns = {}
        for table, column in conn.execute(
                "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = ?",
                (postgres_db.schema,)):
            pg_columns.setdefault(table, set()).add(column)
        pg_indexes = {row[0] for row in conn.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = ?", (postgres_db.schema,))}

    missing = {table: sorted(cols - pg_columns.get(table, set()))
               for table, cols in sqlite_columns.items() if cols - pg_columns.get(table, set())}
    assert missing == {}, "add a POSTGRES_MIGRATIONS step for these SQLite columns"
    assert sorted(sqlite_indexes - pg_indexes) == [], "add a POSTGRES_MIGRATIONS step for these indexes"


def test_older_schema_upgrades_in_place_on_next_open(postgres_home):
    from hermes_state_postgres_schema import POSTGRES_SCHEMA_VERSION

    with SessionDB(postgres_home / "state.db") as db:
        db.create_session("kept", "cli")
        db.append_message("kept", "user", "written before the upgrade")
        # Rewind to a version-1 schema: no created_source column, version 1 recorded.
        db._execute_write(lambda conn: conn.execute("ALTER TABLE sessions DROP COLUMN created_source"))
        db._execute_write(lambda conn: conn.execute("UPDATE postgres_schema_version SET version = 1"))

    with SessionDB(postgres_home / "state.db") as db:
        with db._read_ctx() as conn:
            assert conn.execute("SELECT version FROM postgres_schema_version").fetchone()[0] == POSTGRES_SCHEMA_VERSION
            assert conn.execute("SELECT created_source FROM sessions WHERE id = 'kept'").fetchone() is not None
        assert [m["content"] for m in db.get_messages("kept")] == ["written before the upgrade"]


def _sqlite_settings():
    from hermes_state_backend import DatabaseSettings

    return DatabaseSettings()
