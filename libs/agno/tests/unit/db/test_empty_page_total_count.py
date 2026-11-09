"""An empty page must keep the filtered total, not report an empty collection.

Several adapters compute the filtered total before LIMIT/OFFSET and then
discarded it in the empty-result branch: any page past the last row returned
``([], 0)`` while earlier pages reported the real total. The AgentOS list
routes turn that into ``meta.total_count=0`` / ``meta.total_pages=0``, so a
client reading pagination metadata concludes the collection is empty and
offers no way back to the pages that do have rows.

The SQLite half always runs. The Postgres half skips itself when psycopg is
missing or nothing answers on the CI service port, matching the other
``*_postgres`` suites under this directory.
"""

from __future__ import annotations

import pytest

from agno.db.schemas.evals import EvalRunRecord, EvalType
from agno.db.schemas.memory import UserMemory
from agno.db.sqlite import AsyncSqliteDb, SqliteDb

PAGE_LIMIT = 2
SEEDED = 3


def _seed(db) -> None:
    for i in range(SEEDED):
        db.create_eval_run(
            EvalRunRecord(
                run_id=f"run_{i}",
                eval_type=EvalType.ACCURACY,
                eval_data={"score": 0.9},
                eval_input={},
            )
        )
        db.upsert_user_memory(UserMemory(memory_id=f"mem_{i}", memory=f"synthetic {i}", user_id="alice"))


def _assert_eval_pagination(db) -> None:
    eval_page_one, eval_total = db.get_eval_runs(limit=PAGE_LIMIT, page=1, deserialize=False)
    assert len(eval_page_one) == PAGE_LIMIT
    assert eval_total == SEEDED

    eval_empty_page, eval_total = db.get_eval_runs(limit=PAGE_LIMIT, page=3, deserialize=False)
    assert eval_empty_page == []
    assert eval_total == SEEDED, "an out-of-range page still knows how many runs exist"


def _assert_memory_pagination(db) -> None:
    mem_page_one, mem_total = db.get_user_memories(user_id="alice", limit=PAGE_LIMIT, page=1, deserialize=False)
    assert len(mem_page_one) == PAGE_LIMIT
    assert mem_total == SEEDED

    mem_empty_page, mem_total = db.get_user_memories(user_id="alice", limit=PAGE_LIMIT, page=3, deserialize=False)
    assert mem_empty_page == []
    assert mem_total == SEEDED, "an out-of-range page still knows how many memories exist"


class TestSqliteEmptyPageTotals:
    def test_sync_eval_runs_empty_page_keeps_total(self, tmp_path):
        db = SqliteDb(db_file=str(tmp_path / "empty_page.db"))
        try:
            _seed(db)
            _assert_eval_pagination(db)
        finally:
            db.close()

    def test_sync_a_genuinely_empty_filter_still_reports_zero(self, tmp_path):
        db = SqliteDb(db_file=str(tmp_path / "empty_filter.db"))
        try:
            _seed(db)
            rows, total = db.get_eval_runs(agent_id="nobody", limit=PAGE_LIMIT, page=1, deserialize=False)
            assert rows == []
            assert total == 0, "no match at all is still zero; only page position must not zero it"
        finally:
            db.close()

    def test_sync_deserialize_true_empty_page_is_still_an_empty_list(self, tmp_path):
        db = SqliteDb(db_file=str(tmp_path / "empty_page_deser.db"))
        try:
            _seed(db)
            assert db.get_eval_runs(limit=PAGE_LIMIT, page=3) == []
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_async_eval_runs_empty_page_keeps_total(self, tmp_path):
        db = AsyncSqliteDb(db_file=str(tmp_path / "empty_page_async.db"))
        try:
            for i in range(SEEDED):
                await db.create_eval_run(
                    EvalRunRecord(
                        run_id=f"arun_{i}",
                        eval_type=EvalType.ACCURACY,
                        eval_data={"score": 0.9},
                        eval_input={},
                    )
                )
                await db.upsert_user_memory(UserMemory(memory_id=f"amem_{i}", memory=f"synthetic {i}", user_id="alice"))

            rows, total = await db.get_eval_runs(limit=PAGE_LIMIT, page=3, deserialize=False)
            assert rows == []
            assert total == SEEDED
        finally:
            await db.close()


POSTGRES_URL = "postgresql+psycopg://ai:***@localhost:5532/ai"


def _have_psycopg() -> bool:
    import importlib.util

    return importlib.util.find_spec("psycopg") is not None


def _postgres_reachable() -> bool:
    if not _have_psycopg():
        return False
    from sqlalchemy import create_engine, text

    engine = create_engine(POSTGRES_URL)
    try:
        with engine.connect() as conn:
            conn.execute(text("select 1"))
        return True
    except Exception:  # noqa: BLE001 - any connection failure means "skip this half"
        return False
    finally:
        engine.dispose()


@pytest.mark.skipif(not _postgres_reachable(), reason="no PostgreSQL on the CI service port")
class TestPostgresEmptyPageTotals:
    @pytest.fixture
    def db(self):
        import uuid

        from agno.db.postgres import PostgresDb

        schema = f"empty_page_{uuid.uuid4().hex[:8]}"
        database = PostgresDb(db_url=POSTGRES_URL, db_schema=schema)
        try:
            yield database
        finally:
            from sqlalchemy import create_engine, text

            engine = create_engine(POSTGRES_URL)
            with engine.connect() as conn:
                conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
                conn.commit()
            engine.dispose()

    def test_sync_eval_runs_and_memories(self, db):
        _seed(db)
        _assert_eval_pagination(db)
        _assert_memory_pagination(db)

    def test_sync_a_genuinely_empty_filter_still_reports_zero(self, db):
        _seed(db)
        rows, total = db.get_eval_runs(agent_id="nobody", limit=PAGE_LIMIT, page=1, deserialize=False)
        assert rows == []
        assert total == 0
