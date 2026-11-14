"""An empty page must keep the filtered total, not report an empty collection.

Several adapters compute the filtered total before LIMIT/OFFSET and then
discarded it in the empty-result branch: any page past the last row returned
``([], 0)`` while earlier pages reported the real total. The AgentOS list
routes turn that into ``meta.total_count=0`` / ``meta.total_pages=0``, so a
client reading pagination metadata concludes the collection is empty and
offers no way back to the pages that do have rows.

``get_eval_runs`` is covered in all four affected adapters, sync and async.
``get_user_memories`` is covered for PostgreSQL and MySQL, the two adapters
where that method shares the bug; the SQLite memory path belongs to the
separate ``#11499`` report and is intentionally not asserted here.

The SQLite half always runs. The server-backed halves skip themselves when
nothing answers on the port the CI services publish (5532 for PostgreSQL,
3306 for MySQL, 27017 for MongoDB) or when the driver is not installed.
"""

from __future__ import annotations

import sys
import uuid

import pytest

from agno.db.schemas.evals import EvalRunRecord, EvalType
from agno.db.schemas.memory import UserMemory
from agno.db.sqlite import AsyncSqliteDb, SqliteDb

PAGE_LIMIT = 2
SEEDED = 3

POSTGRES_URL = "postgresql+psycopg://ai:ai@127.0.0.1:5532/ai"
MYSQL_SYNC_URL = "mysql+pymysql://ai:ai@127.0.0.1:3306/ai"
MYSQL_ASYNC_URL = "mysql+asyncmy://ai:ai@127.0.0.1:3306/ai"
MONGO_URL = "mongodb://127.0.0.1:27017"


# ---------------------------------------------------------------------------
# seeding / assertions
# ---------------------------------------------------------------------------


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


async def _seed_async(db) -> None:
    for i in range(SEEDED):
        await db.create_eval_run(
            EvalRunRecord(
                run_id=f"run_{i}",
                eval_type=EvalType.ACCURACY,
                eval_data={"score": 0.9},
                eval_input={},
            )
        )
        await db.upsert_user_memory(UserMemory(memory_id=f"mem_{i}", memory=f"synthetic {i}", user_id="alice"))


def _assert_eval_pagination(db) -> None:
    eval_page_one, eval_total = db.get_eval_runs(limit=PAGE_LIMIT, page=1, deserialize=False)
    assert len(eval_page_one) == PAGE_LIMIT
    assert eval_total == SEEDED

    eval_empty_page, eval_total = db.get_eval_runs(limit=PAGE_LIMIT, page=3, deserialize=False)
    assert eval_empty_page == []
    assert eval_total == SEEDED, "an out-of-range page still knows how many runs exist"


async def _assert_eval_pagination_async(db) -> None:
    eval_page_one, eval_total = await db.get_eval_runs(limit=PAGE_LIMIT, page=1, deserialize=False)
    assert len(eval_page_one) == PAGE_LIMIT
    assert eval_total == SEEDED

    eval_empty_page, eval_total = await db.get_eval_runs(limit=PAGE_LIMIT, page=3, deserialize=False)
    assert eval_empty_page == []
    assert eval_total == SEEDED, "an out-of-range page still knows how many runs exist"


def _assert_memory_pagination(db) -> None:
    mem_page_one, mem_total = db.get_user_memories(user_id="alice", limit=PAGE_LIMIT, page=1, deserialize=False)
    assert len(mem_page_one) == PAGE_LIMIT
    assert mem_total == SEEDED

    mem_empty_page, mem_total = db.get_user_memories(user_id="alice", limit=PAGE_LIMIT, page=3, deserialize=False)
    assert mem_empty_page == []
    assert mem_total == SEEDED, "an out-of-range page still knows how many memories exist"


async def _assert_memory_pagination_async(db) -> None:
    mem_page_one, mem_total = await db.get_user_memories(user_id="alice", limit=PAGE_LIMIT, page=1, deserialize=False)
    assert len(mem_page_one) == PAGE_LIMIT
    assert mem_total == SEEDED

    mem_empty_page, mem_total = await db.get_user_memories(user_id="alice", limit=PAGE_LIMIT, page=3, deserialize=False)
    assert mem_empty_page == []
    assert mem_total == SEEDED, "an out-of-range page still knows how many memories exist"


def _close(db) -> None:
    close = getattr(db, "close", None)
    if callable(close):
        close()


# ---------------------------------------------------------------------------
# reachability probes: a missing server skips the half, it does not fail it
# ---------------------------------------------------------------------------


def _postgres_reachable() -> bool:
    try:
        import psycopg

        with psycopg.connect("host=127.0.0.1 port=5532 user=ai password=ai dbname=ai", connect_timeout=2) as conn:
            conn.execute("select 1")
        return True
    except Exception:  # noqa: BLE001 - any failure means "no server here"
        return False


def _mysql_reachable() -> bool:
    try:
        import pymysql

        conn = pymysql.connect(host="127.0.0.1", port=3306, user="ai", password="ai", connect_timeout=2)
        try:
            conn.ping()
        finally:
            conn.close()
        return True
    except Exception:  # noqa: BLE001 - any failure means "no server here"
        return False


def _mongo_reachable() -> bool:
    try:
        from pymongo import MongoClient

        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=1500)
        try:
            client.admin.command("ping")
        finally:
            client.close()
        return True
    except Exception:  # noqa: BLE001 - any failure means "no server here"
        return False


HAVE_POSTGRES = _postgres_reachable()
HAVE_MYSQL = _mysql_reachable()
HAVE_MONGO = _mongo_reachable()


# ---------------------------------------------------------------------------
# SQLite: always runs; get_eval_runs only (its memory path is #11499's scope)
# ---------------------------------------------------------------------------


class TestSqliteEmptyPageTotals:
    def test_sync_eval_runs_empty_page_keeps_total(self, tmp_path):
        db = SqliteDb(db_file=str(tmp_path / "empty_page.db"))
        try:
            _seed(db)
            _assert_eval_pagination(db)
        finally:
            _close(db)

    def test_sync_a_genuinely_empty_filter_still_reports_zero(self, tmp_path):
        db = SqliteDb(db_file=str(tmp_path / "empty_filter.db"))
        try:
            _seed(db)
            rows, total = db.get_eval_runs(agent_id="nobody", limit=PAGE_LIMIT, page=1, deserialize=False)
            assert rows == []
            assert total == 0, "no match at all is still zero; only page position must not zero it"
        finally:
            _close(db)

    def test_sync_deserialize_true_empty_page_is_still_an_empty_list(self, tmp_path):
        db = SqliteDb(db_file=str(tmp_path / "empty_page_deser.db"))
        try:
            _seed(db)
            assert db.get_eval_runs(limit=PAGE_LIMIT, page=3) == []
        finally:
            _close(db)

    async def test_async_eval_runs_empty_page_keeps_total(self, tmp_path):
        db = AsyncSqliteDb(db_file=str(tmp_path / "empty_page_async.db"))
        try:
            await _seed_async(db)
            await _assert_eval_pagination_async(db)
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------


def _drop_postgres_schema(schema: str) -> None:
    from sqlalchemy import create_engine, text

    engine = create_engine(POSTGRES_URL)
    try:
        with engine.connect() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            conn.commit()
    finally:
        engine.dispose()


@pytest.mark.skipif(not HAVE_POSTGRES, reason="no PostgreSQL on 127.0.0.1:5532")
class TestPostgresEmptyPageTotals:
    @pytest.fixture
    def db(self):
        from agno.db.postgres import PostgresDb

        schema = f"empty_page_{uuid.uuid4().hex[:8]}"
        database = PostgresDb(db_url=POSTGRES_URL, db_schema=schema)
        try:
            yield database
        finally:
            _close(database)
            _drop_postgres_schema(schema)

    def test_sync_eval_runs_and_memories(self, db):
        _seed(db)
        _assert_eval_pagination(db)
        _assert_memory_pagination(db)

    def test_sync_a_genuinely_empty_filter_still_reports_zero(self, db):
        _seed(db)
        rows, total = db.get_eval_runs(agent_id="nobody", limit=PAGE_LIMIT, page=1, deserialize=False)
        assert rows == []
        assert total == 0
        rows, total = db.get_user_memories(user_id="nobody", limit=PAGE_LIMIT, page=1, deserialize=False)
        assert rows == []
        assert total == 0


@pytest.mark.skipif(not HAVE_POSTGRES, reason="no PostgreSQL on 127.0.0.1:5532")
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="psycopg async needs a selector event loop; Windows defaults to Proactor (CI runs Linux)",
)
class TestAsyncPostgresEmptyPageTotals:
    @pytest.fixture
    async def db(self):
        from agno.db.postgres import AsyncPostgresDb

        schema = f"empty_page_{uuid.uuid4().hex[:8]}"
        database = AsyncPostgresDb(db_url=POSTGRES_URL, db_schema=schema)
        try:
            # The async adapters do not create their tables on construction.
            await database._create_all_tables()
            yield database
        finally:
            await database.close()
            _drop_postgres_schema(schema)

    async def test_async_eval_runs_and_memories(self, db):
        await _seed_async(db)
        await _assert_eval_pagination_async(db)
        await _assert_memory_pagination_async(db)


# ---------------------------------------------------------------------------
# MySQL
# ---------------------------------------------------------------------------


def _create_mysql_schema(schema: str) -> None:
    """The async adapter does not create its database, the sync one does."""
    import pymysql

    conn = pymysql.connect(host="127.0.0.1", port=3306, user="ai", password="ai", connect_timeout=2)
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{schema}`")
        conn.commit()
    finally:
        conn.close()


def _drop_mysql_schema(schema: str) -> None:
    import pymysql

    conn = pymysql.connect(host="127.0.0.1", port=3306, user="ai", password="ai", connect_timeout=2)
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS `{schema}`")
        conn.commit()
    finally:
        conn.close()


@pytest.mark.skipif(not HAVE_MYSQL, reason="no MySQL on 127.0.0.1:3306")
class TestMySqlEmptyPageTotals:
    @pytest.fixture
    def db(self):
        from agno.db.mysql import MySQLDb

        schema = f"empty_page_{uuid.uuid4().hex[:8]}"
        _create_mysql_schema(schema)
        database = MySQLDb(db_url=MYSQL_SYNC_URL, db_schema=schema)
        try:
            yield database
        finally:
            _close(database)
            _drop_mysql_schema(schema)

    def test_sync_eval_runs_and_memories(self, db):
        _seed(db)
        _assert_eval_pagination(db)
        _assert_memory_pagination(db)

    def test_sync_a_genuinely_empty_filter_still_reports_zero(self, db):
        _seed(db)
        rows, total = db.get_eval_runs(agent_id="nobody", limit=PAGE_LIMIT, page=1, deserialize=False)
        assert rows == []
        assert total == 0
        rows, total = db.get_user_memories(user_id="nobody", limit=PAGE_LIMIT, page=1, deserialize=False)
        assert rows == []
        assert total == 0


@pytest.mark.skipif(not HAVE_MYSQL, reason="no MySQL on 127.0.0.1:3306")
class TestAsyncMySqlEmptyPageTotals:
    @pytest.fixture
    async def db(self):
        from agno.db.mysql import AsyncMySQLDb

        schema = f"empty_page_{uuid.uuid4().hex[:8]}"
        _create_mysql_schema(schema)
        database = AsyncMySQLDb(db_url=MYSQL_ASYNC_URL, db_schema=schema)
        try:
            # The async adapters do not create their tables on construction.
            await database._create_all_tables()
            yield database
        finally:
            await database.close()
            _drop_mysql_schema(schema)

    async def test_async_eval_runs_and_memories(self, db):
        await _seed_async(db)
        await _assert_eval_pagination_async(db)
        await _assert_memory_pagination_async(db)


# ---------------------------------------------------------------------------
# MongoDB: get_eval_runs only (its memory path already reports the total)
# ---------------------------------------------------------------------------


def _drop_mongo_database(name: str) -> None:
    from pymongo import MongoClient

    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=1500)
    try:
        client.drop_database(name)
    finally:
        client.close()


@pytest.mark.skipif(not HAVE_MONGO, reason="no MongoDB on 127.0.0.1:27017")
class TestMongoEmptyPageTotals:
    @pytest.fixture
    def db(self):
        from agno.db.mongo import MongoDb

        name = f"empty_page_{uuid.uuid4().hex[:8]}"
        database = MongoDb(db_url=MONGO_URL, db_name=name)
        try:
            yield database
        finally:
            _close(database)
            _drop_mongo_database(name)

    def test_sync_eval_runs_empty_page_keeps_total(self, db):
        _seed(db)
        _assert_eval_pagination(db)

    def test_sync_a_genuinely_empty_filter_still_reports_zero(self, db):
        _seed(db)
        rows, total = db.get_eval_runs(agent_id="nobody", limit=PAGE_LIMIT, page=1, deserialize=False)
        assert rows == []
        assert total == 0


@pytest.mark.skipif(not HAVE_MONGO, reason="no MongoDB on 127.0.0.1:27017")
class TestAsyncMongoEmptyPageTotals:
    @pytest.fixture
    async def db(self):
        from agno.db.mongo import AsyncMongoDb

        name = f"empty_page_{uuid.uuid4().hex[:8]}"
        database = AsyncMongoDb(db_url=MONGO_URL, db_name=name)
        try:
            yield database
        finally:
            await database.close()
            _drop_mongo_database(name)

    async def test_async_eval_runs_empty_page_keeps_total(self, db):
        await _seed_async(db)
        await _assert_eval_pagination_async(db)
