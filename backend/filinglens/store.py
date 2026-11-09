"""SQLite storage for filings, chunks, and their embeddings."""
import sqlite3
from contextlib import contextmanager

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    company TEXT NOT NULL,
    form TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    filing_date TEXT NOT NULL,
    url TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    filing_id TEXT NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    section TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_filing ON chunks(filing_id);
CREATE TABLE IF NOT EXISTS financial_facts (
    ticker TEXT NOT NULL,
    company TEXT NOT NULL,
    concept TEXT NOT NULL,
    label TEXT NOT NULL,
    unit TEXT NOT NULL,
    value REAL NOT NULL,
    period_start TEXT,
    period_end TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    filed TEXT NOT NULL,
    accession TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS facts_lookup ON financial_facts(ticker, concept, fiscal_year);
"""


@contextmanager
def connect():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def filing_exists(filing_id: str) -> bool:
    with connect() as db:
        return db.execute("SELECT 1 FROM filings WHERE id = ?", (filing_id,)).fetchone() is not None


def save_filing(filing: dict, chunks: list[dict], embeddings: np.ndarray) -> None:
    with connect() as db:
        db.execute("DELETE FROM filings WHERE id = ?", (filing["id"],))
        db.execute(
            "INSERT INTO filings VALUES (:id, :ticker, :company, :form, :fiscal_year, :filing_date, :url)",
            filing,
        )
        db.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    f"{filing['id']}:{i}",
                    filing["id"],
                    i,
                    chunk["section"],
                    chunk["kind"],
                    chunk["text"],
                    embeddings[i].astype(np.float32).tobytes(),
                )
                for i, chunk in enumerate(chunks)
            ],
        )


def has_facts(ticker: str) -> bool:
    with connect() as db:
        query = "SELECT 1 FROM financial_facts WHERE ticker = ? LIMIT 1"
        return db.execute(query, (ticker,)).fetchone() is not None


def save_facts(ticker: str, facts: list[dict]) -> None:
    with connect() as db:
        db.execute("DELETE FROM financial_facts WHERE ticker = ?", (ticker,))
        db.executemany(
            """INSERT INTO financial_facts VALUES (:ticker, :company, :concept, :label, :unit,
               :value, :period_start, :period_end, :fiscal_year, :filed, :accession)""",
            facts,
        )


def list_filings() -> list[dict]:
    with connect() as db:
        rows = db.execute(
            """SELECT f.*, COUNT(c.id) AS chunk_count
               FROM filings f LEFT JOIN chunks c ON c.filing_id = f.id
               GROUP BY f.id ORDER BY f.ticker, f.fiscal_year DESC"""
        ).fetchall()
    return [dict(row) for row in rows]


def load_chunks(filing_ids: list[str] | None = None) -> list[dict]:
    query = """SELECT c.*, f.ticker, f.company, f.form, f.fiscal_year, f.url
               FROM chunks c JOIN filings f ON f.id = c.filing_id"""
    params: list[str] = []
    if filing_ids:
        query += f" WHERE c.filing_id IN ({','.join('?' * len(filing_ids))})"
        params = filing_ids
    query += " ORDER BY c.filing_id, c.position"
    with connect() as db:
        return [dict(row) for row in db.execute(query, params).fetchall()]
