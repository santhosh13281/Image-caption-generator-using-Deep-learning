"""
SQLite database module for storing caption generation history.

Stores:
  - Uploaded image name
  - Generated caption
  - Timestamp
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config import DB_PATH

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS caption_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name  TEXT    NOT NULL,
    caption     TEXT    NOT NULL,
    confidence  REAL,
    created_at  TEXT    NOT NULL
);
"""


class CaptionDatabase:
    """
    SQLite wrapper for caption generation records.

    Usage:
        db = CaptionDatabase()
        db.save("photo.jpg", "A dog playing in the grass.", confidence=0.82)
        records = db.get_all(limit=10)
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        """
        Initialize database connection and create table if needed.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create the caption_history table if it does not exist."""
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()
        logger.info("Database initialized at %s", self.db_path)

    @contextmanager
    def _connect(self):
        """Context manager for SQLite connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def save(
        self,
        image_name: str,
        caption: str,
        confidence: Optional[float] = None,
    ) -> int:
        """
        Insert a new caption record.

        Args:
            image_name: Name of the uploaded image file.
            caption: Generated caption text.
            confidence: Optional model confidence score (0–1).

        Returns:
            ID of the inserted row.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO caption_history (image_name, caption, confidence, created_at) "
                "VALUES (?, ?, ?, ?)",
                (image_name, caption, confidence, timestamp),
            )
            conn.commit()
            row_id = cursor.lastrowid

        logger.info("Saved caption record id=%d for '%s'.", row_id, image_name)
        return row_id

    def get_all(self, limit: int = 50) -> List[Dict]:
        """
        Retrieve caption history records, newest first.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of record dictionaries.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, image_name, caption, confidence, created_at "
                "FROM caption_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_by_id(self, record_id: int) -> Optional[Dict]:
        """Retrieve a single record by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, image_name, caption, confidence, created_at "
                "FROM caption_history WHERE id = ?",
                (record_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete(self, record_id: int) -> bool:
        """Delete a record by ID. Returns True if a row was deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM caption_history WHERE id = ?", (record_id,)
            )
            conn.commit()
        return cursor.rowcount > 0

    def clear_all(self) -> int:
        """Delete all records. Returns number of deleted rows."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM caption_history")
            conn.commit()
        return cursor.rowcount

    def count(self) -> int:
        """Return total number of stored records."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM caption_history").fetchone()
        return row[0] if row else 0


# Module-level singleton
_db: Optional[CaptionDatabase] = None


def get_database() -> CaptionDatabase:
    """Return a cached CaptionDatabase instance."""
    global _db
    if _db is None:
        _db = CaptionDatabase()
    return _db
