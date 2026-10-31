# SQLite storage for the cache.
# One row = one saved question, its answer and its vector (the embedding).
# PRIVACY: this file stores users' question text and answers. Treat it like user data.
import sqlite3
import time
from pathlib import Path

import numpy as np


class CacheStore:
    def __init__(self, path="calibcache.db"):
        self.path = Path(path)
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row  # lets us read columns by name

        # Create the table the first time
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                vector BLOB NOT NULL,
                created_at REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                gen_cost REAL NOT NULL DEFAULT 0,
                gen_seconds REAL NOT NULL DEFAULT 0
            )
            """
        )
        # Older database files do not have the two newest columns: add them
        columns = [row["name"] for row in self.conn.execute("PRAGMA table_info(cache_entries)")]
        if "gen_cost" not in columns:
            self.conn.execute("ALTER TABLE cache_entries ADD COLUMN gen_cost REAL NOT NULL DEFAULT 0")
        if "gen_seconds" not in columns:
            self.conn.execute("ALTER TABLE cache_entries ADD COLUMN gen_seconds REAL NOT NULL DEFAULT 0")

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_group "
            "ON cache_entries(namespace, embedding_model)"
        )
        self.conn.commit()

    def add(self, namespace, embedding_model, question, answer, vector,
            gen_cost=0.0, gen_seconds=0.0):
        """Save one question with its answer and vector. Returns the new id.
        gen_cost and gen_seconds: what it cost (dollars) and how long it took
        to generate this answer, so we can report savings later."""
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        with self.conn:  # saves (commits) automatically
            cursor = self.conn.execute(
                "INSERT INTO cache_entries "
                "(namespace, embedding_model, question, answer, vector, created_at, "
                "gen_cost, gen_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (namespace, embedding_model, question, answer, blob, time.time(),
                 gen_cost, gen_seconds),
            )
        return cursor.lastrowid

    def load_vectors(self, namespace, embedding_model, ttl_seconds=None):
        """Load all vectors of one group. Returns (list of ids, matrix) or ([], None)."""
        sql = ("SELECT id, vector FROM cache_entries "
               "WHERE namespace = ? AND embedding_model = ?")
        params = [namespace, embedding_model]
        if ttl_seconds is not None:
            # ignore entries that are older than ttl_seconds
            sql += " AND created_at >= ?"
            params.append(time.time() - ttl_seconds)

        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return [], None

        ids = [row["id"] for row in rows]
        matrix = np.vstack(
            [np.frombuffer(row["vector"], dtype=np.float32) for row in rows]
        )
        return ids, matrix

    def get_entry(self, entry_id):
        """Read one saved entry as a dictionary."""
        row = self.conn.execute(
            "SELECT id, namespace, question, answer, created_at, hit_count, "
            "gen_cost, gen_seconds FROM cache_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        return dict(row) if row else None

    def add_hit(self, entry_id):
        """Count one more hit for this entry."""
        with self.conn:
            self.conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = ?",
                (entry_id,),
            )

    def count(self, namespace=None):
        """Number of saved entries (all, or only one group)."""
        if namespace is None:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM cache_entries").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM cache_entries WHERE namespace = ?",
                (namespace,),
            ).fetchone()
        return row["n"]

    # ----- privacy tools -----
    def remove_expired(self, ttl_seconds):
        """Delete entries older than ttl_seconds. Returns how many were deleted."""
        cutoff = time.time() - ttl_seconds
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM cache_entries WHERE created_at < ?", (cutoff,)
            )
        return cursor.rowcount

    def delete_namespace(self, namespace):
        """Delete every entry of one group. Returns how many were deleted."""
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM cache_entries WHERE namespace = ?", (namespace,)
            )
        return cursor.rowcount

    def clear(self):
        """Delete everything."""
        with self.conn:
            self.conn.execute("DELETE FROM cache_entries")

    def close(self):
        self.conn.close()