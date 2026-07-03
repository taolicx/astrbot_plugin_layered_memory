from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from .schema import MemoryEntry, clamp01, normalize_category, now_iso


class LayeredMemoryStore:
    """SQLite storage for layered memories and summarized conversation windows."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._fts_available = False
        self._init_db()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _init_db(self) -> None:
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT '',
                    persona_id TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.8,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    source TEXT NOT NULL DEFAULT 'manual',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    locked INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL DEFAULT '',
                    access_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sender_id TEXT NOT NULL DEFAULT '',
                    sender_name TEXT NOT NULL DEFAULT '',
                    summarized INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_state (
                    session_id TEXT PRIMARY KEY,
                    last_summarized_id INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS story_state (
                    session_id TEXT PRIMARY KEY,
                    persona_id TEXT NOT NULL DEFAULT '',
                    state_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    memory_id INTEGER PRIMARY KEY,
                    provider_id TEXT NOT NULL DEFAULT '',
                    dim INTEGER NOT NULL DEFAULT 0,
                    vector_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memory_entries(id) ON DELETE CASCADE
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_session_category ON memory_entries(session_id, category, enabled)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_locked ON memory_entries(locked, enabled)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_message_session_id ON conversation_messages(session_id, id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_vectors_provider ON memory_vectors(provider_id, dim)"
            )
            self._init_fts()
            self.conn.commit()

    def _init_fts(self) -> None:
        try:
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_entries_fts
                USING fts5(title, content, tags)
                """
            )
            self._fts_available = True
        except sqlite3.OperationalError:
            self._fts_available = False

    def add_memory(self, entry: MemoryEntry) -> int:
        category = normalize_category(entry.category)
        created = entry.created_at or now_iso()
        updated = entry.updated_at or created
        locked = 1 if entry.locked or category == "locked" else 0
        tags_json = json.dumps(entry.tags or [], ensure_ascii=False)
        metadata_json = json.dumps(entry.metadata or {}, ensure_ascii=False)
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO memory_entries (
                    session_id, persona_id, category, title, content, importance,
                    confidence, tags_json, metadata_json, source, enabled, locked,
                    created_at, updated_at, last_accessed_at, access_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.session_id or "",
                    entry.persona_id or "",
                    category,
                    entry.title or "",
                    entry.content.strip(),
                    clamp01(entry.importance),
                    clamp01(entry.confidence, 0.8),
                    tags_json,
                    metadata_json,
                    entry.source or "manual",
                    1 if entry.enabled else 0,
                    locked,
                    created,
                    updated,
                    entry.last_accessed_at or "",
                    int(entry.access_count or 0),
                ),
            )
            row_id = int(cur.lastrowid)
            self._upsert_fts(row_id, entry.title, entry.content, " ".join(entry.tags or []))
            self.conn.commit()
            return row_id

    def add_or_merge_memory(
        self,
        entry: MemoryEntry,
        *,
        dedup_enabled: bool = True,
        threshold: float = 0.82,
    ) -> tuple[int, bool]:
        if not dedup_enabled:
            return self.add_memory(entry), False
        candidate = self.find_merge_candidate(entry, threshold=threshold)
        if not candidate or candidate.id is None:
            return self.add_memory(entry), False
        merged_content = self._merge_content(candidate.content, entry.content)
        merged_tags = self._merge_list(candidate.tags, entry.tags, limit=18)
        merged_metadata = dict(candidate.metadata or {})
        merged_metadata.setdefault("merge_history", [])
        history = merged_metadata.get("merge_history")
        if isinstance(history, list):
            history.append(
                {
                    "at": now_iso(),
                    "source": entry.source,
                    "category": entry.category,
                    "content": entry.content[:300],
                }
            )
            merged_metadata["merge_history"] = history[-12:]
        merged_metadata.update({k: v for k, v in (entry.metadata or {}).items() if k not in {"fallback_excerpt"}})
        self.update_memory(
            candidate.id,
            title=candidate.title or entry.title,
            content=merged_content,
            importance=max(candidate.importance, entry.importance),
            tags=merged_tags,
            metadata=merged_metadata,
            locked=candidate.locked or entry.locked,
        )
        return candidate.id, True

    def find_merge_candidate(self, entry: MemoryEntry, *, threshold: float = 0.82) -> MemoryEntry | None:
        if not entry.content.strip():
            return None
        rows = self.conn.execute(
            """
            SELECT * FROM memory_entries
            WHERE enabled=1 AND session_id=? AND category=?
            ORDER BY updated_at DESC, importance DESC
            LIMIT 80
            """,
            (entry.session_id or "", normalize_category(entry.category)),
        ).fetchall()
        best: tuple[float, MemoryEntry] | None = None
        for row in rows:
            current = self._row_to_entry(row)
            score = self._content_similarity(entry.content, current.content)
            if score >= threshold and (best is None or score > best[0]):
                best = (score, current)
        return best[1] if best else None

    def update_memory(
        self,
        memory_id: int,
        *,
        title: str | None = None,
        content: str | None = None,
        category: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        enabled: bool | None = None,
        locked: bool | None = None,
    ) -> bool:
        current = self.get_memory(memory_id)
        if not current:
            return False
        new_title = current.title if title is None else title
        new_content = current.content if content is None else content
        new_category = current.category if category is None else normalize_category(category, current.category)
        new_tags = current.tags if tags is None else tags
        new_metadata = current.metadata if metadata is None else metadata
        updates: dict[str, Any] = {
            "title": new_title,
            "content": new_content,
            "category": new_category,
            "importance": current.importance if importance is None else clamp01(importance),
            "tags_json": json.dumps(new_tags, ensure_ascii=False),
            "metadata_json": json.dumps(new_metadata, ensure_ascii=False),
            "enabled": int(current.enabled if enabled is None else enabled),
            "locked": int(current.locked if locked is None else locked or new_category == "locked"),
            "updated_at": now_iso(),
        }
        assignments = ", ".join(f"{key}=?" for key in updates)
        with self._lock:
            cur = self.conn.execute(
                f"UPDATE memory_entries SET {assignments} WHERE id=?",
                (*updates.values(), int(memory_id)),
            )
            self._upsert_fts(memory_id, new_title, new_content, " ".join(new_tags))
            self.conn.execute("DELETE FROM memory_vectors WHERE memory_id=?", (int(memory_id),))
            self.conn.commit()
            return cur.rowcount > 0

    def delete_memory(self, memory_id: int) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM memory_entries WHERE id=?", (int(memory_id),))
            self._delete_fts(memory_id)
            self.conn.execute("DELETE FROM memory_vectors WHERE memory_id=?", (int(memory_id),))
            self.conn.commit()
            return cur.rowcount > 0

    def clear_category(self, session_id: str, category: str | None = None) -> int:
        normalized = normalize_category(category or "", "") if category else ""
        with self._lock:
            if normalized:
                rows = self.conn.execute(
                    "SELECT id FROM memory_entries WHERE session_id=? AND category=?",
                    (session_id or "", normalized),
                ).fetchall()
                self.conn.execute(
                    "DELETE FROM memory_entries WHERE session_id=? AND category=?",
                    (session_id or "", normalized),
                )
            else:
                rows = self.conn.execute(
                    "SELECT id FROM memory_entries WHERE session_id=?",
                    (session_id or "",),
                ).fetchall()
                self.conn.execute(
                    "DELETE FROM memory_entries WHERE session_id=?",
                    (session_id or "",),
                )
            for row in rows:
                self._delete_fts(int(row["id"]))
            if rows:
                placeholders = ",".join("?" for _ in rows)
                self.conn.execute(
                    f"DELETE FROM memory_vectors WHERE memory_id IN ({placeholders})",
                    [int(row["id"]) for row in rows],
                )
            if not normalized:
                self.conn.execute("DELETE FROM story_state WHERE session_id=?", (session_id or "",))
            self.conn.commit()
            return len(rows)

    def get_memory(self, memory_id: int) -> MemoryEntry | None:
        row = self.conn.execute(
            "SELECT * FROM memory_entries WHERE id=?",
            (int(memory_id),),
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def list_memories(
        self,
        session_id: str,
        *,
        category: str | None = None,
        limit: int = 20,
        include_global: bool = True,
    ) -> list[MemoryEntry]:
        params: list[Any] = []
        filters = ["enabled=1"]
        if include_global:
            filters.append("(session_id=? OR session_id='')")
            params.append(session_id or "")
        else:
            filters.append("session_id=?")
            params.append(session_id or "")
        if category:
            filters.append("category=?")
            params.append(normalize_category(category))
        sql = f"""
            SELECT * FROM memory_entries
            WHERE {' AND '.join(filters)}
            ORDER BY locked DESC, importance DESC, updated_at DESC, id DESC
            LIMIT ?
        """
        params.append(max(1, int(limit)))
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_locked_memories(self, session_id: str, limit: int = 12) -> list[MemoryEntry]:
        if limit <= 0:
            return []
        rows = self.conn.execute(
            """
            SELECT * FROM memory_entries
            WHERE enabled=1 AND locked=1 AND (session_id=? OR session_id='')
            ORDER BY importance DESC, updated_at DESC, id DESC
            LIMIT ?
            """,
            (session_id or "", max(1, int(limit))),
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def search_memories(
        self,
        query: str,
        *,
        session_id: str,
        top_k: int = 6,
        categories: Iterable[str] | None = None,
        include_global: bool = True,
    ) -> list[MemoryEntry]:
        query = (query or "").strip()
        if not query or top_k <= 0:
            return []
        like_entries = self._search_like(query, session_id, top_k * 6, categories, include_global)
        fts_entries = self._search_fts(query, session_id, top_k * 6, categories, include_global)
        ranked = self.fuse_retrieval_results(
            [like_entries, fts_entries],
            top_k=max(1, int(top_k)),
            diversity_threshold=0.82,
        )
        result = ranked[: max(1, int(top_k))]
        if result:
            self.mark_accessed([item.id for item in result if item.id is not None])
        return result

    def vector_search_memories(
        self,
        query_vector: list[float],
        *,
        session_id: str,
        top_k: int = 6,
        categories: Iterable[str] | None = None,
        include_global: bool = True,
        provider_id: str = "",
    ) -> list[MemoryEntry]:
        if not query_vector or top_k <= 0:
            return []
        query_vector = [float(item) for item in query_vector if isinstance(item, (int, float))]
        if not query_vector:
            return []
        params: list[Any] = []
        filters = ["m.enabled=1"]
        if include_global:
            filters.append("(m.session_id=? OR m.session_id='')")
            params.append(session_id or "")
        else:
            filters.append("m.session_id=?")
            params.append(session_id or "")
        normalized_categories = [normalize_category(item) for item in categories or [] if item]
        if normalized_categories:
            placeholders = ",".join("?" for _ in normalized_categories)
            filters.append(f"m.category IN ({placeholders})")
            params.extend(normalized_categories)
        if provider_id:
            filters.append("v.provider_id=?")
            params.append(provider_id)
        sql = f"""
            SELECT m.*, v.vector_json
            FROM memory_vectors v
            JOIN memory_entries m ON v.memory_id=m.id
            WHERE {' AND '.join(filters)}
        """
        rows = self.conn.execute(sql, params).fetchall()
        scored: list[MemoryEntry] = []
        for row in rows:
            vector = self._json_loads(row["vector_json"], [])
            if not vector:
                continue
            score = self._cosine(query_vector, vector)
            if score <= 0:
                continue
            entry = self._row_to_entry(row)
            category_bonus = 0.08 if entry.category in {"core", "story_frame", "locked"} else 0.0
            entry.score = score * 3.0 + entry.importance * 0.7 + category_bonus
            scored.append(entry)
        scored.sort(key=lambda item: item.score, reverse=True)
        result = scored[: max(1, int(top_k))]
        if result:
            self.mark_accessed([item.id for item in result if item.id is not None])
        return result

    def fuse_retrieval_results(
        self,
        groups: Iterable[list[MemoryEntry]],
        *,
        top_k: int = 8,
        rrf_k: int = 60,
        diversity_threshold: float = 0.82,
    ) -> list[MemoryEntry]:
        """Fuse ranked retrieval routes with reciprocal rank fusion and MMR-style dedup."""
        fused = self._rrf_fuse(groups, limit=max(top_k * 4, top_k), rrf_k=rrf_k)
        return self.diversify_entries(fused, limit=max(1, int(top_k)), threshold=diversity_threshold)

    def diversify_entries(
        self,
        entries: list[MemoryEntry],
        *,
        limit: int,
        threshold: float = 0.82,
    ) -> list[MemoryEntry]:
        selected: list[MemoryEntry] = []
        for entry in entries:
            if entry.id is None:
                continue
            duplicate = False
            for chosen in selected:
                if entry.category == chosen.category and self._content_similarity(entry.content, chosen.content) >= threshold:
                    duplicate = True
                    break
            if duplicate:
                continue
            selected.append(entry)
            if len(selected) >= max(1, int(limit)):
                break
        return selected

    def _search_like(
        self,
        query: str,
        session_id: str,
        limit: int,
        categories: Iterable[str] | None,
        include_global: bool,
    ) -> list[MemoryEntry]:
        terms = self._query_terms(query)
        params: list[Any] = []
        filters = ["enabled=1"]
        if include_global:
            filters.append("(session_id=? OR session_id='')")
            params.append(session_id or "")
        else:
            filters.append("session_id=?")
            params.append(session_id or "")
        normalized_categories = [normalize_category(item) for item in categories or [] if item]
        if normalized_categories:
            placeholders = ",".join("?" for _ in normalized_categories)
            filters.append(f"category IN ({placeholders})")
            params.extend(normalized_categories)
        like_filters = []
        for term in terms:
            like_filters.append("(title LIKE ? OR content LIKE ? OR tags_json LIKE ?)")
            pattern = f"%{term}%"
            params.extend([pattern, pattern, pattern])
        if like_filters:
            filters.append("(" + " OR ".join(like_filters) + ")")
        sql = f"""
            SELECT * FROM memory_entries
            WHERE {' AND '.join(filters)}
            ORDER BY locked DESC, importance DESC, access_count DESC, updated_at DESC
            LIMIT ?
        """
        params.append(max(1, int(limit)))
        rows = self.conn.execute(sql, params).fetchall()
        entries = [self._row_to_entry(row) for row in rows]
        for entry in entries:
            text = f"{entry.title} {entry.content} {' '.join(entry.tags)}".lower()
            hit_count = sum(1 for term in terms if term.lower() in text)
            category_bonus = 0.2 if entry.category in {"locked", "core", "story_frame"} else 0.0
            entry.score = hit_count * 1.0 + entry.importance * 0.8 + entry.confidence * 0.2 + category_bonus
        return sorted(entries, key=lambda item: item.score, reverse=True)

    def _search_fts(
        self,
        query: str,
        session_id: str,
        limit: int,
        categories: Iterable[str] | None,
        include_global: bool,
    ) -> list[MemoryEntry]:
        if not self._fts_available:
            return []
        terms = [term for term in self._query_terms(query) if self._is_fts_term(term)]
        if not terms:
            return []
        match_query = " OR ".join(term.replace('"', '""') for term in terms[:8])
        params: list[Any] = [match_query]
        filters = ["m.enabled=1"]
        if include_global:
            filters.append("(m.session_id=? OR m.session_id='')")
            params.append(session_id or "")
        else:
            filters.append("m.session_id=?")
            params.append(session_id or "")
        normalized_categories = [normalize_category(item) for item in categories or [] if item]
        if normalized_categories:
            placeholders = ",".join("?" for _ in normalized_categories)
            filters.append(f"m.category IN ({placeholders})")
            params.extend(normalized_categories)
        sql = f"""
            SELECT m.*, bm25(memory_entries_fts) AS rank
            FROM memory_entries_fts
            JOIN memory_entries m ON memory_entries_fts.rowid=m.id
            WHERE memory_entries_fts MATCH ? AND {' AND '.join(filters)}
            ORDER BY rank ASC
            LIMIT ?
        """
        params.append(max(1, int(limit)))
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        entries = [self._row_to_entry(row) for row in rows]
        for index, entry in enumerate(entries):
            entry.score = 2.0 + max(0.0, 1.0 - index * 0.05) + entry.importance * 0.5
        return entries

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        sender_id: str = "",
        sender_name: str = "",
    ) -> int:
        if not content.strip():
            return 0
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO conversation_messages
                (session_id, role, content, sender_id, sender_name, summarized, created_at)
                VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    session_id or "",
                    role,
                    content.strip(),
                    sender_id or "",
                    sender_name or "",
                    now_iso(),
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def get_unsummarized_messages(self, session_id: str, limit: int = 80) -> list[sqlite3.Row]:
        state = self.conn.execute(
            "SELECT last_summarized_id FROM session_state WHERE session_id=?",
            (session_id or "",),
        ).fetchone()
        last_id = int(state["last_summarized_id"]) if state else 0
        return self.conn.execute(
            """
            SELECT * FROM conversation_messages
            WHERE session_id=? AND id>?
            ORDER BY id ASC
            LIMIT ?
            """,
            (session_id or "", last_id, max(1, int(limit))),
        ).fetchall()

    def count_unsummarized_messages(self, session_id: str) -> int:
        state = self.conn.execute(
            "SELECT last_summarized_id FROM session_state WHERE session_id=?",
            (session_id or "",),
        ).fetchone()
        last_id = int(state["last_summarized_id"]) if state else 0
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM conversation_messages WHERE session_id=? AND id>?",
            (session_id or "", last_id),
        ).fetchone()
        return int(row["c"] if row else 0)

    def mark_summarized(self, session_id: str, last_message_id: int) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO session_state(session_id, last_summarized_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_summarized_id=excluded.last_summarized_id,
                    updated_at=excluded.updated_at
                """,
                (session_id or "", int(last_message_id), now_iso()),
            )
            self.conn.execute(
                "UPDATE conversation_messages SET summarized=1 WHERE session_id=? AND id<=?",
                (session_id or "", int(last_message_id)),
            )
            self.conn.commit()

    def trim_messages(self, session_id: str, keep_latest: int) -> int:
        keep_latest = max(20, int(keep_latest))
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id FROM conversation_messages
                WHERE session_id=?
                ORDER BY id DESC
                LIMIT -1 OFFSET ?
                """,
                (session_id or "", keep_latest),
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            if not ids:
                return 0
            placeholders = ",".join("?" for _ in ids)
            self.conn.execute(
                f"DELETE FROM conversation_messages WHERE id IN ({placeholders})",
                ids,
            )
            self.conn.commit()
            return len(ids)

    def stats(self, session_id: str) -> dict[str, Any]:
        rows = self.conn.execute(
            """
            SELECT category, COUNT(*) AS c
            FROM memory_entries
            WHERE enabled=1 AND (session_id=? OR session_id='')
            GROUP BY category
            """,
            (session_id or "",),
        ).fetchall()
        total = sum(int(row["c"]) for row in rows)
        return {
            "total": total,
            "by_category": {row["category"]: int(row["c"]) for row in rows},
            "unsummarized_messages": self.count_unsummarized_messages(session_id),
            "fts_available": self._fts_available,
            "vectors": self.count_vectors(session_id),
            "story_state": bool(self.get_story_state(session_id)),
            "low_quality": self.count_low_quality_memories(session_id),
        }

    def count_low_quality_memories(self, session_id: str = "") -> int:
        params: list[Any] = []
        filters = ["enabled=1", "metadata_json LIKE ?"]
        params.append('%"summary_quality": "low"%')
        if session_id:
            filters.append("(session_id=? OR session_id='')")
            params.append(session_id or "")
        row = self.conn.execute(
            f"SELECT COUNT(*) AS c FROM memory_entries WHERE {' AND '.join(filters)}",
            params,
        ).fetchone()
        return int(row["c"] if row else 0)

    def export_memories(self, session_id: str, category: str | None = None) -> list[dict[str, Any]]:
        entries = self.list_memories(session_id, category=category, limit=500, include_global=True)
        return [self.entry_to_dict(entry) for entry in entries]

    def rebuild_index(self) -> int:
        with self._lock:
            if not self._fts_available:
                self._init_fts()
            if not self._fts_available:
                return 0
            self.conn.execute("DELETE FROM memory_entries_fts")
            rows = self.conn.execute("SELECT * FROM memory_entries").fetchall()
            for row in rows:
                tags = " ".join(self._json_loads(row["tags_json"], []))
                self._upsert_fts(int(row["id"]), row["title"], row["content"], tags)
            self.conn.commit()
            return len(rows)

    def set_memory_vector(
        self,
        memory_id: int,
        *,
        provider_id: str,
        vector: list[float],
        content: str,
    ) -> None:
        clean_vector = [float(item) for item in vector if isinstance(item, (int, float))]
        if not clean_vector:
            return
        content_hash = hashlib.sha1((content or "").encode("utf-8")).hexdigest()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO memory_vectors(memory_id, provider_id, dim, vector_json, content_hash, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    provider_id=excluded.provider_id,
                    dim=excluded.dim,
                    vector_json=excluded.vector_json,
                    content_hash=excluded.content_hash,
                    updated_at=excluded.updated_at
                """,
                (
                    int(memory_id),
                    provider_id or "",
                    len(clean_vector),
                    json.dumps(clean_vector, separators=(",", ":")),
                    content_hash,
                    now_iso(),
                ),
            )
            self.conn.commit()

    def count_vectors(self, session_id: str = "") -> int:
        if session_id:
            row = self.conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM memory_vectors v
                JOIN memory_entries m ON m.id=v.memory_id
                WHERE m.session_id=? OR m.session_id=''
                """,
                (session_id or "",),
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) AS c FROM memory_vectors").fetchone()
        return int(row["c"] if row else 0)

    def vector_missing_entries(self, limit: int = 100, provider_id: str = "") -> list[MemoryEntry]:
        provider_id = provider_id or ""
        provider_filter = "AND (v.memory_id IS NULL OR v.provider_id<>?)" if provider_id else "AND v.memory_id IS NULL"
        params: list[Any] = []
        if provider_id:
            params.append(provider_id)
        params.append(max(1, int(limit)))
        rows = self.conn.execute(
            f"""
            SELECT m.*
            FROM memory_entries m
            LEFT JOIN memory_vectors v ON v.memory_id=m.id
            WHERE m.enabled=1 {provider_filter}
            ORDER BY m.importance DESC, m.updated_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def upsert_story_state(self, session_id: str, persona_id: str, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or not any(state.values()):
            return
        current = self.get_story_state(session_id)
        merged = self._merge_story_state(current, state)
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO story_state(session_id, persona_id, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    persona_id=excluded.persona_id,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (
                    session_id or "",
                    persona_id or "",
                    json.dumps(merged, ensure_ascii=False),
                    now_iso(),
                ),
            )
            self.conn.commit()

    def get_story_state(self, session_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT state_json FROM story_state WHERE session_id=?",
            (session_id or "",),
        ).fetchone()
        if not row:
            return {}
        value = self._json_loads(row["state_json"], {})
        return value if isinstance(value, dict) else {}

    def mark_accessed(self, ids: Iterable[int]) -> None:
        ids = [int(item) for item in ids if item]
        if not ids:
            return
        with self._lock:
            for memory_id in ids:
                self.conn.execute(
                    """
                    UPDATE memory_entries
                    SET access_count=access_count+1, last_accessed_at=?
                    WHERE id=?
                    """,
                    (now_iso(), memory_id),
                )
            self.conn.commit()

    def maintain_memories(
        self,
        session_id: str = "",
        *,
        stale_days: int = 60,
        disable_below_importance: float = 0.18,
        preview: bool = True,
        limit: int = 200,
    ) -> dict[str, Any]:
        cutoff = datetime.now() - timedelta(days=max(1, int(stale_days)))
        params: list[Any] = []
        filters = ["enabled=1", "locked=0", "source!='manual'"]
        if session_id:
            filters.append("(session_id=? OR session_id='')")
            params.append(session_id or "")
        rows = self.conn.execute(
            f"""
            SELECT * FROM memory_entries
            WHERE {' AND '.join(filters)}
            ORDER BY updated_at ASC
            LIMIT ?
            """,
            (*params, max(1, int(limit))),
        ).fetchall()
        candidates: list[MemoryEntry] = []
        for row in rows:
            entry = self._row_to_entry(row)
            updated_at = self._parse_iso(entry.updated_at or entry.created_at)
            if not updated_at or updated_at > cutoff:
                continue
            quality = str((entry.metadata or {}).get("summary_quality") or "")
            if entry.access_count > 0 and quality != "low":
                continue
            candidates.append(entry)
        disabled = 0
        decayed = 0
        if not preview:
            with self._lock:
                for entry in candidates:
                    if entry.id is None:
                        continue
                    new_importance = max(0.0, entry.importance - (0.15 if entry.metadata.get("summary_quality") == "low" else 0.08))
                    if new_importance <= disable_below_importance:
                        self.conn.execute("UPDATE memory_entries SET enabled=0, updated_at=? WHERE id=?", (now_iso(), entry.id))
                        self.conn.execute("DELETE FROM memory_vectors WHERE memory_id=?", (entry.id,))
                        self._delete_fts(entry.id)
                        disabled += 1
                    else:
                        self.conn.execute(
                            "UPDATE memory_entries SET importance=?, updated_at=? WHERE id=?",
                            (new_importance, now_iso(), entry.id),
                        )
                        decayed += 1
                self.conn.commit()
        return {
            "preview": preview,
            "candidates": len(candidates),
            "disabled": disabled,
            "decayed": decayed,
            "ids": [entry.id for entry in candidates[:20] if entry.id is not None],
        }

    def _upsert_fts(self, row_id: int, title: str, content: str, tags: str) -> None:
        if not self._fts_available:
            return
        try:
            self.conn.execute("DELETE FROM memory_entries_fts WHERE rowid=?", (row_id,))
            self.conn.execute(
                "INSERT INTO memory_entries_fts(rowid, title, content, tags) VALUES (?, ?, ?, ?)",
                (row_id, title or "", content or "", tags or ""),
            )
        except sqlite3.OperationalError:
            self._fts_available = False

    def _delete_fts(self, row_id: int) -> None:
        if not self._fts_available:
            return
        try:
            self.conn.execute("DELETE FROM memory_entries_fts WHERE rowid=?", (row_id,))
        except sqlite3.OperationalError:
            self._fts_available = False

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=int(row["id"]),
            session_id=row["session_id"] or "",
            persona_id=row["persona_id"] or "",
            category=row["category"] or "core",
            title=row["title"] or "",
            content=row["content"] or "",
            importance=clamp01(row["importance"]),
            confidence=clamp01(row["confidence"], 0.8),
            tags=self._json_loads(row["tags_json"], []),
            metadata=self._json_loads(row["metadata_json"], {}),
            source=row["source"] or "",
            enabled=bool(row["enabled"]),
            locked=bool(row["locked"]),
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
            last_accessed_at=row["last_accessed_at"] or "",
            access_count=int(row["access_count"] or 0),
        )

    @staticmethod
    def entry_to_dict(entry: MemoryEntry) -> dict[str, Any]:
        return {
            "id": entry.id,
            "session_id": entry.session_id,
            "persona_id": entry.persona_id,
            "category": entry.category,
            "category_name": entry.display_category(),
            "title": entry.title,
            "content": entry.content,
            "importance": entry.importance,
            "confidence": entry.confidence,
            "tags": entry.tags,
            "metadata": entry.metadata,
            "source": entry.source,
            "locked": entry.locked,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "last_accessed_at": entry.last_accessed_at,
            "access_count": entry.access_count,
        }

    @staticmethod
    def _json_loads(text: str, default: Any) -> Any:
        try:
            value = json.loads(text or "")
        except Exception:
            return default
        return value if isinstance(value, type(default)) else default

    @classmethod
    def _rrf_fuse(
        cls,
        groups: Iterable[list[MemoryEntry]],
        *,
        limit: int,
        rrf_k: int = 60,
    ) -> list[MemoryEntry]:
        entries: dict[int, MemoryEntry] = {}
        scores: dict[int, float] = {}
        for group in groups:
            for rank, entry in enumerate(group or [], start=1):
                if entry.id is None:
                    continue
                memory_id = int(entry.id)
                entries.setdefault(memory_id, entry)
                scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (max(1, int(rrf_k)) + rank)
        ranked: list[MemoryEntry] = []
        for memory_id, entry in entries.items():
            category_bonus = 0.08 if entry.category in {"core", "story_frame", "locked"} else 0.0
            access_bonus = min(max(entry.access_count, 0), 20) * 0.005
            lock_bonus = 0.2 if entry.locked else 0.0
            entry.score = scores.get(memory_id, 0.0) * 100.0 + entry.importance * 0.35 + entry.confidence * 0.12 + category_bonus + access_bonus + lock_bonus
            ranked.append(entry)
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: max(1, int(limit))]

    @classmethod
    def _content_similarity(cls, left: str, right: str) -> float:
        left = re.sub(r"\s+", "", left or "")
        right = re.sub(r"\s+", "", right or "")
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        if left in right or right in left:
            return min(len(left), len(right)) / max(len(left), len(right))
        left_set = cls._char_ngrams(left)
        right_set = cls._char_ngrams(right)
        if not left_set or not right_set:
            return 0.0
        return len(left_set & right_set) / len(left_set | right_set)

    @staticmethod
    def _parse_iso(value: str) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @staticmethod
    def _char_ngrams(text: str, n: int = 3) -> set[str]:
        if len(text) <= n:
            return {text}
        return {text[i : i + n] for i in range(0, len(text) - n + 1)}

    @staticmethod
    def _merge_content(left: str, right: str, max_chars: int = 1800) -> str:
        left = (left or "").strip()
        right = (right or "").strip()
        if not right or right in left:
            return left[:max_chars]
        if not left:
            return right[:max_chars]
        merged = f"{left}\n补充：{right}"
        return merged[:max_chars].rstrip()

    @staticmethod
    def _merge_list(left: list[str], right: list[str], limit: int = 18) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in [*(left or []), *(right or [])]:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            result.append(text)
            seen.add(text)
            if len(result) >= limit:
                break
        return result

    @classmethod
    def _merge_story_state(cls, current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        result = dict(current or {})
        for key, value in (incoming or {}).items():
            if value in ("", None, [], {}):
                continue
            if isinstance(value, list):
                result[key] = cls._merge_list([str(x) for x in result.get(key, []) if x], [str(x) for x in value if x], limit=24)
            elif isinstance(value, dict):
                nested = dict(result.get(key) or {})
                nested.update(value)
                result[key] = nested
            else:
                result[key] = value
        result["updated_at"] = now_iso()
        return result

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        dim = min(len(left), len(right))
        if dim <= 0:
            return 0.0
        dot = sum(float(left[i]) * float(right[i]) for i in range(dim))
        norm_l = math.sqrt(sum(float(left[i]) * float(left[i]) for i in range(dim)))
        norm_r = math.sqrt(sum(float(right[i]) * float(right[i]) for i in range(dim)))
        if norm_l <= 0 or norm_r <= 0:
            return 0.0
        return dot / (norm_l * norm_r)

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        compact = re.sub(r"\s+", " ", query.strip())
        parts = re.findall(r"[\w\u4e00-\u9fff]{2,}", compact)
        if compact and compact not in parts and len(compact) <= 80:
            parts.insert(0, compact)
        seen: set[str] = set()
        result: list[str] = []
        for part in parts:
            item = part.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
            if len(result) >= 12:
                break
        return result or [query[:80]]

    @staticmethod
    def _is_fts_term(term: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9_]{2,}", term))
