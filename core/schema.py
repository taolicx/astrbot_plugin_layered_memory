from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


PLUGIN_NAME = "astrbot_plugin_layered_memory"
PLUGIN_COMMAND = "rmem"


MEMORY_CATEGORIES: dict[str, str] = {
    "core": "核心记忆",
    "memo": "备忘录",
    "locked": "锁定记忆",
    "log": "记忆日志",
    "story_frame": "剧情框架",
    "story_summary": "剧情总结",
}


CATEGORY_ALIASES: dict[str, str] = {
    "核心": "core",
    "核心记忆": "core",
    "core": "core",
    "备忘": "memo",
    "备忘录": "memo",
    "memo": "memo",
    "短期": "memo",
    "中期": "memo",
    "锁定": "locked",
    "锁定记忆": "locked",
    "locked": "locked",
    "规则": "locked",
    "日志": "log",
    "日记": "log",
    "记忆日志": "log",
    "log": "log",
    "剧情": "story_frame",
    "剧情框架": "story_frame",
    "框架": "story_frame",
    "story": "story_frame",
    "frame": "story_frame",
    "剧情总结": "story_summary",
    "总结": "story_summary",
    "summary": "story_summary",
}


INJECTION_HEADER = "<LayeredMemory>"
INJECTION_FOOTER = "</LayeredMemory>"
LOCKED_HEADER = "<LockedMemory>"
LOCKED_FOOTER = "</LockedMemory>"
STORY_HEADER = "<StoryContinuity>"
STORY_FOOTER = "</StoryContinuity>"


@dataclass
class MemoryEntry:
    id: int | None = None
    session_id: str = ""
    persona_id: str = ""
    category: str = "core"
    title: str = ""
    content: str = ""
    importance: float = 0.5
    confidence: float = 0.8
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "manual"
    enabled: bool = True
    locked: bool = False
    created_at: str = ""
    updated_at: str = ""
    last_accessed_at: str = ""
    access_count: int = 0
    score: float = 0.0

    def display_category(self) -> str:
        return MEMORY_CATEGORIES.get(self.category, self.category)


@dataclass
class LayeredSummary:
    entries: list[MemoryEntry] = field(default_factory=list)
    story_state: dict[str, Any] = field(default_factory=dict)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_category(value: str, default: str = "core") -> str:
    text = str(value or "").strip().lower()
    return CATEGORY_ALIASES.get(text, default)


def clamp01(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def ensure_str_list(value: Any, limit: int = 12) -> list[str]:
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        result.append(text[:80])
        seen.add(text)
        if len(result) >= limit:
            break
    return result
