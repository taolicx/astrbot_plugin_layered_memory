from __future__ import annotations

from collections import defaultdict

from .schema import (
    INJECTION_FOOTER,
    INJECTION_HEADER,
    LOCKED_FOOTER,
    LOCKED_HEADER,
    STORY_FOOTER,
    STORY_HEADER,
    MEMORY_CATEGORIES,
    MemoryEntry,
)


def format_recall_context(entries: list[MemoryEntry], max_chars: int = 2200) -> str:
    if not entries:
        return ""
    grouped: dict[str, list[MemoryEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.category].append(entry)

    lines = [
        INJECTION_HEADER,
        "以下是从长期记忆系统中自然想起的相关背景。回复时只吸收其含义，不要机械复述，不要说“根据记忆”。",
    ]
    order = ["core", "memo", "story_frame", "story_summary", "log", "locked"]
    for category in order:
        items = grouped.get(category)
        if not items:
            continue
        lines.append(f"\n[{MEMORY_CATEGORIES.get(category, category)}]")
        for item in items:
            title = f"{item.title}: " if item.title else ""
            lines.append(f"- #{item.id} {title}{_entry_injection_text(item)}")
    lines.append(INJECTION_FOOTER)
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[: max(200, max_chars - len(INJECTION_FOOTER) - 8)].rstrip() + "\n...\n" + INJECTION_FOOTER


def format_story_state(state: dict, max_chars: int = 1400) -> str:
    if not isinstance(state, dict) or not any(value for value in state.values() if value not in ("", [], {}, None)):
        return ""
    labels = {
        "current_stage": "当前阶段",
        "world_state": "世界观/环境",
        "important_events": "已发生事件",
        "relationships": "关系变化",
        "unresolved_conflicts": "未解决冲突",
        "short_term_goals": "短期目标",
        "long_term_goals": "长期目标",
        "turning_points": "关键转折",
        "next_hooks": "后续线索",
    }
    lines = [
        STORY_HEADER,
        "以下是持续维护的剧情状态。回复时要优先保持主线、关系、冲突和未完成线索的一致性。",
    ]
    for key in labels:
        value = state.get(key)
        if value in ("", [], {}, None):
            continue
        if isinstance(value, list):
            text = "；".join(str(item) for item in value[:8] if str(item).strip())
        elif isinstance(value, dict):
            text = "；".join(f"{k}: {v}" for k, v in list(value.items())[:8] if str(v).strip())
        else:
            text = str(value)
        if text:
            lines.append(f"- {labels[key]}：{text}")
    lines.append(STORY_FOOTER)
    result = "\n".join(lines)
    if len(result) <= max_chars:
        return result
    return result[: max(200, max_chars - len(STORY_FOOTER) - 8)].rstrip() + "\n...\n" + STORY_FOOTER


def format_locked_context(entries: list[MemoryEntry], max_chars: int = 1600) -> str:
    if not entries:
        return ""
    lines = [
        LOCKED_HEADER,
        "以下是最高优先级的锁定记忆/硬性约束。回复必须遵守；如果与普通记忆冲突，以这里为准。",
    ]
    for item in entries:
        title = f"{item.title}: " if item.title else ""
        lines.append(f"- #{item.id} {title}{item.content}")
    lines.append(LOCKED_FOOTER)
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[: max(200, max_chars - len(LOCKED_FOOTER) - 8)].rstrip() + "\n...\n" + LOCKED_FOOTER


def format_entry_line(entry: MemoryEntry) -> str:
    title = f"{entry.title} | " if entry.title else ""
    locked = " | 锁定" if entry.locked else ""
    tags = f" | tags={','.join(entry.tags[:5])}" if entry.tags else ""
    return (
        f"#{entry.id} [{entry.display_category()}] {title}"
        f"{entry.content} (重要度 {entry.importance:.2f}{locked}{tags})"
    )


def format_entries(entries: list[MemoryEntry]) -> str:
    if not entries:
        return "没有找到记忆。"
    return "\n".join(format_entry_line(entry) for entry in entries)


def _entry_injection_text(entry: MemoryEntry) -> str:
    persona_summary = ""
    if isinstance(entry.metadata, dict):
        raw = entry.metadata.get("persona_summary")
        if isinstance(raw, str):
            persona_summary = raw.strip()
    return persona_summary or entry.content
