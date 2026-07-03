from __future__ import annotations

import asyncio
import json
import random
import re
from datetime import datetime
from typing import Any

from .schema import LayeredSummary, MemoryEntry, clamp01, ensure_str_list, normalize_category


SUMMARY_SCHEMA_VERSION = "layered-v2"


class LayeredMemoryProcessor:
    """LLM-backed extractor for layered memories and story continuity."""

    def __init__(self, context: Any = None, provider_id: str = "", allow_auto_locked: bool = False):
        self.context = context
        self.provider_id = provider_id or ""
        self.allow_auto_locked = allow_auto_locked

    def _get_provider(self, session_id: str = "") -> Any:
        if not self.context:
            return None
        if self.provider_id:
            try:
                provider = self.context.get_provider_by_id(self.provider_id)
                if provider:
                    return provider
            except Exception:
                pass
        try:
            return self.context.get_using_provider(session_id) if session_id else self.context.get_using_provider()
        except TypeError:
            try:
                return self.context.get_using_provider()
            except Exception:
                return None
        except Exception:
            return None

    async def summarize_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str,
        persona_id: str = "",
        is_group: bool = False,
    ) -> LayeredSummary:
        if not messages:
            return LayeredSummary()
        conversation = self._format_messages(messages)
        source_window = self._source_window(messages, session_id=session_id)
        prompt = self._build_prompt(conversation, is_group=is_group)
        system_prompt = self._build_system_prompt(persona_id=persona_id)
        text = await self._call_llm(prompt, system_prompt, session_id=session_id)
        payload = self._parse_json(text)
        entries = self._payload_to_entries(
            payload,
            session_id=session_id,
            persona_id=persona_id,
            fallback_excerpt=conversation[:400],
            source_window=source_window,
        )
        story_state = self._payload_story_state(payload)
        return LayeredSummary(entries=entries, story_state=story_state)

    async def _call_llm(self, prompt: str, system_prompt: str, *, session_id: str) -> str:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                provider = self._get_provider(session_id)
                if provider is None:
                    raise RuntimeError("LLM Provider 不可用")
                response = await provider.text_chat(prompt=prompt, system_prompt=system_prompt)
                return getattr(response, "completion_text", str(response))
            except Exception as exc:
                last_error = exc
                if attempt >= 2:
                    break
                await asyncio.sleep((2**attempt) + random.random())
        if last_error:
            raise last_error
        raise RuntimeError("LLM 调用失败")

    def _build_system_prompt(self, *, persona_id: str = "") -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            "你是 AstrBot 插件里的长期记忆整理器。只输出 JSON，不要输出 Markdown。\n"
            f"当前时间：{now}\n"
            "请把相对时间（今天、明天、昨天、下周等）转换成具体日期或明确时间范围。\n"
            "不要编造事实；没有明确证据就留空数组。把用户事实、角色设定、剧情事实分清楚。\n"
            f"当前人格 ID：{persona_id or 'unknown'}"
        )

    def _build_prompt(self, conversation: str, *, is_group: bool) -> str:
        scope = "群聊" if is_group else "私聊"
        return f"""
请分析下面这段{scope}对话，提取对未来回复有帮助的长期记忆和剧情连续性信息。

输出 JSON，字段必须是：
{{
  "core_memories": [
    {{"title": "短标题", "content": "用户偏好/重要经历/约定/角色核心设定/关键转折", "canonical_summary": "事实化摘要", "persona_summary": "适合注入给角色自然理解的摘要", "key_facts": ["独立关键事实"], "importance": 0.0-1.0, "confidence": 0.0-1.0, "tags": ["关键词"]}}
  ],
  "memos": [
    {{"title": "短标题", "content": "称呼要求、说话方式、待办、小设定、后续线索", "canonical_summary": "事实化摘要", "persona_summary": "适合注入给角色自然理解的摘要", "key_facts": ["独立关键事实"], "importance": 0.0-1.0, "confidence": 0.0-1.0, "tags": ["关键词"]}}
  ],
  "locked_memories": [
    {{"title": "短标题", "content": "只有明确出现绝对禁止、必须遵守、不可更改、严重雷点时才填写", "canonical_summary": "事实化摘要", "persona_summary": "适合注入给角色自然理解的摘要", "key_facts": ["独立关键事实"], "importance": 0.0-1.0, "confidence": 0.0-1.0, "tags": ["关键词"]}}
  ],
  "memory_logs": [
    {{"title": "短标题", "content": "最近发生的剧情/聊天概要/新人物/有趣细节", "canonical_summary": "事实化摘要", "persona_summary": "适合注入给角色自然理解的摘要", "key_facts": ["独立关键事实"], "importance": 0.0-1.0, "confidence": 0.0-1.0, "tags": ["关键词"]}}
  ],
  "story_frames": [
    {{"title": "短标题", "content": "当前剧情阶段、世界观、已发生事件、关系变化、未解决冲突、目标、转折点", "canonical_summary": "事实化摘要", "persona_summary": "适合注入给角色自然理解的摘要", "key_facts": ["独立关键事实"], "importance": 0.0-1.0, "confidence": 0.0-1.0, "tags": ["关键词"]}}
  ],
  "story_summaries": [
    {{"title": "短标题", "content": "阶段性剧情摘要", "canonical_summary": "事实化摘要", "persona_summary": "适合注入给角色自然理解的摘要", "key_facts": ["独立关键事实"], "importance": 0.0-1.0, "confidence": 0.0-1.0, "tags": ["关键词"]}}
  ],
  "story_state": {{
    "current_stage": "当前剧情阶段，一句话",
    "world_state": ["稳定世界观/环境/势力/规则"],
    "important_events": ["已发生的重要事件"],
    "relationships": ["角色关系变化"],
    "unresolved_conflicts": ["尚未解决的矛盾冲突"],
    "short_term_goals": ["接下来几轮要做什么"],
    "long_term_goals": ["长期主线目标"],
    "turning_points": ["关键转折点"],
    "next_hooks": ["后续需要接上的线索"]
  }}
}}

提取规则：
- 每类最多 3 条；无内容输出空数组。
- 只记录稳定、可复用、未来需要记住的信息。
- 普通闲聊、客套、模型自己的临时措辞不要写入。
- 锁定记忆门槛最高：只有用户清晰表达“绝对不能/必须/永远/禁止/雷点/底线”等强约束时才写。
- 剧情内容要保留主线和未解决线索，避免只写泛泛摘要。
- canonical_summary 用客观事实写；persona_summary 用更自然、更适合角色理解的表达，但不要添油加醋。
- key_facts 写成可独立检索的事实短句，避免泛泛而谈。
- confidence 表示证据明确度；推断、模糊、玩笑内容要降低 confidence。
- story_state 只写稳定剧情状态，不要写普通闲聊；没有剧情就留空对象或空数组。

对话：
{conversation}
""".strip()

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for item in messages:
            role = str(item.get("role") or "user")
            name = str(item.get("sender_name") or item.get("sender_id") or role)
            content = str(item.get("content") or "").strip()
            created_at = str(item.get("created_at") or "")
            if content:
                lines.append(f"[{created_at}] {name}({role}): {content}")
        return "\n".join(lines)

    def _parse_json(self, text: str) -> dict[str, Any]:
        cleaned = (text or "").strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _payload_to_entries(
        self,
        payload: dict[str, Any],
        *,
        session_id: str,
        persona_id: str,
        fallback_excerpt: str,
        source_window: dict[str, Any],
    ) -> list[MemoryEntry]:
        mapping = {
            "core_memories": "core",
            "memos": "memo",
            "locked_memories": "locked",
            "memory_logs": "log",
            "story_frames": "story_frame",
            "story_summaries": "story_summary",
        }
        entries: list[MemoryEntry] = []
        for field, category in mapping.items():
            values = payload.get(field, [])
            if not isinstance(values, list):
                continue
            for raw in values[:3]:
                if not isinstance(raw, dict):
                    continue
                content = str(raw.get("canonical_summary") or raw.get("content") or raw.get("summary") or "").strip()
                if not content:
                    continue
                persona_summary = str(raw.get("persona_summary") or raw.get("content") or content).strip()
                key_facts = ensure_str_list(raw.get("key_facts"), limit=8)
                quality = self._validate_entry_quality(content, key_facts)
                confidence = clamp01(raw.get("confidence"), 0.78)
                if quality == "low":
                    confidence = min(confidence, 0.45)
                actual_category = normalize_category(category)
                locked = actual_category == "locked"
                if locked and not self.allow_auto_locked:
                    actual_category = "core"
                    locked = False
                    tags = ensure_str_list(raw.get("tags"), limit=10)
                    tags.append("锁定候选")
                else:
                    tags = ensure_str_list(raw.get("tags"), limit=10)
                entries.append(
                    MemoryEntry(
                        session_id=session_id,
                        persona_id=persona_id,
                        category=actual_category,
                        title=str(raw.get("title") or "").strip()[:80],
                        content=content[:1600],
                        importance=clamp01(raw.get("importance"), 0.55),
                        confidence=confidence,
                        tags=tags,
                        metadata={
                            "schema": SUMMARY_SCHEMA_VERSION,
                            "source_field": field,
                            "fallback_excerpt": fallback_excerpt,
                            "canonical_summary": content[:1600],
                            "persona_summary": persona_summary[:1600],
                            "key_facts": key_facts,
                            "summary_quality": quality,
                            "source_window": source_window,
                        },
                        source="auto_summary",
                        locked=locked,
                    )
                )
        return entries

    @staticmethod
    def _source_window(messages: list[dict[str, Any]], *, session_id: str) -> dict[str, Any]:
        ids = []
        times = []
        for item in messages:
            try:
                ids.append(int(item.get("id")))
            except (TypeError, ValueError):
                pass
            created_at = str(item.get("created_at") or "")
            if created_at:
                times.append(created_at)
        return {
            "session_id": session_id,
            "start_id": min(ids) if ids else None,
            "end_id": max(ids) if ids else None,
            "message_count": len(messages),
            "start_time": times[0] if times else "",
            "end_time": times[-1] if times else "",
        }

    @staticmethod
    def _validate_entry_quality(content: str, key_facts: list[str]) -> str:
        text = re.sub(r"\s+", "", content or "")
        if len(text) < 8:
            return "low"
        vague_markers = [
            "用户聊了一些事情",
            "进行了一些交流",
            "表达了自己的想法",
            "内容比较普通",
            "没有明确",
            "一些信息",
        ]
        if any(marker in content for marker in vague_markers):
            return "low"
        if len(text) < 18 and not key_facts:
            return "low"
        return "normal"

    def _payload_story_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw = payload.get("story_state")
        if not isinstance(raw, dict):
            return {}
        allowed = {
            "current_stage",
            "world_state",
            "important_events",
            "relationships",
            "unresolved_conflicts",
            "short_term_goals",
            "long_term_goals",
            "turning_points",
            "next_hooks",
        }
        result: dict[str, Any] = {}
        for key in allowed:
            value = raw.get(key)
            if isinstance(value, list):
                result[key] = ensure_str_list(value, limit=12)
            elif isinstance(value, dict):
                result[key] = {str(k)[:80]: str(v)[:240] for k, v in value.items() if str(v).strip()}
            elif isinstance(value, str) and value.strip():
                result[key] = value.strip()[:500]
        return result
