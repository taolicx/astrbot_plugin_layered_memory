from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
try:
    from astrbot.api.event.filter import PermissionType, permission_type
except Exception:
    PermissionType = None

    def permission_type(_permission):  # type: ignore
        def decorator(func):
            return func

        return decorator

admin_permission = permission_type(PermissionType.ADMIN) if PermissionType else (lambda func: func)

from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools, register
try:
    from astrbot.core.agent.message import TextPart
except Exception:
    TextPart = None

try:
    from astrbot.api.platform import MessageType
except Exception:
    MessageType = None

from .core.formatter import format_entries, format_locked_context, format_recall_context, format_story_state
from .core.processor import LayeredMemoryProcessor
from .core.schema import (
    INJECTION_FOOTER,
    INJECTION_HEADER,
    LOCKED_FOOTER,
    LOCKED_HEADER,
    MEMORY_CATEGORIES,
    PLUGIN_COMMAND,
    PLUGIN_NAME,
    STORY_FOOTER,
    STORY_HEADER,
    MemoryEntry,
    clamp01,
    ensure_str_list,
    normalize_category,
    now_iso,
)
from .core.storage import LayeredMemoryStore


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    current = config
    for part in key.split("."):
        try:
            if isinstance(current, dict):
                current = current.get(part, default)
            else:
                current = current.get(part, default)
        except Exception:
            return default
        if current is None:
            return default
    return current


def _safe_int(value: Any, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


@register(
    PLUGIN_NAME,
    "Codex",
    "分层长期记忆与剧情延续插件：核心记忆、备忘录、锁定记忆、记忆日志、智能回忆、剧情框架与剧情总结。",
    "0.4.0",
)
class LayeredMemoryPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any]):
        super().__init__(context)
        self.context = context
        self.config = config or {}
        self.data_dir = Path(str(StarTools.get_data_dir(PLUGIN_NAME)))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = LayeredMemoryStore(self.data_dir / "layered_memory.db")
        self.processor = LayeredMemoryProcessor(
            context=context,
            provider_id=str(_cfg(self.config, "provider_settings.llm_provider_id", "") or ""),
            allow_auto_locked=bool(_cfg(self.config, "memory_generation.allow_auto_locked_memory", False)),
        )
        self.embedding_provider_id = str(_cfg(self.config, "provider_settings.embedding_provider_id", "") or "")
        self._background_tasks: set[asyncio.Task] = set()
        self._summarizing_sessions: set[str] = set()
        self._summary_lock = asyncio.Lock()
        self.export_dir = self._resolve_export_dir()
        logger.info("[LayeredMemory] 插件已加载，数据目录：%s", self.data_dir)

    def _resolve_export_dir(self) -> Path:
        configured = str(_cfg(self.config, "storage.export_dir", "") or "").strip()
        if configured:
            path = Path(configured)
        elif Path("D:/创建文件").exists():
            path = Path("D:/创建文件/astrbot_layered_memory_exports")
        else:
            path = self.data_dir / "exports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _track_task(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    # -------------------- LLM hooks --------------------

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if not bool(_cfg(self.config, "plugin_enabled", True)):
            return
        session_id = getattr(event, "unified_msg_origin", "") or ""
        try:
            user_text = await self._event_text(event)
            prompt_text = str(getattr(req, "prompt", "") or "").strip()
            query = user_text or prompt_text
            if bool(_cfg(self.config, "session_capture.capture_user_messages", True)) and query:
                self.store.add_message(
                    session_id,
                    "user",
                    prompt_text or query,
                    sender_id=self._sender_id(event),
                    sender_name=self._sender_name(event),
                )
                self._trim_if_needed(session_id)

            self._remove_previous_injections(req)

            locked_entries = self.store.get_locked_memories(
                session_id,
                limit=_safe_int(_cfg(self.config, "recall.locked_limit", 8), 8, 0, 30),
            )
            locked_context = format_locked_context(
                locked_entries,
                max_chars=_safe_int(_cfg(self.config, "recall.max_locked_chars", 1600), 1600, 300, 6000),
            )
            if locked_context:
                if bool(_cfg(self.config, "recall.inject_locked_to_system_prompt", True)):
                    req.system_prompt = self._append_block(str(getattr(req, "system_prompt", "") or ""), locked_context)
                else:
                    self._append_extra_user_content(req, locked_context)

            top_k = _safe_int(_cfg(self.config, "recall.top_k", 6), 6, 0, 20)
            if query and top_k > 0:
                candidate_k = max(top_k * 3, top_k)
                keyword_entries = self.store.search_memories(
                    query,
                    session_id=session_id,
                    top_k=candidate_k,
                    categories=["core", "memo", "log", "story_frame", "story_summary"],
                    include_global=bool(_cfg(self.config, "recall.include_global_memories", True)),
                )
                vector_entries = await self._vector_recall(query, session_id=session_id, top_k=candidate_k)
                entries = self.store.fuse_retrieval_results(
                    [keyword_entries, vector_entries],
                    top_k=max(top_k, 4),
                    diversity_threshold=float(_cfg(self.config, "recall.mmr_similarity_threshold", 0.82) or 0.82),
                )
                if bool(_cfg(self.config, "recall.always_include_story_frame", True)):
                    entries = self._merge_entries(
                        entries,
                        self.store.list_memories(session_id, category="story_frame", limit=2),
                        self.store.list_memories(session_id, category="story_summary", limit=2),
                    )
                if not entries and bool(_cfg(self.config, "recall.fallback_to_important_memories", True)):
                    entries = self._merge_entries(
                        self.store.list_memories(session_id, category="core", limit=2),
                        self.store.list_memories(session_id, category="memo", limit=2),
                    )
                entries = self.store.diversify_entries(
                    entries,
                    limit=max(top_k, 4),
                    threshold=float(_cfg(self.config, "recall.mmr_similarity_threshold", 0.82) or 0.82),
                )
                recall_context = format_recall_context(
                    entries[: max(top_k, 4)],
                    max_chars=_safe_int(_cfg(self.config, "recall.max_injection_chars", 2200), 2200, 500, 8000),
                )
                if recall_context:
                    self._append_extra_user_content(req, recall_context)
                    logger.info("[LayeredMemory] 已注入 %s 条相关记忆：session=%s", len(entries), session_id)
            story_context = format_story_state(
                self.store.get_story_state(session_id),
                max_chars=_safe_int(_cfg(self.config, "recall.max_story_state_chars", 1400), 1400, 300, 5000),
            )
            if story_context:
                self._append_extra_user_content(req, story_context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[LayeredMemory] LLM 请求前记忆处理失败：%s", exc, exc_info=True)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        if not bool(_cfg(self.config, "plugin_enabled", True)):
            return
        if getattr(resp, "role", "") != "assistant":
            return
        if getattr(resp, "tools_call_name", None) or getattr(resp, "tools_call_extra_content", None):
            return
        text = str(getattr(resp, "completion_text", "") or "").strip()
        if not text:
            return
        if self._looks_like_error(text):
            return
        session_id = getattr(event, "unified_msg_origin", "") or ""
        try:
            self.store.add_message(session_id, "assistant", text, sender_id="bot", sender_name="Bot")
            self._trim_if_needed(session_id)
            trigger_messages = _safe_int(
                _cfg(self.config, "memory_generation.summary_trigger_messages", 16),
                16,
                4,
                200,
            )
            if self.store.count_unsummarized_messages(session_id) >= trigger_messages:
                async with self._summary_lock:
                    if session_id in self._summarizing_sessions:
                        return
                    self._summarizing_sessions.add(session_id)
                self._track_task(self._summarize_session(session_id, event))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[LayeredMemory] LLM 回复后记忆整理失败：%s", exc, exc_info=True)

    async def _summarize_session(self, session_id: str, event: AstrMessageEvent, force: bool = False) -> tuple[int, int]:
        try:
            limit = _safe_int(_cfg(self.config, "memory_generation.summary_window_messages", 60), 60, 4, 240)
            rows = self.store.get_unsummarized_messages(session_id, limit=limit)
            if not rows:
                return (0, 0)
            if not force and len(rows) < _safe_int(_cfg(self.config, "memory_generation.summary_trigger_messages", 16), 16, 4, 200):
                return (0, len(rows))
            messages = [dict(row) for row in rows]
            summary = await self.processor.summarize_messages(
                messages,
                session_id=session_id,
                persona_id=await self._persona_id(event),
                is_group=self._is_group(event),
            )
            if summary.story_state:
                self.store.upsert_story_state(session_id, await self._persona_id(event), summary.story_state)
            stored = 0
            merged = 0
            for entry in summary.entries:
                if entry.content.strip():
                    memory_id, was_merged = self.store.add_or_merge_memory(
                        entry,
                        dedup_enabled=bool(_cfg(self.config, "memory_generation.deduplicate_memories", True)),
                        threshold=float(_cfg(self.config, "memory_generation.dedup_similarity_threshold", 0.82) or 0.82),
                    )
                    stored_entry = self.store.get_memory(memory_id)
                    if stored_entry:
                        self._schedule_vector_index(memory_id, stored_entry)
                    stored += 1
                    if was_merged:
                        merged += 1
            last_id = int(rows[-1]["id"])
            self.store.mark_summarized(session_id, last_id)
            logger.info("[LayeredMemory] 会话总结完成：session=%s stored=%s merged=%s messages=%s", session_id, stored, merged, len(rows))
            return (stored, len(rows))
        except Exception as exc:
            logger.error("[LayeredMemory] 会话总结任务失败：%s", exc, exc_info=True)
            return (0, 0)
        finally:
            self._summarizing_sessions.discard(session_id)

    # -------------------- Commands --------------------

    @filter.command_group(PLUGIN_COMMAND)
    def rmem(self):
        """Layered memory command group."""
        pass

    @admin_permission
    @rmem.command("status")
    async def status(self, event: AstrMessageEvent) -> AsyncGenerator[MessageEventResult, None]:
        stats = self.store.stats(getattr(event, "unified_msg_origin", "") or "")
        lines = ["分层记忆系统状态："]
        lines.append(f"- 总记忆：{stats['total']} 条")
        lines.append(f"- 待整理消息：{stats['unsummarized_messages']} 条")
        lines.append(f"- FTS 索引：{'可用' if stats['fts_available'] else '不可用，已使用 LIKE 检索兜底'}")
        embedding_provider = self._get_embedding_provider()
        lines.append(f"- 语义向量：{'可用' if embedding_provider else '未配置，已降级关键词检索'}")
        lines.append(f"- 向量索引：{stats['vectors']} 条")
        if stats.get("low_quality"):
            lines.append(f"- 待维护低质量记忆：{stats['low_quality']} 条")
        lines.append(f"- 剧情状态：{'已维护' if stats['story_state'] else '暂无'}")
        for category, name in MEMORY_CATEGORIES.items():
            lines.append(f"- {name}：{stats['by_category'].get(category, 0)} 条")
        yield event.plain_result("\n".join(lines))

    @admin_permission
    @rmem.command("search")
    async def search(self, event: AstrMessageEvent, query: str, k: int = 6) -> AsyncGenerator[MessageEventResult, None]:
        entries = self.store.search_memories(
            query,
            session_id=getattr(event, "unified_msg_origin", "") or "",
            top_k=max(1, min(20, int(k or 6))),
        )
        yield event.plain_result(format_entries(entries))

    @admin_permission
    @rmem.command("view")
    async def view(self, event: AstrMessageEvent, category: str = "all", limit: int = 12) -> AsyncGenerator[MessageEventResult, None]:
        normalized = None if category in {"all", "全部", "*"} else normalize_category(category)
        entries = self.store.list_memories(
            getattr(event, "unified_msg_origin", "") or "",
            category=normalized,
            limit=max(1, min(50, int(limit or 12))),
        )
        yield event.plain_result(format_entries(entries))

    @admin_permission
    @rmem.command("add")
    async def add(self, event: AstrMessageEvent, category: str, content: str) -> AsyncGenerator[MessageEventResult, None]:
        memory_id = self._manual_add(event, category, content)
        yield event.plain_result(f"已添加记忆 #{memory_id}。")

    @admin_permission
    @rmem.command("remember")
    async def remember(self, event: AstrMessageEvent, content: str) -> AsyncGenerator[MessageEventResult, None]:
        memory_id = self._manual_add(event, "core", content, importance=0.85)
        yield event.plain_result(f"已记住 #{memory_id}。")

    @admin_permission
    @rmem.command("core")
    async def add_core(self, event: AstrMessageEvent, content: str) -> AsyncGenerator[MessageEventResult, None]:
        memory_id = self._manual_add(event, "core", content)
        yield event.plain_result(f"已添加核心记忆 #{memory_id}。")

    @admin_permission
    @rmem.command("memo")
    async def add_memo(self, event: AstrMessageEvent, content: str) -> AsyncGenerator[MessageEventResult, None]:
        memory_id = self._manual_add(event, "memo", content)
        yield event.plain_result(f"已添加备忘录 #{memory_id}。")

    @admin_permission
    @rmem.command("lock")
    async def add_lock(self, event: AstrMessageEvent, content: str) -> AsyncGenerator[MessageEventResult, None]:
        memory_id = self._manual_add(event, "locked", content, importance=1.0, locked=True)
        yield event.plain_result(f"已添加锁定记忆 #{memory_id}，之后会高优先级注入。")

    @admin_permission
    @rmem.command("log")
    async def add_log(self, event: AstrMessageEvent, content: str) -> AsyncGenerator[MessageEventResult, None]:
        memory_id = self._manual_add(event, "log", content)
        yield event.plain_result(f"已添加记忆日志 #{memory_id}。")

    @admin_permission
    @rmem.command("frame")
    async def add_frame(self, event: AstrMessageEvent, content: str) -> AsyncGenerator[MessageEventResult, None]:
        memory_id = self._manual_add(event, "story_frame", content, importance=0.85)
        yield event.plain_result(f"已添加剧情框架 #{memory_id}。")

    @admin_permission
    @rmem.command("story")
    async def add_story_summary(self, event: AstrMessageEvent, content: str) -> AsyncGenerator[MessageEventResult, None]:
        memory_id = self._manual_add(event, "story_summary", content, importance=0.75)
        yield event.plain_result(f"已添加剧情总结 #{memory_id}。")

    @admin_permission
    @rmem.command("edit")
    async def edit(self, event: AstrMessageEvent, memory_id: int, content: str) -> AsyncGenerator[MessageEventResult, None]:
        ok = self.store.update_memory(int(memory_id), content=content)
        if ok:
            entry = self.store.get_memory(int(memory_id))
            if entry:
                self._schedule_vector_index(int(memory_id), entry)
        yield event.plain_result("已修改。" if ok else "没有找到这条记忆。")

    @admin_permission
    @rmem.command("delete")
    async def delete(self, event: AstrMessageEvent, memory_id: int) -> AsyncGenerator[MessageEventResult, None]:
        ok = self.store.delete_memory(int(memory_id))
        yield event.plain_result("已删除。" if ok else "没有找到这条记忆。")

    @admin_permission
    @rmem.command("forget")
    async def forget(self, event: AstrMessageEvent, memory_id: int) -> AsyncGenerator[MessageEventResult, None]:
        ok = self.store.delete_memory(int(memory_id))
        yield event.plain_result("已忘记。" if ok else "没有找到这条记忆。")

    @admin_permission
    @rmem.command("clear")
    async def clear(self, event: AstrMessageEvent, category: str = "all") -> AsyncGenerator[MessageEventResult, None]:
        normalized = None if category in {"all", "全部", "*"} else normalize_category(category)
        count = self.store.clear_category(getattr(event, "unified_msg_origin", "") or "", normalized)
        yield event.plain_result(f"已清空 {count} 条记忆。")

    @admin_permission
    @rmem.command("export")
    async def export(self, event: AstrMessageEvent, category: str = "all") -> AsyncGenerator[MessageEventResult, None]:
        normalized = None if category in {"all", "全部", "*"} else normalize_category(category)
        data = self.store.export_memories(getattr(event, "unified_msg_origin", "") or "", normalized)
        name = f"layered_memory_{normalized or 'all'}_{now_iso().replace(':', '-')}.json"
        path = self.export_dir / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        yield event.plain_result(f"已导出 {len(data)} 条记忆：{path}")

    @admin_permission
    @rmem.command("rebuild-index")
    async def rebuild_index(self, event: AstrMessageEvent) -> AsyncGenerator[MessageEventResult, None]:
        count = self.store.rebuild_index()
        yield event.plain_result(f"索引已重建：{count} 条。")

    @admin_permission
    @rmem.command("rebuild-vectors")
    async def rebuild_vectors(self, event: AstrMessageEvent, limit: int = 200) -> AsyncGenerator[MessageEventResult, None]:
        provider = self._get_embedding_provider()
        if provider is None:
            yield event.plain_result("当前没有可用的 Embedding Provider，无法补建语义向量。")
            return
        provider_id = self._provider_id(provider)
        entries = self.store.vector_missing_entries(limit=max(1, min(1000, int(limit or 200))), provider_id=provider_id)
        scheduled = 0
        for entry in entries:
            if entry.id is not None:
                self._schedule_vector_index(entry.id, entry)
                scheduled += 1
        yield event.plain_result(f"已安排 {scheduled} 条记忆补建向量索引。")

    @admin_permission
    @rmem.command("rebuild")
    async def rebuild(self, event: AstrMessageEvent, limit: int = 500) -> AsyncGenerator[MessageEventResult, None]:
        index_count = self.store.rebuild_index()
        provider = self._get_embedding_provider()
        if provider is None:
            yield event.plain_result(f"检索索引已重建：{index_count} 条。当前没有 Embedding Provider，语义向量保持降级模式。")
            return
        provider_id = self._provider_id(provider)
        entries = self.store.vector_missing_entries(limit=max(1, min(2000, int(limit or 500))), provider_id=provider_id)
        scheduled = 0
        for entry in entries:
            if entry.id is not None:
                self._schedule_vector_index(entry.id, entry)
                scheduled += 1
        yield event.plain_result(f"检索索引已重建：{index_count} 条；已安排 {scheduled} 条记忆补建语义向量。")

    @admin_permission
    @rmem.command("maintain")
    async def maintain(self, event: AstrMessageEvent, mode: str = "preview", days: int = 60) -> AsyncGenerator[MessageEventResult, None]:
        preview = str(mode or "preview").lower() not in {"exec", "执行", "run"}
        result = self.store.maintain_memories(
            getattr(event, "unified_msg_origin", "") or "",
            stale_days=max(7, min(365, int(days or 60))),
            preview=preview,
        )
        action = "预览" if preview else "已执行"
        yield event.plain_result(
            f"记忆维护{action}：候选 {result['candidates']} 条，降权 {result['decayed']} 条，停用 {result['disabled']} 条。"
        )

    @admin_permission
    @rmem.command("state")
    async def story_state(self, event: AstrMessageEvent) -> AsyncGenerator[MessageEventResult, None]:
        text = format_story_state(self.store.get_story_state(getattr(event, "unified_msg_origin", "") or ""), max_chars=5000)
        yield event.plain_result(text or "当前会话暂无剧情状态。")

    @admin_permission
    @rmem.command("summarize")
    async def summarize(self, event: AstrMessageEvent) -> AsyncGenerator[MessageEventResult, None]:
        session_id = getattr(event, "unified_msg_origin", "") or ""
        async with self._summary_lock:
            if session_id in self._summarizing_sessions:
                yield event.plain_result("当前会话已有整理任务在运行。")
                return
            self._summarizing_sessions.add(session_id)
        stored, messages = await self._summarize_session(session_id, event, force=True)
        yield event.plain_result(f"整理完成：处理 {messages} 条消息，写入 {stored} 条记忆。")

    @admin_permission
    @rmem.command("help")
    async def help(self, event: AstrMessageEvent) -> AsyncGenerator[MessageEventResult, None]:
        yield event.plain_result(
            "\n".join(
                [
                    "分层记忆默认自动工作，通常不用手动操作。",
                    "/rmem status - 查看状态",
                    "/rmem remember <内容> - 手动补充一条重要记忆",
                    "/rmem forget <id> - 删除一条记忆",
                    "/rmem summarize - 立即整理当前会话",
                    "/rmem rebuild [数量] - 重建检索索引并补建语义向量",
                ]
            )
        )

    # -------------------- Agent memory tools --------------------

    @filter.llm_tool(name="recall_layered_memory")
    async def recall_layered_memory(self, event: AstrMessageEvent, query: str, k: int = 5) -> str:
        """主动查询长期记忆。当用户询问过去信息、旧约定、偏好、剧情线索，或当前上下文不足以判断指代时使用。

        Args:
            query(str): 精简的查询关键词。优先使用主题、人物、偏好、约定、剧情线索，不要直接复制整段用户消息。
            k(int): 最多返回多少条记忆，通常 3 到 6 条即可。
        """
        if not bool(_cfg(self.config, "agent_tools.enable_recall_tool", True)):
            return self._json_tool_result({"ok": False, "error": "recall tool disabled", "results": []})
        cleaned_query = _plain(query)
        if not cleaned_query:
            return self._json_tool_result({"ok": False, "error": "query is empty", "results": []})
        session_id = getattr(event, "unified_msg_origin", "") or ""
        try:
            top_k = int(k or 5)
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(12, top_k))
        try:
            keyword_entries = self.store.search_memories(
                cleaned_query,
                session_id=session_id,
                top_k=top_k * 3,
                categories=["core", "memo", "log", "story_frame", "story_summary", "locked"],
                include_global=bool(_cfg(self.config, "recall.include_global_memories", True)),
            )
            vector_entries = await self._vector_recall(cleaned_query, session_id=session_id, top_k=top_k * 3)
            entries = self.store.fuse_retrieval_results(
                [keyword_entries, vector_entries],
                top_k=top_k,
                diversity_threshold=float(_cfg(self.config, "recall.mmr_similarity_threshold", 0.82) or 0.82),
            )
            return self._json_tool_result(
                {
                    "ok": True,
                    "query": cleaned_query,
                    "count": len(entries),
                    "results": [self._entry_tool_payload(entry) for entry in entries],
                }
            )
        except Exception as exc:
            logger.error("[LayeredMemory] 主动回忆工具失败：%s", exc, exc_info=True)
            return self._json_tool_result({"ok": False, "error": "internal_error", "results": []})

    @filter.llm_tool(name="memorize_layered_memory")
    async def memorize_layered_memory(
        self,
        event: AstrMessageEvent,
        memory: str,
        category: str = "core",
        importance: float = 0.75,
        tags: list[str] | None = None,
        key_facts: list[str] | None = None,
        reason: str = "",
    ) -> str:
        """主动写入长期记忆。仅在用户明确要求记住，或出现稳定偏好、身份事实、长期约定、重要剧情状态时使用。

        Args:
            memory(str): 要保存的简洁事实记忆，不要复制整段对话。
            category(str): 记忆分类，可用 core、memo、log、story_frame、story_summary；除非非常确定，不要写 locked。
            importance(float): 重要度，0.0 到 1.0；长期偏好、约定、身份事实、主线剧情应更高。
            tags(list[str]): 关键词标签，最多 8 个。
            key_facts(list[str]): 支撑这条记忆的独立关键事实，最多 8 条。
            reason(str): 为什么需要记住，简短说明即可。
        """
        if not bool(_cfg(self.config, "agent_tools.enable_memorize_tool", True)):
            return self._json_tool_result({"ok": False, "error": "memorize tool disabled"})
        cleaned_memory = str(memory or "").strip()
        if not cleaned_memory:
            return self._json_tool_result({"ok": False, "error": "memory is empty"})
        normalized = normalize_category(category)
        locked = normalized == "locked"
        if locked and not bool(_cfg(self.config, "memory_generation.allow_auto_locked_memory", False)):
            normalized = "core"
            locked = False
            tags = [*ensure_str_list(tags, limit=7), "锁定候选"]
        session_id = getattr(event, "unified_msg_origin", "") or ""
        metadata = {
            "schema": "layered-v2",
            "source_field": "agent_tool",
            "canonical_summary": cleaned_memory[:1600],
            "persona_summary": cleaned_memory[:1600],
            "key_facts": ensure_str_list(key_facts, limit=8),
            "summary_quality": "normal",
            "source_window": {
                "session_id": session_id,
                "triggered_by": "agent_tool",
                "tool_name": "memorize_layered_memory",
            },
        }
        cleaned_reason = str(reason or "").strip()
        if cleaned_reason:
            metadata["memorize_reason"] = cleaned_reason[:240]
        entry = MemoryEntry(
            session_id=session_id,
            persona_id=await self._persona_id(event),
            category=normalized,
            title="主动记忆",
            content=cleaned_memory[:1600],
            importance=clamp01(importance, 0.75),
            confidence=0.9,
            tags=ensure_str_list(tags, limit=8),
            metadata=metadata,
            source="agent_tool",
            locked=locked,
        )
        try:
            memory_id, was_merged = self.store.add_or_merge_memory(
                entry,
                dedup_enabled=bool(_cfg(self.config, "memory_generation.deduplicate_memories", True)),
                threshold=float(_cfg(self.config, "memory_generation.dedup_similarity_threshold", 0.82) or 0.82),
            )
            stored_entry = self.store.get_memory(memory_id)
            if stored_entry:
                self._schedule_vector_index(memory_id, stored_entry)
            return self._json_tool_result(
                {
                    "ok": True,
                    "id": memory_id,
                    "merged": was_merged,
                    "category": normalized,
                    "content": stored_entry.content if stored_entry else cleaned_memory,
                }
            )
        except Exception as exc:
            logger.error("[LayeredMemory] 主动记忆工具失败：%s", exc, exc_info=True)
            return self._json_tool_result({"ok": False, "error": "internal_error"})

    # -------------------- Helpers --------------------

    def _manual_add(
        self,
        event: AstrMessageEvent,
        category: str,
        content: str,
        *,
        importance: float = 0.7,
        locked: bool = False,
    ) -> int:
        normalized = normalize_category(category)
        memory_id, _ = self.store.add_or_merge_memory(
            MemoryEntry(
                session_id=getattr(event, "unified_msg_origin", "") or "",
                persona_id="",
                category=normalized,
                title="手动记录",
                content=content,
                importance=clamp01(importance),
                confidence=1.0,
                tags=ensure_str_list([category, "manual"]),
                metadata={"created_by": self._sender_id(event)},
                source="manual",
                locked=locked or normalized == "locked",
            )
        )
        entry = self.store.get_memory(memory_id)
        if entry:
            self._schedule_vector_index(memory_id, entry)
        return memory_id

    async def _vector_recall(self, query: str, *, session_id: str, top_k: int) -> list[MemoryEntry]:
        if not bool(_cfg(self.config, "recall.enable_vector_retrieval", True)):
            return []
        provider = self._get_embedding_provider()
        if provider is None:
            return []
        try:
            vector = await asyncio.wait_for(provider.get_embedding(query), timeout=8.0)
        except Exception as exc:
            logger.debug("[LayeredMemory] 查询向量生成失败：%s", exc)
            return []
        provider_id = self._provider_id(provider)
        return self.store.vector_search_memories(
            vector,
            session_id=session_id,
            top_k=max(1, int(top_k or 6)),
            categories=["core", "memo", "log", "story_frame", "story_summary", "locked"],
            include_global=bool(_cfg(self.config, "recall.include_global_memories", True)),
            provider_id=provider_id,
        )

    def _schedule_vector_index(self, memory_id: int, entry: MemoryEntry) -> None:
        if not bool(_cfg(self.config, "recall.enable_vector_retrieval", True)):
            return
        if memory_id <= 0:
            return
        self._track_task(self._index_memory_vector(memory_id, entry))

    async def _index_memory_vector(self, memory_id: int, entry: MemoryEntry) -> None:
        provider = self._get_embedding_provider()
        if provider is None:
            return
        metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
        key_facts = metadata.get("key_facts") if isinstance(metadata.get("key_facts"), list) else []
        persona_summary = metadata.get("persona_summary") if isinstance(metadata.get("persona_summary"), str) else ""
        canonical_summary = metadata.get("canonical_summary") if isinstance(metadata.get("canonical_summary"), str) else ""
        text = "\n".join(
            str(part).strip()
            for part in [
                entry.title,
                canonical_summary or entry.content,
                persona_summary,
                " ".join(str(item) for item in key_facts),
                " ".join(entry.tags),
            ]
            if str(part or "").strip()
        )
        if not text:
            return
        try:
            vector = await asyncio.wait_for(provider.get_embedding(text), timeout=12.0)
            self.store.set_memory_vector(
                memory_id,
                provider_id=self._provider_id(provider),
                vector=vector,
                content=text,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("[LayeredMemory] 记忆向量索引失败 id=%s err=%s", memory_id, exc)

    def _get_embedding_provider(self) -> Any | None:
        if self.embedding_provider_id:
            provider = self.context.get_provider_by_id(self.embedding_provider_id)
            if provider is not None and hasattr(provider, "get_embedding"):
                return provider
        getter = getattr(self.context, "get_all_embedding_providers", None)
        if callable(getter):
            try:
                providers = getter()
                if providers:
                    return providers[0]
            except Exception:
                return None
        return None

    @staticmethod
    def _provider_id(provider: Any) -> str:
        try:
            meta = provider.meta()
            return str(getattr(meta, "id", "") or "")
        except Exception:
            return provider.__class__.__name__

    async def _event_text(self, event: AstrMessageEvent) -> str:
        getter = getattr(event, "get_message_str", None)
        if callable(getter):
            value = getter()
            if asyncio.iscoroutine(value):
                value = await value
            return str(value or "").strip()
        return str(getattr(event, "message_str", "") or "").strip()

    def _sender_id(self, event: AstrMessageEvent) -> str:
        getter = getattr(event, "get_sender_id", None)
        if callable(getter):
            try:
                return str(getter() or "")
            except Exception:
                return ""
        return ""

    def _sender_name(self, event: AstrMessageEvent) -> str:
        for attr in ("get_sender_name", "get_sender_nickname"):
            getter = getattr(event, attr, None)
            if callable(getter):
                try:
                    value = getter()
                    if value:
                        return str(value)
                except Exception:
                    pass
        message_obj = getattr(event, "message_obj", None)
        sender = getattr(message_obj, "sender", None)
        return str(getattr(sender, "nickname", "") or getattr(sender, "name", "") or "")

    def _is_group(self, event: AstrMessageEvent) -> bool:
        try:
            if MessageType is not None:
                return event.get_message_type() == MessageType.GROUP_MESSAGE
        except Exception:
            pass
        session_id = str(getattr(event, "unified_msg_origin", "") or "")
        return "group" in session_id.lower()

    async def _persona_id(self, event: AstrMessageEvent) -> str:
        session_id = getattr(event, "unified_msg_origin", "") or ""
        try:
            provider = self.context.get_using_provider(session_id)
            persona_id = getattr(getattr(provider, "curr_persona", None), "id", "")
            if persona_id:
                return str(persona_id)
        except Exception:
            pass
        return ""

    def _append_extra_user_content(self, req: ProviderRequest, text: str) -> None:
        if not text:
            return
        if not hasattr(req, "extra_user_content_parts") or getattr(req, "extra_user_content_parts", None) is None:
            req.extra_user_content_parts = []
        if TextPart is not None:
            part = TextPart(text=text)
            marker = getattr(part, "mark_as_temp", None)
            if callable(marker):
                part = marker()
            else:
                setattr(part, "_no_save", True)
            req.extra_user_content_parts.append(part)
        else:
            req.prompt = self._append_block(str(getattr(req, "prompt", "") or ""), text)

    @staticmethod
    def _append_block(original: str, block: str) -> str:
        original = original.strip()
        block = block.strip()
        if not original:
            return block
        if not block:
            return original
        return f"{original}\n\n{block}"

    def _remove_previous_injections(self, req: ProviderRequest) -> None:
        for attr, header, footer in (
            ("system_prompt", LOCKED_HEADER, LOCKED_FOOTER),
            ("prompt", INJECTION_HEADER, INJECTION_FOOTER),
            ("prompt", STORY_HEADER, STORY_FOOTER),
        ):
            value = getattr(req, attr, "")
            if isinstance(value, str) and header in value and footer in value:
                setattr(req, attr, self._strip_block(value, header, footer))
        parts = getattr(req, "extra_user_content_parts", None)
        if isinstance(parts, list):
            req.extra_user_content_parts = [
                part for part in parts if not self._is_own_temp_part(part)
            ]

    @staticmethod
    def _strip_block(text: str, header: str, footer: str) -> str:
        pattern = re.compile(re.escape(header) + r".*?" + re.escape(footer), re.DOTALL)
        return re.sub(r"\n{3,}", "\n\n", pattern.sub("", text)).strip()

    @staticmethod
    def _is_own_temp_part(part: Any) -> bool:
        text = getattr(part, "text", "")
        return (
            isinstance(text, str)
            and getattr(part, "_no_save", False)
            and (
                (INJECTION_HEADER in text and INJECTION_FOOTER in text)
                or (LOCKED_HEADER in text and LOCKED_FOOTER in text)
                or (STORY_HEADER in text and STORY_FOOTER in text)
            )
        )

    @staticmethod
    def _merge_entries(*groups: list[MemoryEntry]) -> list[MemoryEntry]:
        merged: list[MemoryEntry] = []
        seen: set[int] = set()
        for group in groups:
            for entry in group:
                if entry.id is None or entry.id in seen:
                    continue
                merged.append(entry)
                seen.add(entry.id)
        return merged

    @staticmethod
    def _json_tool_result(data: dict[str, Any]) -> str:
        return json.dumps(data, ensure_ascii=False, default=str)

    @staticmethod
    def _entry_tool_payload(entry: MemoryEntry) -> dict[str, Any]:
        metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
        return {
            "id": entry.id,
            "category": entry.category,
            "title": entry.title,
            "content": metadata.get("persona_summary") or entry.content,
            "canonical_summary": metadata.get("canonical_summary") or entry.content,
            "importance": entry.importance,
            "confidence": entry.confidence,
            "tags": entry.tags,
            "score": round(float(entry.score or 0.0), 4),
            "source": entry.source,
        }

    @staticmethod
    def _looks_like_error(text: str) -> bool:
        lowered = text.lower()
        markers = ["api error", "request failed", "rate limit", "timeout", "请求失败", "接口错误", "服务暂时不可用"]
        return any(marker in lowered for marker in markers)

    def _trim_if_needed(self, session_id: str) -> None:
        keep_latest = _safe_int(_cfg(self.config, "session_capture.max_messages_per_session", 1000), 1000, 100, 10000)
        try:
            self.store.trim_messages(session_id, keep_latest)
        except Exception as exc:
            logger.debug("[LayeredMemory] 清理历史消息失败：%s", exc)

    async def terminate(self):
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self.store.close()
        logger.info("[LayeredMemory] 插件已停止")
