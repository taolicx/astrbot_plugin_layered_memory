import tempfile
import unittest
from pathlib import Path

from core.schema import MemoryEntry
from core.storage import LayeredMemoryStore


class StorageTests(unittest.TestCase):
    def test_add_search_locked_and_rebuild(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LayeredMemoryStore(Path(temp_dir) / "memory.db")
            try:
                core_id = store.add_memory(
                    MemoryEntry(
                        session_id="session-1",
                        category="core",
                        title="口味",
                        content="用户喜欢清淡的茶，不喜欢太甜的饮料。",
                        importance=0.8,
                        tags=["偏好", "茶"],
                    )
                )
                locked_id = store.add_memory(
                    MemoryEntry(
                        session_id="session-1",
                        category="locked",
                        title="底线",
                        content="绝对不能在角色扮演中背叛用户。",
                        importance=1.0,
                        locked=True,
                        tags=["底线"],
                    )
                )

                self.assertGreater(core_id, 0)
                self.assertGreater(locked_id, core_id)

                found = store.search_memories("清淡 茶", session_id="session-1", top_k=3)
                self.assertTrue(found)
                self.assertTrue(found[0].content.startswith("用户喜欢清淡"))

                locked = store.get_locked_memories("session-1")
                self.assertEqual(len(locked), 1)
                self.assertEqual(locked[0].id, locked_id)

                self.assertGreaterEqual(store.rebuild_index(), 2)

                exported = store.export_memories("session-1")
                self.assertEqual(len(exported), 2)
                self.assertEqual({item["category"] for item in exported}, {"core", "locked"})
            finally:
                store.close()

    def test_messages_and_summarized_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LayeredMemoryStore(Path(temp_dir) / "memory.db")
            try:
                first = store.add_message("session-1", "user", "我们继续昨天的剧情。")
                second = store.add_message("session-1", "assistant", "好，主线仍在城堡门口。")
                self.assertGreater(first, 0)
                self.assertGreater(second, first)
                self.assertEqual(store.count_unsummarized_messages("session-1"), 2)

                rows = store.get_unsummarized_messages("session-1")
                self.assertEqual(len(rows), 2)
                store.mark_summarized("session-1", int(rows[-1]["id"]))
                self.assertEqual(store.count_unsummarized_messages("session-1"), 0)
            finally:
                store.close()

    def test_vectors_rebuild_and_update_invalidation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LayeredMemoryStore(Path(temp_dir) / "memory.db")
            try:
                first_id = store.add_memory(
                    MemoryEntry(
                        session_id="session-1",
                        category="core",
                        title="茶",
                        content="用户喜欢清淡绿茶。",
                        importance=0.9,
                        tags=["偏好"],
                    )
                )
                second_id = store.add_memory(
                    MemoryEntry(
                        session_id="session-1",
                        category="memo",
                        title="剧情",
                        content="下一幕要回到城堡门口。",
                        importance=0.7,
                        tags=["剧情"],
                    )
                )

                store.set_memory_vector(first_id, provider_id="embed-a", vector=[1.0, 0.0, 0.0], content="tea")
                found = store.vector_search_memories([0.9, 0.1, 0.0], session_id="session-1", top_k=2, provider_id="embed-a")
                self.assertEqual(found[0].id, first_id)

                missing_for_same_provider = store.vector_missing_entries(limit=10, provider_id="embed-a")
                self.assertEqual([item.id for item in missing_for_same_provider], [second_id])

                missing_for_new_provider = store.vector_missing_entries(limit=10, provider_id="embed-b")
                self.assertEqual({item.id for item in missing_for_new_provider}, {first_id, second_id})

                self.assertTrue(store.update_memory(first_id, content="用户改为喜欢无糖乌龙茶。"))
                self.assertEqual(store.count_vectors("session-1"), 0)
            finally:
                store.close()

    def test_story_state_merge_and_locked_limit_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LayeredMemoryStore(Path(temp_dir) / "memory.db")
            try:
                store.add_memory(
                    MemoryEntry(
                        session_id="session-1",
                        category="locked",
                        title="底线",
                        content="不要越过用户边界。",
                        locked=True,
                    )
                )
                self.assertEqual(store.get_locked_memories("session-1", limit=0), [])

                store.upsert_story_state(
                    "session-1",
                    "persona",
                    {
                        "current_stage": "城堡门口",
                        "important_events": ["发现旧徽章"],
                    },
                )
                store.upsert_story_state(
                    "session-1",
                    "persona",
                    {
                        "important_events": ["发现旧徽章", "守卫开始怀疑"],
                        "next_hooks": ["解释徽章来源"],
                    },
                )
                state = store.get_story_state("session-1")
                self.assertEqual(state["current_stage"], "城堡门口")
                self.assertEqual(state["important_events"], ["发现旧徽章", "守卫开始怀疑"])
                self.assertEqual(state["next_hooks"], ["解释徽章来源"])
            finally:
                store.close()

    def test_vectors_dedup_and_story_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LayeredMemoryStore(Path(temp_dir) / "memory.db")
            try:
                memory_id, merged = store.add_or_merge_memory(
                    MemoryEntry(
                        session_id="session-1",
                        category="core",
                        title="称呼",
                        content="用户希望被称呼为林先生。",
                        importance=0.75,
                        tags=["称呼"],
                    )
                )
                self.assertFalse(merged)

                same_id, merged = store.add_or_merge_memory(
                    MemoryEntry(
                        session_id="session-1",
                        category="core",
                        title="称呼偏好",
                        content="用户希望被称呼为林先生，语气自然一点。",
                        importance=0.8,
                        tags=["称呼", "语气"],
                    ),
                    threshold=0.5,
                )
                self.assertTrue(merged)
                self.assertEqual(same_id, memory_id)
                merged_entry = store.get_memory(memory_id)
                self.assertIsNotNone(merged_entry)
                self.assertIn("语气自然", merged_entry.content)

                store.set_memory_vector(memory_id, provider_id="test-embed", vector=[1.0, 0.0, 0.0], content=merged_entry.content)
                found = store.vector_search_memories(
                    [0.98, 0.01, 0.0],
                    session_id="session-1",
                    top_k=3,
                    provider_id="test-embed",
                )
                self.assertEqual([item.id for item in found], [memory_id])

                store.upsert_story_state(
                    "session-1",
                    "",
                    {
                        "current_stage": "城堡门口对峙",
                        "important_events": ["主角拿到了旧钥匙"],
                        "next_hooks": ["调查塔楼"],
                    },
                )
                store.upsert_story_state(
                    "session-1",
                    "",
                    {
                        "important_events": ["守卫开始怀疑主角"],
                        "next_hooks": ["调查塔楼", "寻找密道"],
                    },
                )
                state = store.get_story_state("session-1")
                self.assertEqual(state["current_stage"], "城堡门口对峙")
                self.assertIn("守卫开始怀疑主角", state["important_events"])
                self.assertEqual(state["next_hooks"].count("调查塔楼"), 1)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
