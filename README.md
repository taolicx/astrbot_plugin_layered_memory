# astrbot_plugin_layered_memory

分层长期记忆与剧情延续插件。它参考了 `astrbot_plugin_livingmemory` 的 AstrBot 钩子模式，也吸收了 `astrbot_plugin_private_companion` 对陪伴状态、提示词边界和长期状态维护的设计思路；当前实现按“效果优先”设计，默认启用语义向量召回、自动去重合并、结构化剧情状态和混合检索兜底。

## 功能

- 核心记忆：长期保存重要经历、偏好、雷点、约定、角色核心设定和剧情转折。
- 备忘录：保存称呼、语气、待办、小设定和后续剧情线索。
- 锁定记忆：高优先级硬约束，默认手动添加，每轮回复前注入。
- 记忆日志：定期总结最近发生的聊天和剧情细节。
- 语义召回：优先使用 AstrBot 的 Embedding Provider 建立向量索引，理解同义表达、隐含线索和长剧情关联。
- 关键词兜底：没有 embedding provider 时自动回退到 SQLite/FTS/LIKE 混合检索，不影响插件运行。
- 自动去重合并：相似记忆会合并补充，避免长期使用后重复条目越来越多。
- 剧情框架：保存当前阶段、世界观、事件、关系变化、冲突和目标。
- 结构化剧情状态：持续维护当前阶段、未解决冲突、短期目标、长期主线和后续线索。
- 剧情总结：形成阶段性剧情摘要，帮助长篇互动延续主线。

## 安装

把整个 `astrbot_plugin_layered_memory` 文件夹放到 AstrBot 的 `data/plugins` 目录，然后重启或重载插件。

若要达到最佳效果，请在 AstrBot 里配置一个 Embedding Provider。插件配置中的 `embedding_provider_id` 可以留空，插件会自动使用第一个可用的 Embedding Provider；没有 embedding provider 时会自动降级关键词检索。

## 常用命令

正常使用不需要靠指令操控。插件会自动捕获对话、整理记忆、召回相关背景、维护剧情状态。下面几个命令只是维护入口：

```text
/rmem status
/rmem remember <内容>
/rmem forget <id>
/rmem summarize
/rmem rebuild [数量]
```

旧的细分命令仍作为高级/兼容入口保留，但默认不用学，也不用日常操作。

## 命令说明

- `/rmem status`：查看当前会话的记忆数量、待整理消息、FTS、语义向量和剧情状态。
- `/rmem remember <内容>`：手动补充一条重要记忆，通常只有自动整理漏掉时才用。
- `/rmem forget <id>`：删除一条不想保留的记忆。
- `/rmem summarize`：立即整理当前会话未整理消息。
- `/rmem rebuild [数量]`：重建 SQLite FTS 检索索引，并为缺失或 provider 不一致的记忆补建语义向量。

## 默认配置可以直接用吗

可以。当前默认值已经偏“效果优先”：更早触发整理、更大的注入窗口、语义向量召回、重要记忆兜底、剧情状态注入和自动去重合并都会开启。唯一建议是配置一个 Embedding Provider，否则插件仍能用，但语义召回效果会下降。

## 设计取舍

- 普通召回使用临时 `extra_user_content_parts` 注入，避免污染历史上下文。
- 锁定记忆默认追加到 `system_prompt`，用于满足“最高优先级强制记忆”的需求。
- 自动生成锁定记忆默认关闭，防止模型把普通偏好误判成不可更改规则。
- 向量索引保存在 SQLite 内，避免额外部署向量数据库；效果来自 AstrBot 的 Embedding Provider。
- 语义召回、关键词检索、剧情框架和重要记忆兜底会混合使用，尽量让回复既想得起旧事，也不被无关记忆淹没。
