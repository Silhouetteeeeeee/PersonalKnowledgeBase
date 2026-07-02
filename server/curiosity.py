"""好奇心引擎：自主知识发现模块。

定时任务，周期性地：
1. 选题 → LLM 分析知识库缺口，选择研究方向
2. 研究 → 网络搜索获取最新资料
3. 蒸馏 → extract_to_wiki 两步 CoT 存入知识库
4. 日志 → 记录学习成果

Phase 1: 默默学习，不影响用户
Phase 2+（未来）: 主动分享学到的知识
"""

import json
import logging
import os
from datetime import datetime

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from agent.tools.web_search import search_web, search_web_from_baidu
from storage.models import get_all_pages_index
from agent.nodes.store import extract_to_wiki

logger = logging.getLogger(__name__)


# ── Pydantic models ──

class ResearchTopic(BaseModel):
    topic: str = Field(description="研究主题名称，例如『FastAPI 异步特性』、『Rust 所有权机制』")
    search_query: str = Field(
        description="用于网络搜索的查询词，具体且可操作，例如『FastAPI async 性能对比 2025』"
    )
    reason: str = Field(
        description="选择这个主题的原因：知识缺口 / 技术更新 / 延伸学习"
    )


# ── Step 1: Topic selection ──

def _select_topic() -> ResearchTopic | None:
    """LLM 分析当前知识库，选择一个值得探索的研究方向。"""
    pages = get_all_pages_index()

    if pages:
        lines = []
        for p in pages:
            tags_raw = p.get("tags", "[]")
            tags = []
            if isinstance(tags_raw, str):
                try:
                    tags = json.loads(tags_raw)[:3]
                except (json.JSONDecodeError, TypeError):
                    tags = []
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"- {p['title']}{tag_str}")
        overview = "当前知识库：\n" + "\n".join(lines)
    else:
        overview = "知识库目前为空。"

    prompt = (
        f"{overview}\n\n"
        "你是一个知识探索助手。请基于当前知识库，推荐一个值得探索的研究主题。\n\n"
        "## 选题原则\n"
        "1. 如果知识库不为空：选择现有知识的自然延伸或补充\n"
        "   - 填补知识缺口（比如有 Django 但没有 FastAPI）\n"
        "   - 跟进技术更新（比如用户学的是 Python 3.10，现在有 3.13）\n"
        "   - 探索相关领域（比如有数据库基础，可以学 Redis）\n"
        "2. 如果知识库为空：选择一个通用且有价值的技术主题\n"
        "3. topic 要具体（不是『Python』，而是『FastAPI 异步特性』）\n"
        "4. search_query 要适合网络搜索，能返回高质量中文或英文内容\n"
        "5. 避免选择知识库中已覆盖得很好的主题\n"
        "6. 用中文输出"
    )

    try:
        result = LLM.generate_structured(prompt, ResearchTopic, use_language=False)
        logger.info("好奇心选题: %s | query: %s | 原因: %s",
                     result.topic, result.search_query, result.reason)
        return result
    except Exception as e:
        logger.warning("选题 LLM 调用失败: %s", e)
        return None


# ── Step 2: Research ──

def _research(query: str) -> str:
    """搜索网络获取主题相关内容，返回合并文本。"""
    # 策略：百度搜索 → DuckDuckGo 回退
    results = search_web_from_baidu(query)
    if not results or any(r.startswith("[Search error") for r in results):
        logger.info("百度搜索回退到 DuckDuckGo: %s", query[:40])
        results = search_web(query, max_results=5)

    if not results:
        logger.warning("搜索无结果: %s", query[:40])
        return ""

    # 合并前 3 条结果
    combined = "\n\n---\n\n".join(results[:3])
    logger.info("搜索到 %d 条结果, 合并 %d 字符", len(results), len(combined))
    return combined


# ── Step 3: Quality assessment + distillation ──
# 由 extract_to_wiki 内置的两步 CoT 完成
# Step 1 (analyze) 决定是否值得存储，Step 2 (generate) 生成页面


# ── Step 4: Logging ──

def _save_curiosity_log(result: dict):
    """将好奇心探索结果写入日志文件。"""
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "reasoning", "curiosity",
    )
    os.makedirs(log_dir, exist_ok=True)
    filename = f"explore_{datetime.now():%Y%m%d_%H%M%S}.md"
    filepath = os.path.join(log_dir, filename)

    lines = [
        f"# 🧠 自主知识发现 - {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"**状态**: {result.get('status', 'unknown')}",
        f"**主题**: {result.get('topic', 'N/A')}",
        f"**原因**: {result.get('reason', 'N/A')}",
        f"**搜索词**: {result.get('search_query', 'N/A')}",
        "",
    ]
    if result.get("pages_created", 0) > 0:
        lines.append(f"**创建页面**: {result['pages_created']} 篇")
        # 从 logic_chain 提取页面详情
        for step in result.get("logic_chain", []):
            if step.get("node") == "store":
                lines.append(f"  详情: {step.get('action', '')}")
                lines.append(f"  推理: {step.get('reasoning', '')}")
    else:
        lines.append(f"**跳过原因**: {result.get('reason', '未知')}")
    lines.append("")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("好奇心日志已保存: %s", filepath)
    except Exception as e:
        logger.warning("保存好奇心日志失败: %s", e)


# ── Main entry point ──

def discover_and_learn() -> dict:
    """一次完整的自主知识发现周期。由 APScheduler 定时调用。

    Returns:
        dict with status, topic, pages_created, etc.
    """
    logger.info("🧠 好奇心引擎启动")

    # Step 1: 选题
    topic = _select_topic()
    if topic is None:
        logger.info("未选择到合适的研究主题，跳过本次学习")
        return {"status": "skipped", "reason": "no topic selected"}

    # Step 2: 研究
    content = _research(topic.search_query)
    if not content:
        logger.info("未获取到研究内容: %s", topic.topic)
        result = {"status": "skipped", "reason": "no research content",
                  "topic": topic.topic}
        _save_curiosity_log(result)
        return result

    # Step 3: 蒸馏存储（两步 CoT 内建质量判断）
    source_id = f"curiosity_{datetime.now():%Y%m%d_%H%M%S}"
    source_label = f"🔍 自主发现: {topic.topic}"

    try:
        wiki_result = extract_to_wiki(
            source_text=content,
            source_id=source_id,
            source_label=source_label,
        )
    except Exception as e:
        logger.exception("知识蒸馏失败: %s", e)
        result = {"status": "failed", "reason": f"distillation error: {e}",
                  "topic": topic.topic, "search_query": topic.search_query}
        _save_curiosity_log(result)
        return result

    page_count = len(wiki_result.get("page_ids", []))
    logic_chain = wiki_result.get("logic_chain", [])

    if page_count > 0:
        logger.info("✅ 自主学习完成: 主题='%s', 创建 %d 个页面",
                     topic.topic, page_count)
    else:
        logger.info("内容质量不足，未创建新页面: %s", topic.topic)

    result = {
        "status": "success" if page_count > 0 else "skipped",
        "topic": topic.topic,
        "search_query": topic.search_query,
        "reason": topic.reason,
        "pages_created": page_count,
        "logic_chain": logic_chain,
    }
    _save_curiosity_log(result)
    return result
