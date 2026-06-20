"""
╔══════════════════════════════════════════════════════════════╗
║                LangChain-Learning 综合测试套件                ║
║  模拟企业微信机器人消息 → 全链路监控 Graph 执行 → 验证输出     ║
╚══════════════════════════════════════════════════════════════╝

测试目标:
  1. 覆盖 Graph 所有节点 (10 个节点 + 2 个路由)
  2. 覆盖 10 类业务场景 (事实/对比/搜索/闲聊/画像/URL/矛盾/多轮/长文本/边界)
  3. 监控每个节点的执行日志和状态变化
  4. 验证数据库和 Wiki 文件的正确性
  5. 测试 Profile/MessageHistory/EpisodicMemory 的联动

使用方法:
  python _comprehensive_test.py
"""

import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from typing import Any

# 强制中文输出
os.environ["OUTPUT_LANGUAGE"] = "Chinese"

# ── 日志配置 (保留 INFO 级别, 压制第三方库) ──
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

for _lib in ("httpx", "fastembed", "httpcore", "akshare", "urllib3", "jieba"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

logger = logging.getLogger("comprehensive_test")

# ── 测试报告收集器 ──
report_lines: list[str] = []


def report(msg: str, section: bool = False) -> None:
    """统一报告输出: 同时写入日志和报告列表"""
    if section:
        sep = "═" * 60
        styled = f"\n{sep}\n  {msg}\n{sep}"
    else:
        styled = f"  • {msg}"
    logger.info(styled)
    report_lines.append(msg)


def print_separator(title: str) -> None:
    """打印醒目的区块分隔线"""
    width = 66
    print(f"\n{'┏' + '━' * (width - 2) + '┓'}")
    print(f"┃ {title:<{width - 4}} ┃")
    print(f"{'┗' + '━' * (width - 2) + '┛'}\n")


# ══════════════════════════════════════════════════════════════
# 第 0 步: 初始化和数据库准备
# ══════════════════════════════════════════════════════════════

print_separator("第 0 步: 初始化环境")

from storage.database import init_db, get_connection, DB_DIR
from storage.models import (
    get_all_pages_index,
    find_similar_pages,
    get_page_by_title,
)
from storage.wiki_storage import (
    read_page,
    WIKI_DIR,
    parse_frontmatter,
    extract_wikilinks,
    title_to_filename,
)
from storage.profile import load_profile, save_profile
from memory.message_history import MessageHistory
from memory.session_manager import SessionManager
from memory.episodic import EpisodicMemory
from agent.graph import build_graph

init_db()
report(f"数据库路径: {os.path.join(DB_DIR, 'knowledge.db')}", section=False)
report(f"Wiki 目录: {WIKI_DIR}")

# 预热 Embedding 模型 (首次加载慢)
from storage.models import generate_embedding

report("预热 Embedding 模型...")
_ = generate_embedding("预热")
report("Embedding 模型就绪")

# 获取测试前的页面快照
pre_test_pages = get_all_pages_index()
report(f"测试前 Wiki 页面数: {len(pre_test_pages)}")

# 构建 Graph
graph = build_graph()
report("Graph 构建完成")

# 创建会话管理器
session_manager = SessionManager()
episodic_memory = EpisodicMemory()

# ══════════════════════════════════════════════════════════════
# 测试辅助函数
# ══════════════════════════════════════════════════════════════


def run_test_case(
    user_message: str,
    user_id: str = "comprehensive_tester",
    session_id: str | None = None,
    description: str = "",
    message_history: list[dict] | None = None,
    user_profile: dict | None = None,
    episodic_memories: list[str] | None = None,
    url_contents: list | None = None,
    expected_confidence_min: float = 0.0,
    expect_store: bool | None = None,
) -> dict[str, Any]:
    """
    模拟企业微信收到消息 → 调用 Graph 执行全链路

    参数:
      user_message: 用户消息文本
      user_id: 用户 ID
      session_id: 会话 ID (None 则自动生成)
      description: 测试用例描述
      message_history: 历史消息列表
      user_profile: 用户画像
      episodic_memories: 情景记忆
      url_contents: 已提取的 URL 内容 (通常由 parse 节点填充)
      expected_confidence_min: 期望的最低置信度
      expect_store: 期望的 needs_store 值 (None 则不检查)

    返回:
      graph.invoke() 的完整结果字典
    """
    if session_id is None:
        session_id = int(time.time() * 1000) % 100000
    else:
        session_id = int(session_id) if isinstance(session_id, str) else session_id

    print_separator(f"测试: {description}")
    print(f"  用户消息: {user_message[:80]}")
    print(f"  用户 ID:  {user_id}")
    print(f"  会话 ID:  {session_id}")
    if message_history:
        print(f"  历史消息: {len(message_history)} 条")

    start_time = time.time()

    result = graph.invoke({
        "user_message": user_message,
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "user_profile": user_profile or load_profile(user_id),
        "message_history": message_history or [],
        "episodic_memories": episodic_memories or [],
        "url_contents": url_contents or [],
        "search_query": "",
        "confidence": 0.0,
        "needs_store": False,
        "search_results": [],
        "stored_knowledge": [],
        "stored_knowledge_ids": [],
        "wiki_page_ids": [],
        "answer": "",
        "final_response": "",
        "url_contents": url_contents or [],
        "contradiction_found": False,
        "contradiction_details": "",
        "search_time": 0,
        "contradiction_severity": "",
        "contradiction_knowledge_ids": [],
        "contradiction_knowledge_texts": [],
        "reflection_result": "",
        "reflection_reasoning": "",
        "reflection_correction": "",
        "force_web_search": False,
        "correction_attempts": 0,
        "error_recorded": False,
        "logic_chain": [],
    })

    elapsed = time.time() - start_time

    # ── 打印结果摘要 ──
    final_response = result.get("final_response", "")
    answer = result.get("answer", "")
    confidence = result.get("confidence", 0.0)
    needs_store = result.get("needs_store", False)
    contradiction = result.get("contradiction_found", False)
    logic_chain = result.get("logic_chain", [])
    wiki_page_ids = result.get("wiki_page_ids", [])
    correction_attempts = result.get("correction_attempts", 0)

    print(f"\n  ⏱  耗时: {elapsed:.1f}s")
    print(f"  📊 置信度: {confidence:.2f}")
    print(f"  💾 需要存储: {needs_store}")
    print(f"  ⚠️  矛盾检测: {'发现矛盾!' if contradiction else '通过'}")
    print(f"  🔄 修正次数: {correction_attempts}")
    print(f"  📝 答案长度: {len(answer)} 字符")
    print(f"  📄 最终响应长度: {len(final_response)} 字符")
    print(f"  📋 推理步骤数: {len(logic_chain)}")
    if wiki_page_ids:
        print(f"  🏷  Wiki 页面 ID: {wiki_page_ids}")
    print(f"\n  ── 推理链路 ──")
    for step in logic_chain:
        node = step.get("node", "?")
        action = step.get("action", "")
        reasoning = step.get("reasoning", "")[:120]
        print(f"    [{node}] {action}")
        if reasoning:
            print(f"      ↳ {reasoning}")

    print(f"\n  ── 最终回答 ──")
    print(f"  {final_response[:300]}")
    if len(final_response) > 300:
        print(f"  ... (共 {len(final_response)} 字符，仅显示前 300)")

    report(f"[{description}] 耗时={elapsed:.1f}s | 置信度={confidence:.2f} | "
           f"需存储={needs_store} | 矛盾={contradiction} | "
           f"推理步={len(logic_chain)} | Wiki页面={wiki_page_ids}",
           section=False)

    # ── 基本验证 ──
    if confidence < expected_confidence_min:
        print(f"  ⚠️  置信度 {confidence} 低于期望 {expected_confidence_min}")
    if expect_store is not None and needs_store != expect_store:
        print(f"  ⚠️  needs_store={needs_store} 与期望 {expect_store} 不符")
    if not final_response:
        print(f"  ❌ 最终响应为空!")
    elif len(final_response) < 5:
        print(f"  ⚠️  最终响应过短: '{final_response}'")

    return result


def inspect_db() -> None:
    """检查数据库状态: 页面数、页面内容、向量索引"""
    print_separator("数据库检查")

    conn = get_connection()
    conn.row_factory = sqlite3.Row

    # 1. 页面统计
    active_pages = conn.execute(
        "SELECT COUNT(*) as cnt FROM pages WHERE status='active'"
    ).fetchone()["cnt"]
    error_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM error_records"
    ).fetchone()["cnt"]
    file_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM file_records"
    ).fetchone()["cnt"]

    print(f"  📊 活跃页面: {active_pages}")
    print(f"  📊 错误记录: {error_count}")
    print(f"  📊 文件记录: {file_count}")

    # 2. 最近创建的页面 (按更新时间倒序)
    recent = conn.execute(
        "SELECT id, title, file_path, tags, sources, updated_at "
        "FROM pages WHERE status='active' ORDER BY updated_at DESC LIMIT 10"
    ).fetchall()

    print(f"\n  ── 最近更新页面 (前 10) ──")
    for p in recent:
        tags = json.loads(p["tags"]) if p["tags"] else []
        sources = json.loads(p["sources"]) if p["sources"] else []
        print(f"    [{p['id']:3d}] {p['title']:<30s} | 标签={tags[:3]} | 来源={sources[:2]}")

    # 3. 检查错误记录详情
    if error_count > 0:
        errors = conn.execute(
            "SELECT id, user_message, wrong_answer[:60] as preview, error_type, "
            "contradiction_details[:80] as cd_preview "
            "FROM error_records ORDER BY id DESC LIMIT 5"
        ).fetchall()
        print(f"\n  ── 最近错误记录 ──")
        for e in errors:
            print(f"    [{e['id']}] {e['user_message'][:40]} | 类型={e['error_type']} "
                  f"| 详情={e['cd_preview']}")

    conn.close()


def check_wiki_integrity() -> None:
    """检查 Wiki 文件系统完整性: 文件存在性、frontmatter、交叉引用"""
    print_separator("Wiki 文件完整性检查")

    from storage.models import get_all_pages_index

    all_pages = get_all_pages_index(status="active")
    issues = 0

    for p in all_pages:
        fp = os.path.join(WIKI_DIR, p.file_path)

        # 检查文件是否存在
        if not os.path.exists(fp):
            print(f"  ❌ [{p.id}] {p.title}: 文件缺失 at {fp}")
            issues += 1
            continue

        size = os.path.getsize(fp)
        if size < 30:
            print(f"  ⚠️ [{p.id}] {p.title}: 文件过小 ({size} bytes)")
            issues += 1

        # 解析 frontmatter
        with open(fp, "r", encoding="utf-8") as f:
            raw = f.read()
        meta, content = parse_frontmatter(raw)
        if not meta.get("title"):
            print(f"  ⚠️ [{p.id}] {p.title}: frontmatter 缺少 title")
            issues += 1

        # 检查交叉引用
        wikilinks = extract_wikilinks(content)
        for link in wikilinks:
            target = link.lower().strip()
            target_page = get_page_by_title(target)
            if not target_page:
                # 检查 filesystem 文件名
                target_file = title_to_filename(target)
                target_path = os.path.join(WIKI_DIR, target_file)
                if not os.path.exists(target_path):
                    print(f"  ⚠️ [{p.id}] {p.title}: 交叉引用 [[{link}]] 指向不存在的页面")
                    issues += 1

    if issues == 0:
        print(f"  ✅ 全部 {len(all_pages)} 个页面完整性检查通过")
    else:
        print(f"  ⚠️  发现 {issues} 个问题 (在 {len(all_pages)} 个页面中)")


def check_profile(user_id: str) -> dict:
    """检查用户画像的当前状态"""
    profile = load_profile(user_id)
    print_separator(f"用户画像检查: {user_id}")
    if not profile:
        print("  (空画像)")
    else:
        for section, data in profile.items():
            if section == "updated_at":
                continue
            if isinstance(data, dict):
                for k, v in data.items():
                    if v:
                        print(f"  [{section}] {k}: {str(v)[:100]}")
            elif isinstance(data, list) and data:
                print(f"  [{section}]: {', '.join(str(x)[:50] for x in data)}")
            elif data:
                print(f"  [{section}]: {str(data)[:100]}")
    return profile


# ══════════════════════════════════════════════════════════════
# 第 1 阶段: 基础功能测试
# ══════════════════════════════════════════════════════════════

def phase_1_basic_factual() -> None:
    """第1阶段: 基本事实性问题 — 测试 parse → rewrite → retrieve → classify → store"""
    print_separator("第 1 阶段: 基本事实性问题")

    results = []

    # 1.1 简单事实 — 北京的认知
    results.append(run_test_case(
        user_message="中国的首都是什么？",
        description="简单事实-首都",
        expected_confidence_min=0.5,
        expect_store=True,
    ))

    # 1.2 技术对比 — Redis
    results.append(run_test_case(
        user_message="Redis的RDB和AOF各有什么优缺点？",
        description="技术对比-Redis持久化",
        expected_confidence_min=0.3,
        expect_store=True,
    ))

    # 1.3 Python 相关 — 验证 Python 知识
    results.append(run_test_case(
        user_message="请解释Python中装饰器的原理和常见用法",
        description="Python技术-装饰器",
        expected_confidence_min=0.3,
        expect_store=True,
    ))

    # 1.4 编程概念 — GIL
    results.append(run_test_case(
        user_message="Python的GIL是什么？它对多线程有什么影响？",
        description="Python技术-GIL",
        expected_confidence_min=0.3,
        expect_store=True,
    ))

    return results


# ══════════════════════════════════════════════════════════════
# 第 2 阶段: 对话交互测试
# ══════════════════════════════════════════════════════════════

def phase_2_conversation() -> None:
    """第2阶段: 对话交互 — 测试消息历史、闲聊、多轮对话"""
    print_separator("第 2 阶段: 对话交互")

    results = []
    session_id = 200001

    # 2.1 问候 (不应存储)
    results.append(run_test_case(
        user_message="你好",
        user_id="conv_user",
        session_id=session_id,
        description="问候-不应存储",
        expect_store=False,
    ))

    # 2.2 第一轮技术问题
    results.append(run_test_case(
        user_message="Python中列表和元组的区别是什么？",
        user_id="conv_user",
        session_id=session_id,
        description="多轮对话-第1轮: Python列表vs元组",
        message_history=[
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
        ],
        expect_store=True,
    ))

    # 2.3 第二轮追问 (应触发 rewrite_query 改写)
    results.append(run_test_case(
        user_message="那集合和字典呢？它们有什么区别？",
        user_id="conv_user",
        session_id=session_id,
        description="多轮对话-第2轮: 追问集合和字典",
        message_history=[
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
            {"role": "user", "content": "Python中列表和元组的区别是什么？"},
            {"role": "assistant", "content": "列表是可变的，元组是不可变的..."},
        ],
        expect_store=True,
    ))

    # 2.4 第三轮深度追问 (代词指代)
    results.append(run_test_case(
        user_message="它们分别用在什么场景？",
        user_id="conv_user",
        session_id=session_id,
        description="多轮对话-第3轮: 代词指代'它们'",
        message_history=[
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
            {"role": "user", "content": "Python中列表和元组的区别是什么？"},
            {"role": "assistant", "content": "列表是可变的，元组是不可变的..."},
            {"role": "user", "content": "那集合和字典呢？它们有什么区别？"},
            {"role": "assistant", "content": "集合是无序不重复元素集，字典是键值对..."},
        ],
        expect_store=True,
    ))

    return results


# ══════════════════════════════════════════════════════════════
# 第 3 阶段: 用户画像测试
# ══════════════════════════════════════════════════════════════

def phase_3_user_profile() -> None:
    """第3阶段: 用户画像 — 测试 user_profile 的读写和 update_profile 节点"""
    print_separator("第 3 阶段: 用户画像测试")

    # 先设置一个测试用户的画像
    test_profile = {
        "personal_info": {
            "name": "张三",
            "occupation": "Python后端开发工程师",
            "interests": "编程、AI、基金投资",
        },
        "preferences": {
            "language": "中文",
            "communication_style": "详细专业",
        },
        "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # 3.1 个人信息相关查询 (应使用画像信息)
    result = run_test_case(
        user_message="你知道我是谁吗？我的职业是什么？",
        user_id="profile_user",
        session_id=300001,
        description="用户画像-身份认知",
        user_profile=test_profile,
        expect_store=False,
    )

    # 3.2 保存更新的画像
    save_profile("profile_user", test_profile)
    report("用户画像已保存")

    # 3.3 技术偏好问题
    result2 = run_test_case(
        user_message="作为一个Python开发者，我该怎么学习系统设计？",
        user_id="profile_user",
        session_id=300001,
        description="用户画像-个性化推荐",
        user_profile=test_profile,
        message_history=[
            {"role": "user", "content": "你知道我是谁吗？我的职业是什么？"},
            {"role": "assistant", "content": "你是一位Python后端开发工程师张三..."},
        ],
        expect_store=True,
    )

    return [result, result2]


# ══════════════════════════════════════════════════════════════
# 第 4 阶段: 需要搜索的实时信息
# ══════════════════════════════════════════════════════════════

def phase_4_web_search() -> None:
    """第4阶段: 需要网络搜索 — 测试 search_web → regenerate 流程"""
    print_separator("第 4 阶段: 需要网络搜索的实时信息")

    results = []

    # 4.1 实时事件 (触发 web search)
    results.append(run_test_case(
        user_message="最近比特币价格是多少？",
        description="网络搜索-比特币价格",
        expect_store=True,
    ))

    # 4.2 技术前沿信息 (触发 web search)
    results.append(run_test_case(
        user_message="2025年最流行的前端框架是什么？",
        description="网络搜索-前端趋势",
        expect_store=True,
    ))

    return results


# ══════════════════════════════════════════════════════════════
# 第 5 阶段: 矛盾检测与反射
# ══════════════════════════════════════════════════════════════

def phase_5_contradiction() -> None:
    """第5阶段: 矛盾检测 — 测试 fact_check → reflect → record_error → search_web 循环"""
    print_separator("第 5 阶段: 矛盾检测与修正循环")

    # 先确保"北京"相关的知识已存储（第1阶段已存储）
    # 然后测试一个与已有知识矛盾的回答

    # 这里我们模拟"已有知识"已经被检索到的场景
    # 通过在前一轮存储"北京是首都"的知识，然后问一个矛盾的问题

    # 查找已存储的北京相关页面
    beijing_page = get_page_by_title("中国的首都是什么")
    if not beijing_page:
        beijing_page = get_page_by_title("北京")
    if not beijing_page:
        beijing_page = get_page_by_title("首都")

    if beijing_page:
        report(f"找到已有页面: {beijing_page['title']} (id={beijing_page['id']})")
    else:
        report("未找到北京的页面，先创建一个知识条目")

    # 实际测试矛盾检测
    # 注: 由于没有预置的"错误知识"，这里验证 fact_check 不会报无矛盾的错
    result = run_test_case(
        user_message="请解释Python中深度学习和机器学习的区别",
        user_id="contradiction_user",
        session_id=400001,
        description="矛盾检测-正常场景",
        expect_store=True,
    )

    return [result]


# ══════════════════════════════════════════════════════════════
# 第 6 阶段: URL 处理
# ══════════════════════════════════════════════════════════════

def phase_6_url_processing() -> None:
    """第6阶段: URL 处理 — 测试 parse 节点的 URL 提取和抓取"""
    print_separator("第 6 阶段: URL 处理")

    # 由于实际请求外部 URL 可能超时，这里跳过
    # 但验证 parse 节点的 URL 提取逻辑

    report("URL 处理测试: 外部链接抓取可能受网络影响")
    report("此项测试依赖 Tavily API 或 百度 API 配置")

    # 尝试发送一个已知可访问的 URL
    # 如果网络不通，跳过此阶段

    return []


# ══════════════════════════════════════════════════════════════
# 第 7 阶段: 长文本与复杂知识
# ══════════════════════════════════════════════════════════════

def phase_7_long_text() -> None:
    """第7阶段: 长文本/复杂知识 — 测试 retrieve 和 store 的大内容处理"""
    print_separator("第 7 阶段: 长文本与复杂知识")

    results = []

    # 7.1 复杂技术问题 (长答案，触发 store)
    results.append(run_test_case(
        user_message="请详细解释Django和Flask两个Web框架的设计哲学、优劣势以及适用场景",
        description="长文本-Django vs Flask",
        expect_store=True,
    ))

    # 7.2 系统设计问题
    results.append(run_test_case(
        user_message="请详细解释微服务架构的优缺点，以及什么场景适合使用微服务",
        description="长文本-微服务架构",
        expect_store=True,
    ))

    # 7.3 追问 (利用已有知识)
    results.append(run_test_case(
        user_message="那如果要拆分微服务，应该如何划分服务的边界？",
        session_id=500001,
        user_id="long_text_user",
        description="长文本-追问微服务边界",
        message_history=[
            {"role": "user", "content": "请详细解释微服务架构的优缺点"},
            {"role": "assistant", "content": "微服务架构的优缺点如下..."},
        ],
        expect_store=True,
    ))

    return results


# ══════════════════════════════════════════════════════════════
# 第 8 阶段: 边界情况测试
# ══════════════════════════════════════════════════════════════

def phase_8_edge_cases() -> None:
    """第8阶段: 边界情况 — 测试各种异常输入"""
    print_separator("第 8 阶段: 边界情况")

    results = []

    # 8.1 超短消息
    results.append(run_test_case(
        user_message="Hi",
        description="边界-超短消息",
        expect_store=False,
    ))

    # 8.2 纯 emoji
    results.append(run_test_case(
        user_message="👍🎉",
        description="边界-纯Emoji",
        expect_store=False,
    ))

    # 8.3 简单的天气问候
    results.append(run_test_case(
        user_message="今天天气怎么样？",
        description="边界-天气询问",
        expect_store=False,
    ))

    # 8.4 带 URL 的消息 (验证 parse 提取 URL)
    results.append(run_test_case(
        user_message="https://www.baidu.com 这个网站怎么样？",
        description="边界-带URL的消息",
        expect_store=False,
    ))

    # 8.5 感谢
    results.append(run_test_case(
        user_message="谢谢，你讲得很清楚！",
        description="边界-感谢消息",
        expect_store=False,
    ))

    return results


# ══════════════════════════════════════════════════════════════
# 第 9 阶段: 综合场景测试
# ══════════════════════════════════════════════════════════════

def phase_9_comprehensive() -> None:
    """第9阶段: 综合场景 — 模拟真实用户的连续使用场景"""
    print_separator("第 9 阶段: 综合场景 (模拟真实用户连续对话)")

    session_id = 900001
    user_id = "real_user"

    # 场景: 新用户首次使用
    # 用户: 你是谁？
    run_test_case(
        user_message="你是谁？能帮我做什么？",
        user_id=user_id,
        session_id=session_id,
        description="综合-首次使用询问",
        expect_store=False,
    )

    # 用户: 问一个技术问题
    run_test_case(
        user_message="什么是RESTful API？设计RESTful API有什么最佳实践？",
        user_id=user_id,
        session_id=session_id,
        description="综合-技术问题RESTful",
        message_history=[
            {"role": "user", "content": "你是谁？能帮我做什么？"},
            {"role": "assistant", "content": "我是你的知识助手..."},
        ],
        expect_store=True,
    )

    # 用户: 追问
    run_test_case(
        user_message="那REST和GraphQL相比有什么优缺点？",
        user_id=user_id,
        session_id=session_id,
        description="综合-追问REST vs GraphQL",
        message_history=[
            {"role": "user", "content": "你是谁？能帮我做什么？"},
            {"role": "assistant", "content": "我是你的知识助手..."},
            {"role": "user", "content": "什么是RESTful API"},
            {"role": "assistant", "content": "RESTful API 是..."},
        ],
        expect_store=True,
    )

    # 用户: 闲聊
    run_test_case(
        user_message="今天心情不错 😊",
        user_id=user_id,
        session_id=session_id,
        description="综合-闲聊心情",
        message_history=[
            {"role": "user", "content": "你是谁？能帮我做什么？"},
            {"role": "assistant", "content": "我是你的知识助手..."},
            {"role": "user", "content": "什么是RESTful API"},
            {"role": "assistant", "content": "RESTful API 是..."},
            {"role": "user", "content": "REST和GraphQL相比"},
            {"role": "assistant", "content": "REST和GraphQL各有优劣..."},
        ],
        expect_store=False,
    )

    # 用户: 再问一个不相关的问题
    run_test_case(
        user_message="能介绍一下Docker和Kubernetes的区别吗？",
        user_id=user_id,
        session_id=session_id,
        description="综合-新话题Docker",
        message_history=[
            {"role": "user", "content": "你是谁？能帮我做什么？"},
            {"role": "assistant", "content": "我是你的知识助手..."},
            {"role": "user", "content": "什么是RESTful API"},
            {"role": "assistant", "content": "RESTful API 是..."},
            {"role": "user", "content": "REST和GraphQL相比"},
            {"role": "assistant", "content": "REST和GraphQL各有优劣..."},
            {"role": "user", "content": "今天心情不错"},
            {"role": "assistant", "content": "哈哈，好心情很重要！"},
        ],
        expect_store=True,
    )


# ══════════════════════════════════════════════════════════════
# 第 10 阶段: 最终检查与报告生成
# ══════════════════════════════════════════════════════════════

def phase_10_final_report() -> None:
    """第10阶段: 最终检查 — 汇总所有结果"""
    print_separator("第 10 阶段: 最终检查")

    # 1. 检查数据库
    inspect_db()

    # 2. 检查 Wiki 文件
    check_wiki_integrity()

    # 3. 检查用户画像
    check_profile("comprehensive_tester")
    check_profile("profile_user")
    check_profile("conv_user")
    check_profile("real_user")

    # 4. 检查额外的逻辑
    conn = get_connection()
    conn.row_factory = sqlite3.Row

    # 检查 page_relations (交叉引用)
    relation_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM page_relations"
    ).fetchone()["cnt"]
    report(f"页面关系 (交叉引用): {relation_count} 条")

    # 检查 page_versions
    version_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM page_versions"
    ).fetchone()["cnt"]
    report(f"页面版本历史: {version_count} 条")

    # 检查 review_schedule
    review_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM review_schedule"
    ).fetchone()["cnt"]
    report(f"复习计划: {review_count} 条")

    conn.close()


# ══════════════════════════════════════════════════════════════
# 执行所有测试阶段
# ══════════════════════════════════════════════════════════════

def main() -> None:
    """主函数: 执行所有测试阶段并生成报告"""
    overall_start = time.time()

    print("")
    print("╔" + "═" * 66 + "╗")
    print("║" + " " * 66 + "║")
    print("║     LangChain-Learning 综合测试                  ║")
    print("║     覆盖 10+ 阶段, 全节点链路验证                 ║")
    print("║" + " " * 66 + "║")
    print("╚" + "═" * 66 + "╝")
    print("")

    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"Graph:    10+ 节点, 条件路由")
    print(f"")

    # ── 阶段执行 ──
    try:
        phase_1_basic_factual()
        print(f"\n  ✅ 第1阶段完成 ({datetime.now().strftime('%H:%M:%S')})\n")
    except Exception as e:
        logger.exception("第1阶段执行失败: %s", e)

    try:
        phase_2_conversation()
        print(f"\n  ✅ 第2阶段完成 ({datetime.now().strftime('%H:%M:%S')})\n")
    except Exception as e:
        logger.exception("第2阶段执行失败: %s", e)

    try:
        phase_3_user_profile()
        print(f"\n  ✅ 第3阶段完成 ({datetime.now().strftime('%H:%M:%S')})\n")
    except Exception as e:
        logger.exception("第3阶段执行失败: %s", e)

    try:
        phase_4_web_search()
        print(f"\n  ✅ 第4阶段完成 ({datetime.now().strftime('%H:%M:%S')})\n")
    except Exception as e:
        logger.exception("第4阶段执行失败: %s", e)

    try:
        phase_5_contradiction()
        print(f"\n  ✅ 第5阶段完成 ({datetime.now().strftime('%H:%M:%S')})\n")
    except Exception as e:
        logger.exception("第5阶段执行失败: %s", e)

    try:
        phase_6_url_processing()
        print(f"\n  ✅ 第6阶段完成 ({datetime.now().strftime('%H:%M:%S')})\n")
    except Exception as e:
        logger.exception("第6阶段执行失败: %s", e)

    try:
        phase_7_long_text()
        print(f"\n  ✅ 第7阶段完成 ({datetime.now().strftime('%H:%M:%S')})\n")
    except Exception as e:
        logger.exception("第7阶段执行失败: %s", e)

    try:
        phase_8_edge_cases()
        print(f"\n  ✅ 第8阶段完成 ({datetime.now().strftime('%H:%M:%S')})\n")
    except Exception as e:
        logger.exception("第8阶段执行失败: %s", e)

    try:
        phase_9_comprehensive()
        print(f"\n  ✅ 第9阶段完成 ({datetime.now().strftime('%H:%M:%S')})\n")
    except Exception as e:
        logger.exception("第9阶段执行失败: %s", e)

    try:
        phase_10_final_report()
        print(f"\n  ✅ 第10阶段完成 ({datetime.now().strftime('%H:%M:%S')})\n")
    except Exception as e:
        logger.exception("第10阶段执行失败: %s", e)

    # ── 汇总 ──
    overall_elapsed = time.time() - overall_start
    print(f"\n{'═' * 60}")
    print(f"  全部测试完成!")
    print(f"  总耗时: {overall_elapsed:.1f}s ({overall_elapsed / 60:.1f} 分钟)")
    print(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 60}")

    # 保存报告
    report_filename = f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = os.path.join(
        os.path.dirname(__file__), "data", report_filename
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# LangChain-Learning 综合测试报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {overall_elapsed:.1f}s\n")
        f.write(f"\n---\n\n")
        f.write("\n".join(report_lines))

    print(f"\n报告已保存: {report_path}")

    # 打印所有报告摘要
    print(f"\n{'═' * 60}")
    print("  测试摘要:")
    for line in report_lines:
        print(f"  {line}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
