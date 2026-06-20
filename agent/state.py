from typing import Annotated
from typing_extensions import TypedDict
import operator

from agent.models.value_objects import UrlContent, LogicChainStep, StoredKnowledge


class AgentState(TypedDict):
    """LangGraph 的全局状态定义，贯穿所有节点。

    字段按功能分组：
    - 用户输入与上下文
    - 检索与知识
    - 回答与推理
    - 矛盾检测与修正
    - 控制流
    """

    # ── 用户输入与上下文 ──
    user_message: str
    search_query: str                          # LLM 改写的独立检索查询
    user_id: str
    timestamp: str
    session_id: str
    message_history: list[dict]                # 当前会话的近期消息
    episodic_memories: list[str]               # 跨会话情景记忆
    user_profile: dict                         # 用户画像 JSON

    # ── URL 内容 ──
    url_contents: list[UrlContent]             # 从消息中提取的 URL 抓取结果

    # ── 检索与知识 ──
    stored_knowledge: list[StoredKnowledge]    # 检索到的 wiki 页面
    stored_knowledge_ids: list[int]
    wiki_page_ids: list[int]

    # ── 回答与推理 ──
    confidence: float                          # 置信度 [0.0, 1.0]
    needs_store: bool                          # 是否需要存储为 wiki 页面
    answer: str                                # 原始回答
    final_response: str                        # 最终响应（含矛盾警告等附加信息）
    search_results: list[str]                  # 网络搜索结果
    search_time: int                           # 当前请求中已搜索次数
    logic_chain: Annotated[list[LogicChainStep], operator.add]  # 推理链路

    # ── 矛盾检测 ──
    contradiction_found: bool
    contradiction_details: str
    contradiction_severity: str
    contradiction_knowledge_ids: list[int]
    contradiction_knowledge_texts: list[str]

    # ── 矛盾修正 ──
    reflection_result: str                     # stored_knowledge_wrong | answer_wrong | unresolved
    reflection_reasoning: str
    reflection_correction: str
    force_web_search: bool
    correction_attempts: int                   # 当前已修正次数（上限 2）
    error_recorded: bool
