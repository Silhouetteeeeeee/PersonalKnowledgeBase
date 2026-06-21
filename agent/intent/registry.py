"""Intent definitions — single source of truth for all intent types.

Central registry: adding a new intent means adding one entry here
and one handler function in handlers.py — zero graph changes.
"""

from agent.intent.schemas import IntentNode, ParamSchema


# ════════════════════════════════════════════════════════════════════
# Intent definitions (used by classifier prompt + handlers)
# ════════════════════════════════════════════════════════════════════

INTENTS: list[IntentNode] = [
    IntentNode(
        id="knowledge_qa",
        name="知识问答",
        description="用户询问事实性、教育性问题，需要检索知识库或搜索网络来回答",
        params=[ParamSchema(name="query", type="string", description="用户的核心问题")],
        examples=[
            "Python的装饰器是什么",
            "Redis的持久化机制有哪些",
            "解释一下CAP定理",
            "如何优化SQL查询性能",
            "Docker和虚拟机有什么区别",
        ],
    ),
    IntentNode(
        id="chitchat",
        name="闲聊问候",
        description="用户打招呼、寒暄、表达情绪或进行日常交流，不需要检索知识",
        examples=[
            "你好",
            "早上好",
            "今天心情不错",
            "谢谢你的帮助",
            "再见",
        ],
    ),
    IntentNode(
        id="knowledge_mgmt",
        name="知识管理",
        description="用户想管理知识库，如查看已有知识、删除或更新知识页面",
        params=[ParamSchema(name="action", type="string", description="管理操作类型")],
        examples=[
            "帮我看看我有哪些知识",
            "删除关于Redis的知识",
            "更新Python相关的页面",
            "整理一下我的知识库",
        ],
    ),
    IntentNode(
        id="personal_info",
        name="个人信息",
        description="用户在谈论自己的个人信息，如工作、学习背景、兴趣偏好、计划目标等",
        params=[
            ParamSchema(name="info_type", type="string", description="信息类别", required=False),
        ],
        examples=[
            "我在学Python",
            "我是做后端开发的",
            "我打算考CFA",
            "我喜欢打篮球",
            "我目前在深圳工作",
        ],
    ),
    IntentNode(
        id="link_handling",
        name="链接处理",
        description="用户发送了一个或多个网页链接，需要总结、分析或提取网页内容",
        params=[ParamSchema(name="url_count", type="integer", description="链接数量", required=False)],
        examples=[
            "https://example.com/article",
            "帮我看看这篇文章 https://blog.xxx.com/post",
            "总结一下这个网页 https://news.xxx.com",
            "这几个链接讲了什么 https://a.com https://b.com",
        ],
    ),
    IntentNode(
        id="error_feedback",
        name="错误反馈",
        description="用户指出之前的回答有错误、不准确，或者要求修正某个回答",
        params=[ParamSchema(name="error_description", type="string", description="用户指出的错误描述")],
        examples=[
            "你刚才说的不对",
            "上次的回答有误",
            "纠正一下，Python的GIL不是那个意思",
            "你搞错了，Redis是单线程的",
        ],
    ),
    IntentNode(
        id="learning_plan",
        name="学习计划",
        description="用户询问学习路线、计划建议、资源推荐或成长路径",
        params=[ParamSchema(name="topic", type="string", description="学习主题")],
        examples=[
            "怎么学机器学习",
            "Go语言学习路线",
            "推荐几本系统设计的书",
            "零基础怎么学Python",
        ],
    ),
    IntentNode(
        id="todo",
        name="待办事项",
        description="用户想记录、查询或管理待办事项和提醒",
        params=[ParamSchema(name="action", type="string", description="待办操作", required=False)],
        examples=[
            "帮我记一下明天开会",
            "我的待办有哪些",
            "提醒我下周三交报告",
            "帮我设置一个提醒",
        ],
    ),
]


def validate_registry(handler_ids: set[str]) -> list[str]:
    """Verify every INTENT has a matching handler, and vice versa.

    Args:
        handler_ids: Set of intent IDs that have handlers in HANDLER_MAP.

    Returns:
        List of warning strings (empty = all good).
    """
    warnings = []
    intent_ids = {n.id for n in INTENTS}
    handler_ids = handler_ids.copy()  # don't mutate caller's set

    # Special entries not in INTENTS: knowledge_qa uses standard graph path
    handler_ids.discard("low_confidence")

    # knowledge_qa doesn't need a HANDLER_MAP entry (uses standard graph path)
    intent_ids.discard("knowledge_qa")

    missing_handlers = intent_ids - handler_ids
    extra_handlers = handler_ids - intent_ids
    if missing_handlers:
        warnings.append(f"INTENTS missing handlers: {missing_handlers}")
    if extra_handlers:
        warnings.append(f"HANDLER_MAP entries without INTENT: {extra_handlers}")
    return warnings
