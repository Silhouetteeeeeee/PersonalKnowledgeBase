import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.models import (
    save_knowledge_points_bulk_with_embeddings,
    ensure_category,
    find_similar_knowledge,
    get_normalized_categories,
    normalize_category_str,
)

logger = logging.getLogger(__name__)

class DistilledPoint(BaseModel):
    knowledge_text: str = Field(
        description="A concise, standalone knowledge point distilled from the Q&A"
    )
    category: str = Field(
        description="Category for the knowledge point. Each knowledge point has different category."
                    "Support up to four-tier hierarchical categories (e.g., databases/nosql/redis/commands)"
    )
    tags: list[str] = Field(description="Relevant tags for this knowledge point")


class DistillOutput(BaseModel):
    category: str = Field(
        description="Category for the knowledge. "
                    "Support up to four-tier hierarchical categories (e.g., databases/nosql/redis/commands)"
    )
    knowledge_points: list[DistilledPoint] = Field(
        description="Knowledge points distilled from the Q&A"
    )

def _check_duplicate(kp) -> tuple:
    """检查知识点是否重复，返回 (kp, is_duplicate)"""
    try:
        similar = find_similar_knowledge(kp.knowledge_text, threshold=0.25)
        if similar:
            logger.info("Skipping duplicate knowledge: '%s' (distance=%.3f)",
                       kp.knowledge_text[:50], similar[0].get("distance", 0))
            return kp, True
    except Exception as e:
        logger.warning("Dedup embedding failed for '%s': %s, saving without dedup",
                      kp.knowledge_text[:30], e)
    return kp, False

def store(state: dict) -> dict:
    if not state.get("needs_store", True):
        logger.info("Skipping store: needs_store=False")
        return {}

    if not state.get("answer"):
        logger.info("Skipping store: no answer")
        return {}

    if state.get("contradiction_found"):
        logger.info("Skipping store: contradiction detected")
        return {}

    logger.info("Distilling knowledge from Q&A...")
    existing_cats = get_normalized_categories()
    prompt = (
        '''## 角色与核心任务
            你是一名专业的知识内容分类专家，核心任务是**对用户输入的知识类问题/内容，输出标准化、层级递进的分类路径**，分类路径格式为`一级分类/二级分类/三级分类/...`，层级深度与内容的细分程度完全匹配，严格遵循本提示词的所有规则执行。
            
            ---
            
            ## 核心分类原则
            1.  **层级递进原则**：分类路径必须从通用大类到细分领域逐级下沉，每一层级均为上一层级的精准子集，禁止跳级、反向层级、逻辑混乱。
            2.  **精准匹配原则**：内容越具体，分类层级越深；仅泛化提问时，分类到对应层级即可，禁止无意义的层级叠加。
            3.  **术语标准化原则**：所有分类标签统一使用**通用、规范、无歧义的小写英文术语**，无空格、无特殊字符、无中英文混用，符合行业通用命名习惯。
            4.  **核心主题优先原则**：若输入包含多主题内容，以占比最高、最核心的知识主题为分类基准；若多主题权重相当，用`&`分隔并列一级分类。
            5.  **兜底原则**：无法归入预设分类体系的内容，统一归入`other`分类。
            
            ---
            
            ## 标准化分类体系
            ### 固定一级分类（禁止自创一级分类，必须从以下列表中选择）
            | 一级分类 | 覆盖范围 |
            |----------|----------|
            | programming | 所有编程开发、计算机语言、算法、开发框架、软件工程、IT技术相关内容 |
            | mathematics | 数学、代数、几何、统计、概率论、数论、微积分等数学相关内容 |
            | physics | 物理学、力学、电磁学、量子物理、热力学、光学等物理相关内容 |
            | chemistry | 化学、有机化学、无机化学、分析化学、材料化学等化学相关内容 |
            | biology | 生物学、生命科学、遗传学、生态学、生物工程等生物相关内容 |
            | history | 历史、考古、世界史、中国史、近代史、历史事件与人物等相关内容 |
            | literature | 文学、小说、诗歌、散文、文学理论、作家作品、文学评论等相关内容 |
            | art | 艺术、绘画、音乐、舞蹈、影视、设计、雕塑、书法、美学等相关内容 |
            | philosophy | 哲学、逻辑学、伦理学、哲学史、哲学流派与思想家等相关内容 |
            | economics | 经济学、宏观/微观经济、金融、贸易、财税、投资理财等相关内容 |
            | law | 法律、法学、法规、司法、法律实务、法律职业等相关内容 |
            | medicine | 医学、药学、临床医学、基础医学、健康护理、疾病防控等相关内容 |
            | education | 教育、教学、考试、学习方法、校园、升学、职业教育等相关内容 |
            | career | 职场、求职、职业发展、企业管理、办公技能、职场沟通等相关内容 |
            | life | 生活、家居、美食、旅行、穿搭、日常技能、生活常识等相关内容 |
            | sports | 体育、运动、赛事、健身、户外、体育理论与技巧等相关内容 |
            | personal | 记录用户个人信息，以及生活习惯等相关内容 |
            | other | 无法归入以上所有分类的非知识类、无意义、闲聊类内容 |
            
            ### 二级及以下层级细分规则（核心贴合你的业务需求）
            1.  针对`programming`一级分类，**二级分类固定为编程语言/技术栈名称**，例如：`python`、`java`、`javascript`、`c`、`go`、`linux`等；
            2.  三级分类为该语言/技术栈下的核心细分领域，例如：`web`、`data-analysis`、`ai`、`network`、`microservice`、`game-development`、`automation`等；
            3.  四级及更深层级为该细分领域下的具体框架、库、技术点，例如：`django`、`flask`、`pandas`、`requests`、`springboot`等，直到无法再精准细分为止。
            
            ---
            
            ## 严格执行步骤
            1.  主题提纯：识别用户输入以及回答的核心知识主题，过滤语气词、修饰词、冗余表述，精准锁定核心知识点；
            2.  一级分类匹配：从预设的固定一级分类列表中，匹配与核心主题对应的一级分类，禁止自创；
            3.  层级下沉拆解：基于核心主题的细分程度，向下逐级拆解细分层级，每一层级必须是上一层级的子集，逻辑闭环；
            4.  合规校验：检查分类路径的术语规范、层级逻辑、格式标准，剔除所有不符合规则的内容；
            5.  结果输出：严格按照输出规范，仅输出最终分类路径。
            
            ---
            
            ## 输出规范
            1.  仅输出最终的分类路径字符串，禁止输出任何解释、说明、前缀、后缀、标点符号等额外内容；
            2.  分类路径使用`/`分隔层级，全小写英文，无空格、无特殊字符；
            3.  层级数量无强制上限，仅以内容的精准细分程度为准，禁止为了增加层级而添加无意义标签。
            
            ---
            
            ## 示例参考
            | 用户输入 | 标准输出结果 |
            |----------|--------------|
            | 什么是python | programming/python |
            | python的网络编程怎么入门 | programming/python/web |
            | python中django框架的路由配置方法 | programming/python/web/django |
            | python用pandas做数据清洗的技巧 | programming/python/data-analysis/pandas |
            | java的springboot微服务开发实战 | programming/java/microservice/springboot |
            | 微积分的极限定理详解 | mathematics/calculus/limit |
            | 唐朝安史之乱的历史影响 | history/chinese-history/tang-dynasty/an-shi-rebellion |
            | 红楼梦的人物形象分析 | literature/chinese-literature/classical-novel/hong-lou-meng |
            | 零基础怎么学编程 | programming |
            | 我的姓名叫XXX | personal/info |
            | 今天天气怎么样 | other |
            
            ---
            
            ## 边界情况处理规则
            1.  多主题输入：优先按核心主题分类，若多主题权重相当，用`&`分隔并列一级分类，例如输入“python和java的区别，以及考研数学复习方法”，输出`programming&mathematics`；
            2.  模糊泛化输入：仅能匹配到一级分类时，仅输出一级分类，例如输入“编程怎么学”，输出`programming`；
            3.  极端细分输入：拆解到最细分的技术点/知识点为止，例如输入“python requests库的get请求超时设置”，输出`programming/python/web/requests`；
            4.  非知识类输入：闲聊、无意义内容、指令类非知识提问，统一输出`other`,除了用户输入的是个人信息,需要进行保存并记录。
            
            ---
            
            ## 额外要求（可以不强制要求提炼）
            1.  本系统作为个人知识库，本质是为个人使用，需要保留用户个人信息，生活习惯等等
            2.  若从用户的提问能分析出用户的个人特质或者生活习惯，请也提取出来作为一个知识点，比如
                 - 用户说："我叫李明，是一名后端工程师，主要用Go语言"
                    - 提取："用户姓名：李明" (personal/identity)
                    - 提取："用户职业：后端工程师" (personal/career)
                    - 提取："用户主要技术栈：Go语言" (personal/preferences)
                 - 用户在凌晨2点问："这个bug怎么调试？明天要上线了"
                    - 提取："用户经常在深夜工作/学习" (personal/habits)
                    - 提取："用户工作压力较大，有紧急交付任务" (personal/career)
            3.  如果用户没有特别的语气，请不要随便提取用户信息，只有当你确定了，这句话展示了用户某种特征才会提取。
        '''
        f"Question: {state['user_message']}\n"
        f"Answer: {state['answer']}"
    )
    if existing_cats:
        prompt += (f"\n\n已有分类：{existing_cats}\n请优先选择最匹配的已有分类，仅当完全不匹配时创建新分类。"
                   f"如果当前知识点属于某个已有分类的子类别，请使用多级分类格式（如：）")
    result = LLM.generate_structured(prompt, DistillOutput, use_language=False)

    for d in result.knowledge_points:
        d.category = normalize_category_str(d.category)
        ensure_category(d.category)

    # Dedup: skip knowledge points that are semantically similar to existing ones
    new_points = []
    # 并行去重检查
    with ThreadPoolExecutor(max_workers=16, thread_name_prefix="dedup") as executor:
        futures = {executor.submit(_check_duplicate, kp): kp
                   for kp in result.knowledge_points}

        # 收集非重复的知识点
        new_points = []
        for future in as_completed(futures):
            kp, is_duplicate = future.result()
            if not is_duplicate:
                new_points.append(kp)

    if not new_points:
        logger.info("All knowledge points already exist, nothing to store")
        return {}

    reasoning_log_path = state.get("reasoning_log_path", "")
    knowledge_points = [
        {
            "knowledge_text": kp.knowledge_text,
            "source_question": state["user_message"],
            "category": kp.category,
            "tags": kp.tags,
            "reasoning_log_path": reasoning_log_path,
        }
        for kp in new_points
    ]
    ids = save_knowledge_points_bulk_with_embeddings(knowledge_points)

    return {
        "category": result.category,
        "stored_knowledge_ids": ids,
        "logic_chain": [{
            "node": "store",
            "action": f"存储 {len(ids)} 条知识点",
            "reasoning": f"分类: {result.category}, 知识点数: {len(ids)}",
        }],
    }
