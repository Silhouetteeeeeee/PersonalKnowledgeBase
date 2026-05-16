import logging

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.profile import load_profile, save_profile, update_profile_field

logger = logging.getLogger(__name__)


class ProfileUpdate(BaseModel):
    field: str = Field(
        description="Dot-notation field path, e.g. 'identity.name', 'preferences.tech_stack'"
    )
    value: object = Field(
        description="The value to store. Use string for single values, list for multiple items."
    )
    should_update: bool = Field(
        description="True if the conversation contains new or changed personal information worth saving"
    )


class ProfileOutput(BaseModel):
    profiles: list[ProfileUpdate] = Field(
        default_factory=list,
        description="List of profile updates to apply"
    )


def update_profile(state: dict) -> dict:
    """Extract personal information from the conversation and update user profile."""
    user_message = state.get("user_message", "")
    answer = state.get("answer", "")
    user_id = state.get("user_id", "")

    if not user_message or not user_id:
        return {"user_profile": state.get("user_profile", load_profile())}

    profile = load_profile(user_id)
    prompt = (
        f"分析以下对话，判断用户是否在谈论个人信息（姓名、职业、生活习惯、学习计划、偏好等）。\n\n"
        f"用户消息：{user_message}\n"
        f"回答：{answer}\n"
        f"这是当前用户的信息结构：{str(profile)}\n\n"
        f"如果包含值得记录的个人信息，提取所有相关的字段路径和值，生成多个更新项。否则返回空的 updates 列表。"
    )

    try:
        result = LLM.generate_structured(prompt, ProfileOutput, use_language=False)
    except Exception as e:
        logger.warning("Profile extraction failed: %s", e)
        return {"user_profile": state.get("user_profile", load_profile())}

    updated_fields = [p.field for p in result.profiles if p.should_update]
    for p in result.profiles:
        if p.should_update:
            logger.info("Profile field: %s, value: %s", p.field, p.value)
            profile = update_profile_field(profile, p.field, p.value)
    save_profile(profile, user_id)

    if updated_fields:
        logger.info("Updated profile fields: %s", updated_fields)
        return {"user_profile": profile, "logic_chain": [{
            "node": "update_profile",
            "action": f"更新 {len(updated_fields)} 个画像字段",
            "reasoning": f"字段：{', '.join(updated_fields)}",
        }]}
    return {"user_profile": profile}
