import logging

from pydantic import BaseModel, Field

from agent.utils.llm import LLM
from storage.profile import load_profile, save_profile, update_profile_field

logger = logging.getLogger(__name__)


class ProfileUpdate(BaseModel):
    """Extracted personal information from the conversation."""
    field: str = Field(
        description="Dot-notation field path, e.g. 'identity.name', 'plans.current_study', 'preferences.tech_stack'"
    )
    value: object = Field(
        description="The value to store. Use string for single values, list for multiple items."
    )
    should_update: bool = Field(
        description="True if the conversation contains new or changed personal information worth saving"
    )


def update_profile(state: dict) -> dict:
    """Extract personal information from the conversation and update profile."""
    user_message = state.get("user_message", "")
    answer = state.get("answer", "")

    if not user_message:
        return {"user_profile": state.get("user_profile", load_profile())}

    prompt = (
        f"分析以下对话，判断用户是否在谈论个人信息（姓名、职业、生活习惯、学习计划、偏好等）。\n\n"
        f"用户消息：{user_message}\n"
        f"回答：{answer}\n\n"
        f"如果包含值得记录的个人信息，提取字段路径和值。否则 should_update=false。"
    )

    try:
        result = LLM.generate_structured(prompt, ProfileUpdate, use_language=False)
    except Exception as e:
        logger.warning("Profile extraction failed: %s", e)
        return {"user_profile": state.get("user_profile", load_profile())}

    if not result.should_update:
        logger.info("No personal info detected in conversation")
        return {"user_profile": state.get("user_profile", load_profile())}

    profile = load_profile()
    profile = update_profile_field(profile, result.field, result.value)
    save_profile(profile)

    logger.info("Profile updated: %s = %s", result.field, result.value)
    return {"user_profile": profile}
