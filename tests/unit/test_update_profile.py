import pytest


def test_update_profile_no_pii():
    """No personal info → should_update=false, profile unchanged."""
    from agent.nodes.update_profile import update_profile

    result = update_profile({
        "user_message": "What is Python?",
        "answer": "Python is a programming language.",
    })
    assert "user_profile" in result


def test_update_profile_extracts_name(monkeypatch):
    """Simulate LLM extracting a name from the conversation."""
    from agent.nodes.update_profile import update_profile, ProfileUpdate, ProfileOutput

    class FakeLLM:
        @staticmethod
        def generate_structured(prompt, output_model, **kwargs):
            return ProfileOutput(profiles=[
                ProfileUpdate(field="identity.name", value="李明", should_update=True),
            ])

    monkeypatch.setattr("agent.nodes.update_profile.LLM", FakeLLM)

    result = update_profile({
        "user_message": "我叫李明",
        "answer": "你好李明！",
    })
    assert result["user_profile"]["identity"]["name"] == "李明"
