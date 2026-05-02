def parse(state: dict) -> dict:
    return {
        "user_message": state["user_message"].strip(),
        "user_id": state.get("user_id", "unknown"),
        "timestamp": state.get("timestamp", ""),
    }
