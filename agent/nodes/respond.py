def respond(state: dict) -> dict:
    return {"final_response": state.get("answer", "")}
