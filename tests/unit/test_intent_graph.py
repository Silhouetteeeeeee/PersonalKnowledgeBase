"""Tests for intent-related graph structure and routing."""

from agent.graph import build_graph, intent_router


class TestGraphStructure:
    """Verify the graph has the new intent nodes."""

    def test_graph_has_classify_intent_node(self):
        g = build_graph()
        node_names = list(g.nodes.keys())
        assert "classify_intent" in node_names

    def test_graph_has_intent_handler_node(self):
        g = build_graph()
        assert "intent_handler" in g.nodes

    def test_graph_branches_include_intent(self):
        """Compiled graph's trigger_to_nodes should include intent nodes."""
        g = build_graph()
        triggers = list(g.trigger_to_nodes.keys())
        assert any("classify_intent" in str(t) for t in triggers)
        assert any("intent_handler" in str(t) for t in triggers)

    def test_graph_has_parse_as_entry(self):
        """parse is still in the node list (entry point)."""
        g = build_graph()
        assert "parse" in g.nodes

    def test_knowledge_qa_path_nodes_exist(self):
        """All knowledge_qa path nodes still exist."""
        g = build_graph()
        required_nodes = {"rewrite_query", "retrieve", "classify_and_answer",
                          "fact_check", "reflect", "respond"}
        for node in required_nodes:
            assert node in g.nodes, f"Missing required node: {node}"


class TestIntentRouter:
    """Verify the intent_router function routing logic."""

    def test_knowledge_qa_goes_to_rewrite_query(self):
        assert intent_router({"intent": "knowledge_qa"}) == "rewrite_query"

    def test_chitchat_goes_to_intent_handler(self):
        assert intent_router({"intent": "chitchat"}) == "intent_handler"

    def test_link_handling_goes_to_intent_handler(self):
        assert intent_router({"intent": "link_handling"}) == "intent_handler"

    def test_personal_info_goes_to_intent_handler(self):
        assert intent_router({"intent": "personal_info"}) == "intent_handler"

    def test_error_feedback_goes_to_intent_handler(self):
        assert intent_router({"intent": "error_feedback"}) == "intent_handler"

    def test_low_confidence_goes_to_intent_handler(self):
        assert intent_router({"intent": "low_confidence"}) == "intent_handler"

    def test_learning_plan_goes_to_intent_handler(self):
        assert intent_router({"intent": "learning_plan"}) == "intent_handler"

    def test_todo_goes_to_intent_handler(self):
        assert intent_router({"intent": "todo"}) == "intent_handler"

    def test_empty_defaults_to_knowledge_qa(self):
        assert intent_router({}) == "rewrite_query"
        assert intent_router({"intent": ""}) == "rewrite_query"
        assert intent_router({"intent": None}) == "rewrite_query"

    def test_unknown_defaults_to_knowledge_qa(self):
        assert intent_router({"intent": "unknown_thing"}) == "intent_handler"
