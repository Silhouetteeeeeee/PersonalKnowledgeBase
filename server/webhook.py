import logging
from flask import Flask, request, jsonify

from server.config import WEWORK_TOKEN
from agent.graph import build_graph

logger = logging.getLogger(__name__)

app = Flask(__name__)
graph = build_graph()


@app.route("/webhook", methods=["GET"])
def verify_url():
    """WeChat Work URL verification (GET request)."""
    msg_signature = request.args.get("msg_signature", "")
    timestamp = request.args.get("timestamp", "")
    nonce = request.args.get("nonce", "")
    echostr = request.args.get("echostr", "")
    return echostr


@app.route("/webhook", methods=["POST"])
def handle_message():
    """Handle incoming WeChat Work message."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"code": 400, "msg": "invalid request"}), 400

        content = data.get("content", "")
        user_id = data.get("userid", "unknown")

        if not content:
            return jsonify({"code": 0, "msg": "ok"})

        result = graph.invoke({
            "user_message": content,
            "user_id": user_id,
            "timestamp": str(request.args.get("timestamp", "")),
        })

        return jsonify({
            "code": 0,
            "msg": "ok",
            "response": result.get("final_response", ""),
        })

    except Exception as e:
        logger.exception("Error processing message")
        return jsonify({"code": 500, "msg": "internal error"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def run_server():
    from server.config import FLASK_HOST, FLASK_PORT
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)
