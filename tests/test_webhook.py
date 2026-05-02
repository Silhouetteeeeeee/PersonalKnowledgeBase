from server.webhook import app


def test_health():
    with app.test_client() as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"


def test_webhook_get_returns_echostr():
    with app.test_client() as client:
        resp = client.get("/webhook?echostr=hello123")
        assert resp.status_code == 200
        assert resp.data.decode() == "hello123"


def test_webhook_post_no_data():
    with app.test_client() as client:
        resp = client.post("/webhook", content_type="application/json", data="{}")
        assert resp.status_code == 400


def test_webhook_post_empty_content():
    with app.test_client() as client:
        resp = client.post("/webhook", json={"content": ""})
        assert resp.status_code == 200
