from fastapi.testclient import TestClient

from valendra_privacy_filter.api.app import create_app
from valendra_privacy_filter.inference.detector import Detector
from test_inference import CharacterModel, CharacterTokenizer


def test_endpoints_batch_limits_and_sensitive_log_exclusion(caplog):
    detector = Detector(CharacterModel(), CharacterTokenizer(), max_length=8, overlap=2)
    app = create_app(detector=detector, max_chars=32, max_batch=2, max_body_bytes=256)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        result = client.post("/detect", json={"text": "Hi Ada!"})
        assert result.status_code == 200
        assert result.json()["entities"][0]["value"] == "Ada"
        assert client.post("/redact", json={"text": "Hi Ada!"}).json()["text"] == "Hi [PERSON]!"
        batch = client.post("/redact", json={"texts": ["Hi Ada!", "Hi!"]})
        assert [item["text"] for item in batch.json()["results"]] == ["Hi [PERSON]!", "Hi!"]
        assert client.post("/detect", json={"texts": ["a", "b", "c"]}).status_code == 413
        assert client.post("/detect", json={"text": "x" * 33}).status_code == 413
        assert client.post("/detect", json={"text": "x" * 300}).status_code == 413
        assert client.post("/detect", json={"text": "Ada", "texts": ["Ada"]}).status_code == 422
    assert "Ada" not in caplog.text
