from fairflex import data
from fairflex.data import ACNDataClient, NSRDBDataClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_acn_ping_uses_basic_token_auth_without_returning_the_token(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse({"_items": [{"sessionID": "one"}]})

    monkeypatch.setattr(data.requests, "get", fake_get)
    assert ACNDataClient("test-token").ping("caltech") == 1
    assert captured["auth"] == ("test-token", "")
    assert captured["url"].endswith("sessions/caltech")


def test_nsrdb_discovery_sends_a_key_only_as_a_request_parameter(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse({"errors": [], "outputs": [{"name": "dataset"}]})

    monkeypatch.setattr(data.requests, "get", fake_get)
    result = NSRDBDataClient("test-key").discover(17.385, 78.4867)
    assert result["outputs"][0]["name"] == "dataset"
    assert captured["params"]["api_key"] == "test-key"
    assert "api_key" not in captured["url"]
