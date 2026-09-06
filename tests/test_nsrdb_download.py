from fairflex import data
from fairflex.data import NSRDBDataClient


class FakeResponse:
    content = b"metadata\nYear,Month,Day,Hour,GHI\n2019,1,1,0,0\n"

    def raise_for_status(self):
        return None


def test_nsrdb_download_url_requires_a_real_contact_email(monkeypatch):
    monkeypatch.delenv("NSRDB_EMAIL", raising=False)
    client = NSRDBDataClient("test-key")
    try:
        client.download_url(
            dataset="nsrdb-GOES-conus-v4-0-0",
            year=2019,
            interval_minutes=15,
            latitude=34.1,
            longitude=-118.1,
        )
    except ValueError as error:
        assert "NSRDB_EMAIL" in str(error)
    else:
        raise AssertionError("missing contact email should be rejected")


def test_nsrdb_download_writes_the_exact_response_to_requested_raw_path(monkeypatch, tmp_path):
    monkeypatch.setattr(data.requests, "get", lambda *args, **kwargs: FakeResponse())
    output = tmp_path / "weather.csv"
    client = NSRDBDataClient("test-key")
    target = client.download_csv(
        output,
        dataset="nsrdb-GOES-conus-v4-0-0",
        year=2019,
        interval_minutes=15,
        latitude=34.1,
        longitude=-118.1,
        email="researcher@example.com",
    )
    assert target == output
    assert output.read_bytes() == FakeResponse.content
