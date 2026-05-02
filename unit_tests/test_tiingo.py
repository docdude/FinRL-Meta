import pytest
import requests

from meta.data_processors.tiingo import Tiingo


class _MockResponse:
    def __init__(self, status_code: int, payload: dict, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = str(payload)

    def json(self):
        return self._payload


class _MockSession:
    def __init__(self, response):
        self._response = response

    def get(self, *args, **kwargs):
        return self._response


def test_tiingo_fetch_ticker_surfaces_rate_limit_detail() -> None:
    processor = Tiingo(
        data_source="tiingo",
        start_date="2010-01-01",
        end_date="2026-01-01",
        time_interval="1d",
        TIINGO_API_KEY="dummy",
        max_retry=1,
    )
    processor.session = _MockSession(
        _MockResponse(
            429,
            {
                "detail": (
                    "Error: You have run over your hourly request allocation."
                )
            },
            headers={"Retry-After": "3600"},
        )
    )

    with pytest.raises(requests.HTTPError) as exc_info:
        processor._fetch_ticker("AAPL")

    message = str(exc_info.value)
    assert "AAPL" in message
    assert "429" in message
    assert "hourly request allocation" in message
    assert "Retry-After: 3600" in message