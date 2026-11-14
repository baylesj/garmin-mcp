"""HRV history, the HRV status headline, overnight HRV in sleep, and long resting-HR ranges."""

from __future__ import annotations

import datetime as dt
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

import garmin_mcp.server as server_module
from garmin_mcp.cache import NightStore
from garmin_mcp.garmin_client import GarminClientError
from tests.test_tools import FakeGarminClient


@pytest.fixture(autouse=True)
async def _reset_state() -> Any:
    await server_module.get_cache_for_testing().clear()
    server_module.set_garmin_client_for_testing(None)
    yield
    server_module.set_garmin_client_for_testing(None)


def _summary(day: str, avg: float = 30.0, status: str = "BALANCED") -> dict[str, Any]:
    return {
        "calendarDate": day,
        "weeklyAvg": 31,
        "lastNightAvg": avg,
        "lastNight5MinHigh": 45,
        "baseline": {"lowUpper": 28, "balancedLow": 30, "balancedUpper": 42},
        "status": status,
        "feedbackPhrase": "HRV_BALANCED_2",
    }


def _range_responder(skip: set[str] = frozenset()) -> Any:  # type: ignore[assignment]
    def respond(start: str, end: str) -> dict[str, Any]:
        s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
        days = [(s + timedelta(days=i)).isoformat() for i in range((e - s).days + 1)]
        return {"hrvSummaries": [_summary(d) for d in days if d not in skip]}

    return respond


@pytest.mark.asyncio
async def test_hrv_history_chunks_and_maps_fields() -> None:
    fake = FakeGarminClient({"get_hrv_data_range": _range_responder({"2026-08-20"})})
    server_module.set_garmin_client_for_testing(fake)
    result = await server_module.get_hrv_history("2026-08-01", "2026-11-28")  # 120 nights

    assert len(result.nights) == 120
    assert result.nights_with_data == 119
    windows = [c[1] for c in fake.call_log]
    assert windows == [("2026-08-01", "2026-10-29"), ("2026-10-30", "2026-11-28")]
    first = result.nights[0]
    assert first.date == "2026-08-01"
    assert first.last_night_avg_ms == 30.0
    assert first.last_night_5min_high_ms == 45
    assert first.weekly_avg_ms == 31
    assert first.status == "BALANCED"
    assert first.baseline_low_ms == 30  # balancedLow, not lowUpper
    assert first.baseline_high_ms == 42
    gap = next(n for n in result.nights if n.date == "2026-08-20")
    assert gap.status is None and gap.last_night_avg_ms is None


@pytest.mark.asyncio
async def test_hrv_history_serves_past_nights_from_disk(tmp_path: Path) -> None:
    store_path = tmp_path / "nights.json"
    server_module.set_hrv_store_for_testing(NightStore(store_path))
    fake = FakeGarminClient({"get_hrv_data_range": _range_responder()})
    server_module.set_garmin_client_for_testing(fake)
    await server_module.get_hrv_history("2026-01-01", "2026-01-31")
    assert len(fake.call_log) == 1
    assert store_path.exists()

    # A fresh process: new in-memory cache, new store object on the same file.
    await server_module.get_cache_for_testing().clear()
    server_module.set_hrv_store_for_testing(NightStore(store_path))
    fake2 = FakeGarminClient({})  # any Garmin call would fail the test
    server_module.set_garmin_client_for_testing(fake2)
    result = await server_module.get_hrv_history("2026-01-05", "2026-01-10")
    assert fake2.call_log == []
    assert result.nights[0].date == "2026-01-05"
    assert result.nights_with_data == 6


@pytest.mark.asyncio
async def test_hrv_history_recent_nights_are_not_cached_for_good() -> None:
    today = dt.date.today()
    fake = FakeGarminClient({"get_hrv_data_range": _range_responder()})
    server_module.set_garmin_client_for_testing(fake)
    start = (today - timedelta(days=4)).isoformat()
    await server_module.get_hrv_history(start, today.isoformat())
    await server_module.get_cache_for_testing().clear()
    await server_module.get_hrv_history(start, today.isoformat())
    # Second call only asks Garmin again for yesterday and today.
    assert fake.call_log[-1][1] == ((today - timedelta(days=1)).isoformat(), today.isoformat())


@pytest.mark.asyncio
async def test_hrv_history_falls_back_to_per_night_calls() -> None:
    def broken(*_a: Any) -> Any:
        raise GarminClientError("range endpoint down")

    class Fake(FakeGarminClient):
        async def call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
            if method_name == "get_hrv_data_range":
                self.call_log.append((method_name, args, kwargs))
                broken()
            return await super().call(method_name, *args, **kwargs)

    fake = Fake({"get_hrv_data": lambda d: {"hrvSummary": _summary(d, avg=25.0)}})
    server_module.set_garmin_client_for_testing(fake)
    result = await server_module.get_hrv_history("2026-03-01", "2026-03-03")
    assert [n.last_night_avg_ms for n in result.nights] == [25.0, 25.0, 25.0]


@pytest.mark.asyncio
async def test_hrv_history_rejects_bad_ranges() -> None:
    with pytest.raises(ValueError):
        await server_module.get_hrv_history("2026-03-02", "2026-03-01")
    with pytest.raises(ValueError):
        await server_module.get_hrv_history("2025-01-01", "2026-03-01")


@pytest.mark.asyncio
async def test_hrv_status_headline_skips_unsynced_last_night() -> None:
    """Before last night syncs, the newest day has no summary; the headline must still fill in."""
    today = dt.date.today().isoformat()
    fake = FakeGarminClient({"get_hrv_data_range": _range_responder({today})})
    server_module.set_garmin_client_for_testing(fake)
    result = await server_module.get_hrv_status()
    assert len(result.last_7_days) == 7
    assert result.last_7_days[-1].date == today
    assert result.last_7_days[-1].status is None
    assert result.date == (dt.date.today() - timedelta(days=1)).isoformat()
    assert result.status == "BALANCED"
    assert result.weekly_avg_ms == 31
    assert result.baseline_low_ms == 30
    assert result.baseline_high_ms == 42
    assert result.low_threshold_ms == 28
    assert result.feedback == "HRV_BALANCED_2"


@pytest.mark.asyncio
async def test_hrv_status_honours_date() -> None:
    fake = FakeGarminClient({"get_hrv_data_range": _range_responder()})
    server_module.set_garmin_client_for_testing(fake)
    result = await server_module.get_hrv_status(date="2026-10-10")
    assert result.last_7_days[0].date == "2026-10-04"
    assert result.date == "2026-10-10"


@pytest.mark.asyncio
async def test_sleep_reads_overnight_hrv_beside_daily_dto() -> None:
    payload = {
        "dailySleepDTO": {"calendarDate": "2026-10-10", "sleepTimeSeconds": 27000},
        "avgOvernightHrv": 43.0,
        "hrvStatus": "BALANCED",
    }
    server_module.set_garmin_client_for_testing(FakeGarminClient({"get_sleep_data": payload}))
    result = await server_module.get_sleep(date="2026-10-10")
    assert result.avg_overnight_hrv_ms == 43.0
    assert result.overnight_hrv_status == "BALANCED"


@pytest.mark.asyncio
async def test_rhr_goes_back_90_days_in_windows() -> None:
    def respond(start: str, end: str) -> list[dict[str, Any]]:
        s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
        return [
            {"calendarDate": (s + timedelta(days=i)).isoformat(), "value": 60.0}
            for i in range((e - s).days + 1)
        ]

    fake = FakeGarminClient({"get_rhr_daily": respond})
    server_module.set_garmin_client_for_testing(fake)
    result = await server_module.get_resting_heart_rate(days=120)
    assert len(result.days) == 120
    assert result.days[-1].date == dt.date.today().isoformat()
    assert all(d.rhr_bpm == 60 for d in result.days)
    assert len(fake.call_log) == 2
