"""Tests for get_ftp (current FTP, FTP history, and power zones).

Payload shapes are taken from live Garmin Connect responses for the
garminconnect methods the tool calls.
"""

from __future__ import annotations

from datetime import date, timedelta
from itertools import pairwise
from typing import Any

import pytest

from garmin_mcp import server as server_module
from garmin_mcp.garmin_client import _READ_METHODS, _WRITE_METHODS, GarminClientError
from garmin_mcp.models import FTPSummary
from tests.test_tools import FakeGarminClient

CYCLING_FTP_PAYLOAD: dict[str, Any] = {
    "userProfilePK": 75618171,
    "version": 1791963957268,
    "calendarDate": "2026-10-14T00:45:35.0",
    "isStale": False,
    "sequence": 1791963957268,
    "sport": "CYCLING",
    "functionalThresholdPower": 217,
    "biometricSourceType": "CHANGE_LOG",
}

FTP_RANGE_MONTHLY_PAYLOAD: list[dict[str, Any]] = [
    {
        "from": "2025-10-01",
        "until": "2025-10-31",
        "series": "cycling",
        "value": 172.0,
        "updatedDate": "2025-10-27",
    },
    {
        "from": "2026-10-01",
        "until": "2026-10-31",
        "series": "cycling",
        "value": 217.0,
        "updatedDate": "2026-10-18",
    },
]

CYCLING_ZONES_PAYLOAD: dict[str, Any] = {
    "sport": "CYCLING",
    "functionalThresholdPower": 217.0,
    "zone1Floor": 0.0,
    "zone2Floor": 117.0,
    "zone3Floor": 164.0,
    "zone4Floor": 197.0,
    "zone5Floor": 226.0,
    "zone6Floor": 263.0,
    "zone7Floor": 328.0,
    "userLocalTime": None,
}

# Running power zones only use five zones; Garmin reports the rest as 0.0.
RUNNING_ZONES_PAYLOAD: dict[str, Any] = {
    "sport": "RUNNING",
    "functionalThresholdPower": 290.0,
    "zone1Floor": 0.0,
    "zone2Floor": 232.0,
    "zone3Floor": 267.0,
    "zone4Floor": 290.0,
    "zone5Floor": 319.0,
    "zone6Floor": 0.0,
    "zone7Floor": 0.0,
    "userLocalTime": None,
}


@pytest.fixture(autouse=True)
async def _reset_state() -> Any:
    cache = server_module.get_cache_for_testing()
    await cache.clear()
    server_module.set_garmin_client_for_testing(None)
    yield
    server_module.set_garmin_client_for_testing(None)


def _range_calls(fake: FakeGarminClient) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    return [
        (args, kwargs)
        for name, args, kwargs in fake.call_log
        if name == "get_functional_threshold_power_range"
    ]


@pytest.mark.asyncio
async def test_get_ftp_cycling_full_payload() -> None:
    fake = FakeGarminClient(
        {
            "get_cycling_ftp": CYCLING_FTP_PAYLOAD,
            "get_functional_threshold_power_range": FTP_RANGE_MONTHLY_PAYLOAD,
            "get_power_zones_for_sport": CYCLING_ZONES_PAYLOAD,
        }
    )
    server_module.set_garmin_client_for_testing(fake)

    result = await server_module.get_ftp()

    assert isinstance(result, FTPSummary)
    assert result.sport == "CYCLING"
    assert result.sections_unavailable == []
    assert result.note is None

    assert result.current is not None
    assert result.current.ftp_watts == 217.0
    assert result.current.date_set == "2026-10-14"
    assert result.current.source == "CHANGE_LOG"
    assert result.current.is_stale is False

    assert result.history is not None
    assert result.history.aggregation == "monthly"  # default for a 365-day window
    assert [(e.period_start, e.ftp_watts) for e in result.history.entries] == [
        ("2025-10-01", 172.0),
        ("2026-10-01", 217.0),
    ]
    assert result.history.entries[-1].period_end == "2026-10-31"
    assert result.history.entries[-1].updated_date == "2026-10-18"

    # 365 days fits in one request.
    today = date.today()
    calls = _range_calls(fake)
    assert calls == [
        (
            ((today - timedelta(days=364)).isoformat(), today.isoformat()),
            {"sport": "CYCLING", "aggregation": "monthly"},
        )
    ]
    assert result.history.start_date == (today - timedelta(days=364)).isoformat()
    assert result.history.end_date == today.isoformat()


@pytest.mark.asyncio
async def test_get_ftp_zone_ceilings_and_percent_of_ftp() -> None:
    fake = FakeGarminClient(
        {
            "get_cycling_ftp": CYCLING_FTP_PAYLOAD,
            "get_functional_threshold_power_range": [],
            "get_power_zones_for_sport": CYCLING_ZONES_PAYLOAD,
        }
    )
    server_module.set_garmin_client_for_testing(fake)

    result = await server_module.get_ftp()

    assert ("get_power_zones_for_sport", ("CYCLING",), {}) in fake.call_log
    assert result.power_zones is not None
    assert result.power_zones.ftp_watts == 217.0
    zones = [
        (z.zone, z.floor_watts, z.ceiling_watts, z.floor_pct_of_ftp)
        for z in result.power_zones.zones
    ]
    assert zones == [
        (1, 0.0, 117.0, 0.0),
        (2, 117.0, 164.0, 53.9),
        (3, 164.0, 197.0, 75.6),
        (4, 197.0, 226.0, 90.8),
        (5, 226.0, 263.0, 104.1),
        (6, 263.0, 328.0, 121.2),
        (7, 328.0, None, 151.2),
    ]


@pytest.mark.asyncio
async def test_get_ftp_running_drops_unused_zones_and_skips_cycling_endpoint() -> None:
    fake = FakeGarminClient(
        {
            "get_functional_threshold_power_range": [],
            "get_power_zones_for_sport": RUNNING_ZONES_PAYLOAD,
        }
    )
    server_module.set_garmin_client_for_testing(fake)

    result = await server_module.get_ftp(sport="running", days=30)

    assert result.sport == "RUNNING"
    # The latest-FTP endpoint is cycling-only: not called, and not a failure.
    assert all(name != "get_cycling_ftp" for name, _, _ in fake.call_log)
    assert result.current is None
    assert result.sections_unavailable == []
    assert result.note is not None and "cycling-only" in result.note

    assert result.history is not None
    assert result.history.aggregation == "daily"  # default for windows <= 90 days
    assert _range_calls(fake)[0][1] == {"sport": "RUNNING", "aggregation": "daily"}

    assert result.power_zones is not None
    assert [z.zone for z in result.power_zones.zones] == [1, 2, 3, 4, 5]
    assert result.power_zones.zones[-1].floor_watts == 319.0
    assert result.power_zones.zones[-1].ceiling_watts is None


@pytest.mark.asyncio
async def test_get_ftp_history_is_chunked_under_garmins_366_day_limit() -> None:
    def ftp_range(start: str, end: str, *, sport: str, aggregation: str) -> list[dict[str, Any]]:
        span = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
        if span > 366:
            raise GarminClientError(
                "API Error 400 - End date is before start date or maximum date "
                "range of 366 days is exceeded."
            )
        # One reading on the last day of each window.
        return [
            {"from": end, "until": end, "series": "cycling", "value": 200.0, "updatedDate": end}
        ]

    fake = FakeGarminClient(
        {
            "get_cycling_ftp": CYCLING_FTP_PAYLOAD,
            "get_functional_threshold_power_range": ftp_range,
            "get_power_zones_for_sport": CYCLING_ZONES_PAYLOAD,
        }
    )
    server_module.set_garmin_client_for_testing(fake)

    result = await server_module.get_ftp(days=1095, aggregation="daily")

    assert result.sections_unavailable == []
    today = date.today()
    windows = [
        (date.fromisoformat(args[0]), date.fromisoformat(args[1])) for args, _ in _range_calls(fake)
    ]
    assert len(windows) == 3
    assert windows[0][0] == today - timedelta(days=1094)
    assert windows[-1][1] == today
    for (_, prev_end), (next_start, _) in pairwise(windows):
        assert next_start == prev_end + timedelta(days=1)  # contiguous, no overlap
    assert all((end - start).days + 1 <= 366 for start, end in windows)

    assert result.history is not None
    assert [e.period_start for e in result.history.entries] == [
        end.isoformat() for _, end in windows
    ]


@pytest.mark.asyncio
async def test_get_ftp_days_is_capped() -> None:
    fake = FakeGarminClient(
        {
            "get_cycling_ftp": CYCLING_FTP_PAYLOAD,
            "get_functional_threshold_power_range": [],
            "get_power_zones_for_sport": CYCLING_ZONES_PAYLOAD,
        }
    )
    server_module.set_garmin_client_for_testing(fake)

    result = await server_module.get_ftp(days=5000)

    assert result.history is not None
    assert result.history.start_date == (date.today() - timedelta(days=1094)).isoformat()
    assert len(_range_calls(fake)) == 3


def test_parse_ftp_history_merges_periods_split_across_windows() -> None:
    # A monthly period cut by a window boundary comes back from both windows;
    # the more recently updated reading wins and the output is sorted.
    first = [
        {"from": "2025-09-01", "until": "2025-09-30", "value": 165.0, "updatedDate": "2025-09-12"},
        {"from": "2025-10-01", "until": "2025-10-31", "value": 172.0, "updatedDate": "2025-10-09"},
    ]
    second = [
        {"from": "2025-10-01", "until": "2025-10-31", "value": 180.0, "updatedDate": "2025-10-27"},
        {"from": "2026-10-01", "until": "2026-10-31", "value": 217.0, "updatedDate": "2026-10-18"},
    ]

    history = server_module._parse_ftp_history(
        [second, first], "2024-10-26", "2026-10-25", "monthly"
    )

    assert [(e.period_start, e.ftp_watts) for e in history.entries] == [
        ("2025-09-01", 165.0),
        ("2025-10-01", 180.0),
        ("2026-10-01", 217.0),
    ]


def test_parse_ftp_history_ignores_rows_without_value() -> None:
    history = server_module._parse_ftp_history(
        [[{"from": "2026-10-14", "value": None}, "junk"], None],
        "2026-10-01",
        "2026-10-25",
        "daily",
    )
    assert history.entries == []


def test_date_windows_split() -> None:
    windows = server_module._date_windows(date(2024, 1, 1), date(2025, 1, 5), 366)
    assert windows == [("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-01-05")]
    assert server_module._date_windows(date(2026, 10, 25), date(2026, 10, 25), 366) == [
        ("2026-10-25", "2026-10-25")
    ]


@pytest.mark.asyncio
async def test_get_ftp_failed_section_degrades_to_null() -> None:
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise GarminClientError("Garmin call failed")

    fake = FakeGarminClient(
        {
            "get_cycling_ftp": CYCLING_FTP_PAYLOAD,
            "get_functional_threshold_power_range": fail,
            "get_power_zones_for_sport": CYCLING_ZONES_PAYLOAD,
        }
    )
    server_module.set_garmin_client_for_testing(fake)

    result = await server_module.get_ftp()

    assert result.history is None
    assert result.sections_unavailable == ["history"]
    assert result.current is not None and result.current.ftp_watts == 217.0
    assert result.power_zones is not None and len(result.power_zones.zones) == 7


@pytest.mark.asyncio
async def test_get_ftp_empty_responses() -> None:
    fake = FakeGarminClient(
        {
            "get_cycling_ftp": None,
            "get_functional_threshold_power_range": [],
            "get_power_zones_for_sport": {},
        }
    )
    server_module.set_garmin_client_for_testing(fake)

    result = await server_module.get_ftp()

    assert result.sections_unavailable == []
    assert result.current is not None and result.current.ftp_watts is None
    assert result.history is not None and result.history.entries == []
    assert result.power_zones is not None and result.power_zones.zones == []


@pytest.mark.asyncio
async def test_get_ftp_rejects_bad_arguments() -> None:
    server_module.set_garmin_client_for_testing(FakeGarminClient({}))
    with pytest.raises(ValueError, match="sport"):
        await server_module.get_ftp(sport="cycling/../x")
    with pytest.raises(ValueError, match="aggregation"):
        await server_module.get_ftp(aggregation="hourly")


def test_ftp_methods_are_read_only() -> None:
    for name in (
        "get_cycling_ftp",
        "get_functional_threshold_power_range",
        "get_power_zones_for_sport",
    ):
        assert name in _READ_METHODS
        assert name not in _WRITE_METHODS


@pytest.mark.asyncio
async def test_fitness_metrics_includes_current_ftp() -> None:
    server_module.set_garmin_client_for_testing(
        FakeGarminClient(
            {
                "get_max_metrics": None,
                "get_race_predictions": None,
                "get_cycling_ftp": CYCLING_FTP_PAYLOAD,
            }
        )
    )
    result = await server_module.get_fitness_metrics(date="2026-10-24")
    assert result.cycling_ftp is not None
    assert result.cycling_ftp.ftp_watts == 217
    assert result.cycling_ftp.date_set == "2026-10-14"
    # An FTP on file counts as a fitness metric, so there is no "nothing recorded" note.
    assert result.note is None


@pytest.mark.asyncio
async def test_fitness_metrics_without_ftp() -> None:
    def fail() -> Any:
        raise GarminClientError("Garmin call failed")

    server_module.set_garmin_client_for_testing(
        FakeGarminClient(
            {"get_max_metrics": None, "get_race_predictions": None, "get_cycling_ftp": fail}
        )
    )
    result = await server_module.get_fitness_metrics(date="2026-10-24")
    assert result.cycling_ftp is None
    assert result.note is not None
