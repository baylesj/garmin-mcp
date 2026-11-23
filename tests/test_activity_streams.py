"""get_activity_streams: Garmin's activity detail metrics as parallel columns."""

from __future__ import annotations

from typing import Any

import pytest

import garmin_mcp.server as server_module
from tests.test_tools import FakeGarminClient

SUMMARY = {
    "activityName": "Portland Cycling",
    "activityTypeDTO": {"typeKey": "cycling"},
    "summaryDTO": {
        "startTimeLocal": "2026-11-19T07:10:00.0",
        "duration": 3979.2,
        "movingDuration": 3700.0,
        "distance": 16716.7,
        "elevationGain": 164.5,
        "averagePower": 117.0,
        "normalizedPower": 151.0,
        "maxPower": 550.0,
        "trainingStressScore": 55.6,
        "intensityFactor": 0.711,
        "averageHR": 168.0,
    },
}

DETAILS = {
    "metricDescriptors": [
        {"metricsIndex": 0, "key": "directLatitude"},
        {"metricsIndex": 1, "key": "directLongitude"},
        {"metricsIndex": 2, "key": "directElevation"},
        {"metricsIndex": 3, "key": "sumElapsedDuration"},
        {"metricsIndex": 4, "key": "directPower"},
        {"metricsIndex": 5, "key": "directHeartRate"},
        {"metricsIndex": 6, "key": "sumDistance"},
        {"metricsIndex": 7, "key": "directBodyBattery"},
    ],
    "activityDetailMetrics": [
        {"metrics": [45.5, -122.6, 50.0, 0.0, 120.0, 130.0, 0.0, 80.0]},
        {"metrics": [45.51, -122.61, 52.4, 1.0, None, 131.0, 5.2, 80.0]},
        {"metrics": [45.52, -122.62, 55.0, 2.0, 210.0]},
    ],
}


@pytest.fixture(autouse=True)
async def _reset_state() -> Any:
    await server_module.get_cache_for_testing().clear()
    server_module.set_garmin_client_for_testing(None)
    yield
    server_module.set_garmin_client_for_testing(None)


@pytest.mark.asyncio
async def test_streams_map_columns_and_summary() -> None:
    fake = FakeGarminClient({"get_activity": SUMMARY, "get_activity_details": DETAILS})
    server_module.set_garmin_client_for_testing(fake)
    s = await server_module.get_activity_streams("25137321211", max_points=4000)

    assert ("get_activity_details", ("25137321211", 4000, 4000), {}) in fake.call_log
    assert s.points == 3
    assert s.latitude == [45.5, 45.51, 45.52]
    assert s.elevation_m == [50.0, 52.4, 55.0]
    assert s.power_w == [120.0, None, 210.0]
    assert s.heart_rate == [130.0, 131.0, None]  # a short row pads with None
    assert s.distance_m == [0.0, 5.2, None]
    assert s.cadence_rpm == []  # not recorded
    assert s.normalised_power == 151.0
    assert s.training_stress_score == 55.6
    assert s.intensity_factor == 0.711
    assert s.activity_type == "cycling"


@pytest.mark.asyncio
async def test_streams_clamp_points_and_handle_no_samples() -> None:
    fake = FakeGarminClient({"get_activity": SUMMARY, "get_activity_details": {}})
    server_module.set_garmin_client_for_testing(fake)
    s = await server_module.get_activity_streams("1", max_points=999999)

    assert ("get_activity_details", ("1", 20000, 20000), {}) in fake.call_log
    assert s.points == 0
    assert s.note and "no recorded samples" in s.note
    with pytest.raises(ValueError):
        await server_module.get_activity_streams(" ")
