"""Builds small FIT files with fit_tool and checks what pedal_dynamics reads back."""

import datetime as dt
import io
import zipfile

import pytest
from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.profile.messages.device_info_message import DeviceInfoMessage
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.messages.session_message import SessionMessage
from fit_tool.profile.profile_type import AntplusDeviceType, FileType, Manufacturer, SourceType

from garmin_mcp import pedal_dynamics as pd

T0 = int(dt.datetime(2026, 11, 24, 18, 34, tzinfo=dt.UTC).timestamp() * 1000)


def build(rows, session_balance=None, strap=True):
    b = FitFileBuilder(auto_define=True, min_string_size=50)
    f = FileIdMessage()
    f.type = FileType.ACTIVITY
    f.manufacturer = Manufacturer.GARMIN.value
    f.product = 0
    f.time_created = T0
    f.serial_number = 1
    b.add(f)
    pm = DeviceInfoMessage()
    pm.timestamp = T0
    pm.device_index = 1
    pm.antplus_device_type = AntplusDeviceType.BIKE_POWER.value
    pm.serial_number = 4242
    pm.source_type = SourceType.ANTPLUS.value
    pm.battery_voltage = 3.9
    b.add(pm)
    if strap:
        hr = DeviceInfoMessage()
        hr.timestamp = T0
        hr.device_index = 2
        hr.antplus_device_type = AntplusDeviceType.HEART_RATE.value
        hr.serial_number = 777
        hr.source_type = SourceType.ANTPLUS.value
        b.add(hr)
    for i, (w, rb, te, ps) in enumerate(rows):
        r = RecordMessage()
        r.timestamp = T0 + i * 1000
        r.power = w
        r.cadence = 85
        r.heart_rate = 135
        if rb is not None:
            r.left_right_balance = rb
        if te is not None:
            r.left_torque_effectiveness, r.right_torque_effectiveness = te
        if ps is not None:
            r.left_pedal_smoothness, r.right_pedal_smoothness = ps
        b.add(r)
    s = SessionMessage()
    s.timestamp = T0 + len(rows) * 1000
    s.start_time = T0
    if session_balance is not None:
        s.left_right_balance = session_balance
    s.avg_left_torque_effectiveness = 70.0
    s.avg_right_torque_effectiveness = 72.0
    b.add(s)
    return b.build().to_bytes()


def test_decoders():
    assert pd.right_pct_from_record(0x80 | 54) == 54
    assert pd.right_pct_from_record(54) is None  # side unknown (single-sided)
    assert pd.right_pct_from_record(0xFF) is None
    assert pd.right_pct_from_session(0x8000 | 6730) == 67.3
    assert pd.right_pct_from_session(6730) is None


def test_summary_from_fit_and_zip():
    rows = (
        [(100, 0x80 | 55, (70, 74), (20, 22))] * 60
        + [(200, 0x80 | 65, (68, 76), (18, 24))] * 30
        + [(0, None, None, None)] * 10
    )
    fit = build(rows, session_balance=0x8000 | 5833)
    z = io.BytesIO()
    zipfile.ZipFile(z, "w").writestr("123_ACTIVITY.fit", fit)
    for blob in (fit, z.getvalue()):
        out = pd.summarise(pd.read_fit(pd.extract_fit(blob)))
        assert out["pedalling_s"] == 90
        # power-weighted: (60*100*55 + 30*200*65) / (60*100 + 30*200) = 60
        assert out["balance"]["right_pct_power_weighted"] == 60.0
        assert out["balance"]["left_pct_power_weighted"] == 40.0
        assert out["balance"]["right_pct_session"] == 58.33
        bands = {b["watts"]: b["right_pct_median"] for b in out["balance"]["by_power_band"]}
        assert bands == {"80–120": 55.0, "180–250": 65.0}  # noqa: RUF001
        assert out["torque_effectiveness_pct"]["left"] == pytest.approx(69.3, abs=0.1)
        assert out["torque_effectiveness_pct"]["session_right"] == 72.0
        assert out["pedal_smoothness_pct"]["right"] == pytest.approx(22.7, abs=0.1)
        assert out["hr_source"] == "chest strap"
        assert {s["type"] for s in out["sensors"]} >= {"bike_power", "heart_rate"}
        assert any("8+ points" in n for n in out["notes"])
        assert "cycling_dynamics" not in out


def test_no_strap_no_balance():
    out = pd.summarise(pd.read_fit(build([(150, None, None, None)] * 20, strap=False)))
    assert out["hr_source"] == "wrist (no strap paired)"
    assert out["balance"]["right_pct_power_weighted"] is None
    assert out["notes"] == []


def test_tool_registers_under_its_name():
    try:
        from mcp.server.mcpserver import MCPServer as Server  # mcp 2.x
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server  # mcp 1.x
    m = Server("t")
    pd.register_pedal_dynamics(m, lambda: None)
    import asyncio

    names = [t.name for t in asyncio.run(m.list_tools())]
    assert "get_pedal_dynamics" in names


def test_power_phase_of_all_invalid_values_is_absent():
    parsed = pd.read_fit(build([(150, 0x80 | 50, None, None)] * 20))
    parsed["sessions"][-1]["avg_left_power_phase"] = [None, None, None, None]
    parsed["sessions"][-1]["avg_right_power_phase"] = (None, None, None, None)
    assert "cycling_dynamics" not in pd.summarise(parsed)
