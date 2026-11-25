"""Pedal dynamics from an activity's original FIT file.

Garmin Connect's JSON (summary, details, streams) drops most of what dual-sided power pedals send. The
original FIT the head unit recorded keeps it: left/right balance, torque effectiveness and pedal
smoothness per leg (Favero Assioma, Garmin Rally, SRM X, ...), platform centre offset and power phase
(Garmin Rally only), plus device_info rows that say which sensors were paired: power meter, chest strap,
battery state.

Wire it into the server with:

    from garmin_mcp.pedal_dynamics import register_pedal_dynamics
    register_pedal_dynamics(mcp, _get_garmin)   # this server's GarminClient wrapper

Needs `fitdecode` (pure Python).
"""

from __future__ import annotations

import io
import itertools
import statistics as st
import zipfile
from collections.abc import Callable, Iterable
from typing import Any

import fitdecode

# Power bands (W) for the balance-by-effort table: balance often shifts with load.
DEFAULT_BANDS = (0, 80, 120, 180, 250, 10_000)

_RECORD_FIELDS = (
    "power",
    "cadence",
    "heart_rate",
    "left_right_balance",
    "left_torque_effectiveness",
    "right_torque_effectiveness",
    "left_pedal_smoothness",
    "right_pedal_smoothness",
    "combined_pedal_smoothness",
    "left_pco",
    "right_pco",
)
_SESSION_FIELDS = (
    "left_right_balance",
    "avg_left_torque_effectiveness",
    "avg_right_torque_effectiveness",
    "avg_left_pedal_smoothness",
    "avg_right_pedal_smoothness",
    "avg_combined_pedal_smoothness",
    "avg_left_pco",
    "avg_right_pco",
    "avg_left_power_phase",
    "avg_right_power_phase",
    "avg_power",
    "normalized_power",
    "avg_cadence",
    "avg_heart_rate",
)


# ---------- decoding ----------
def right_pct_from_record(raw: Any) -> float | None:
    """FIT record.left_right_balance (uint8): bit 7 set = the low 7 bits are the RIGHT leg's %.
    Without bit 7 the side is unknown (single-sided meters), so it's not a balance at all."""
    if raw is None:
        return None
    if isinstance(raw, str):  # some decoders render it as 'right'/'left' text; not usable
        return None
    raw = int(raw)
    if raw == 0xFF or not raw & 0x80:
        return None
    pct = raw & 0x7F
    return float(pct) if pct <= 100 else None


def right_pct_from_session(raw: Any) -> float | None:
    """FIT session.left_right_balance is left_right_balance_100 (uint16): bit 15 = right, low 14 bits
    = percent * 100."""
    if raw is None or isinstance(raw, str):
        return None
    raw = int(raw)
    if raw == 0xFFFF or not raw & 0x8000:
        return None
    pct = (raw & 0x3FFF) / 100
    return pct if pct <= 100 else None


def extract_fit(blob: bytes) -> bytes:
    """Garmin's ORIGINAL download is a zip holding one .fit; accept a bare FIT too."""
    if blob[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".fit"))
            return z.read(name)
    return blob


def _val(frame, name):
    return frame.get_value(name, fallback=None) if frame.has_field(name) else None


def read_fit(fit: bytes) -> dict:
    records, sessions, devices = [], [], []
    with fitdecode.FitReader(io.BytesIO(fit), check_crc=fitdecode.CrcCheck.WARN) as reader:
        for frame in reader:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name == "record":
                records.append({f: _val(frame, f) for f in _RECORD_FIELDS})
            elif frame.name == "session":
                sessions.append({f: _val(frame, f) for f in _SESSION_FIELDS})
            elif frame.name == "device_info":
                devices.append(
                    {
                        f: _val(frame, f)
                        for f in (
                            "device_index",
                            "device_type",
                            "antplus_device_type",
                            "ble_device_type",
                            "manufacturer",
                            "product",
                            "garmin_product",
                            "serial_number",
                            "source_type",
                            "battery_status",
                            "battery_voltage",
                            "software_version",
                        )
                    }
                )
    return {"records": records, "sessions": sessions, "devices": devices}


# ---------- summarising ----------
def _mean(xs: Iterable[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 1) if xs else None


def _present(value: Any) -> Any:
    """None for a missing field, including fitdecode's array of all-invalid values (e.g. power phase
    [None, None, None, None] from pedals that don't measure it)."""
    if isinstance(value, (list, tuple)) and all(v is None for v in value):
        return None
    return value


def _sensor_list(devices: list[dict]) -> dict:
    """Collapse device_info rows (one per sensor per timestamp) to one entry per sensor."""
    seen: dict[tuple, dict] = {}
    for d in devices:
        kind = d.get("antplus_device_type") or d.get("ble_device_type") or d.get("device_type")
        if kind in (None, 0, "unknown"):
            continue
        key = (str(kind), d.get("serial_number"), d.get("device_index"))
        e = seen.setdefault(
            key,
            {
                "type": str(kind),
                "manufacturer": d.get("manufacturer"),
                "serial": d.get("serial_number"),
                "source": d.get("source_type"),
            },
        )
        if d.get("battery_status") is not None:
            e["battery_status"] = d["battery_status"]  # last report wins
        if d.get("battery_voltage") is not None:
            e["battery_v"] = round(float(d["battery_voltage"]), 2)
        if d.get("software_version") is not None:
            e["firmware"] = d["software_version"]
    # The head unit also writes rows without a serial number for a sensor it lists with one elsewhere;
    # fold those into the serial-numbered entry instead of showing the sensor twice.
    sensors = list(seen.values())
    with_serial = {s["type"]: s for s in sensors if s.get("serial") is not None}
    merged = []
    for s in sensors:
        twin = with_serial.get(s["type"]) if s.get("serial") is None else None
        if twin is not None and twin is not s:
            for k, v in s.items():
                if v is not None and twin.get(k) is None:
                    twin[k] = v
            continue
        merged.append(s)
    return {
        "sensors": merged,
        "hr_source": (
            "chest strap"
            if any(s["type"] == "heart_rate" and s.get("source") != "local" for s in merged)
            else "wrist (no strap paired)"
        ),
    }


def summarise(parsed: dict, bands: tuple[int, ...] = DEFAULT_BANDS) -> dict:
    recs = parsed["records"]
    pedalling = [r for r in recs if (r["power"] or 0) > 20]

    # Balance, power-weighted: the average share of the WORK, not of the seconds.
    bal = [(r["power"], right_pct_from_record(r["left_right_balance"])) for r in pedalling]
    bal = [(p, b) for p, b in bal if b is not None]
    right_w = round(sum(p * b for p, b in bal) / sum(p for p, _ in bal), 1) if bal else None
    by_band = []
    for lo, hi in itertools.pairwise(bands):
        xs = [b for p, b in bal if lo <= p < hi]
        if xs:
            by_band.append(
                {
                    "watts": f"{lo}–{hi if hi < 10_000 else 'max'}",  # noqa: RUF001 (the label's dash is meant)
                    "seconds": len(xs),
                    "right_pct_median": round(st.median(xs), 1),
                }
            )

    def leg(name):  # 0..100 %, Garmin/Favero send 0xFF (invalid) when coasting; fitdecode → None
        return _mean(r[name] for r in pedalling if r[name] is not None and r[name] <= 100)

    sess = parsed["sessions"][-1] if parsed["sessions"] else {}
    out = {
        "records": len(recs),
        "pedalling_s": len(pedalling),
        "balance": {
            "right_pct_power_weighted": right_w,
            "left_pct_power_weighted": round(100 - right_w, 1) if right_w is not None else None,
            "right_pct_session": right_pct_from_session(sess.get("left_right_balance")),
            "by_power_band": by_band,
            "samples": len(bal),
        },
        "torque_effectiveness_pct": {
            "left": leg("left_torque_effectiveness"),
            "right": leg("right_torque_effectiveness"),
            "session_left": sess.get("avg_left_torque_effectiveness"),
            "session_right": sess.get("avg_right_torque_effectiveness"),
        },
        "pedal_smoothness_pct": {
            "left": leg("left_pedal_smoothness"),
            "right": leg("right_pedal_smoothness"),
            "combined": leg("combined_pedal_smoothness"),
            "session_left": sess.get("avg_left_pedal_smoothness"),
            "session_right": sess.get("avg_right_pedal_smoothness"),
        },
        "cadence_rpm_pedalling": _mean(r["cadence"] for r in pedalling if r["cadence"]),
        **_sensor_list(parsed["devices"]),
    }
    # Cycling Dynamics (Garmin Rally / Vector only): only report when present.
    pco_l = _mean(r["left_pco"] for r in pedalling if r["left_pco"] is not None)
    pco_r = _mean(r["right_pco"] for r in pedalling if r["right_pco"] is not None)
    phase_l = _present(sess.get("avg_left_power_phase"))
    phase_r = _present(sess.get("avg_right_power_phase"))
    if pco_l is not None or pco_r is not None or phase_l is not None or phase_r is not None:
        out["cycling_dynamics"] = {
            "pco_mm_left": pco_l,
            "pco_mm_right": pco_r,
            "power_phase_left": phase_l,
            "power_phase_right": phase_r,
        }
    notes = []
    if right_w is not None and abs(right_w - 50) >= 8:
        notes.append(
            "Balance is 8+ points off 50/50: check crank length and zero offset on both pedals "
            "before reading it as a real asymmetry."
        )
    if out["torque_effectiveness_pct"]["left"] is None and right_w is not None:
        notes.append(
            "Balance is recorded but torque effectiveness is not: the head unit may not be "
            "logging pedal metrics for this sensor."
        )
    out["notes"] = notes
    return out


# ---------- MCP wiring ----------
def _original_format():
    from garminconnect import Garmin

    return Garmin.ActivityDownloadFormat.ORIGINAL


def get_pedal_dynamics(client, activity_id: str, bands: tuple[int, ...] = DEFAULT_BANDS) -> dict:
    """Synchronous version for a logged-in ``garminconnect.Garmin`` (scripts, checks)."""
    blob = client.download_activity(activity_id, dl_fmt=_original_format())
    out = summarise(read_fit(extract_fit(blob)), bands)
    out["activity_id"] = str(activity_id)
    return out


async def fetch_pedal_dynamics(
    client, activity_id: str, bands: tuple[int, ...] = DEFAULT_BANDS
) -> dict:
    """For this server's ``GarminClient`` wrapper (async, allowlisted, rate-limited, retried)."""
    if not activity_id or not str(activity_id).strip():
        raise ValueError("activity_id is required.")
    blob = await client.call("download_activity", str(activity_id), dl_fmt=_original_format())
    out = summarise(read_fit(extract_fit(blob)), bands)
    out["activity_id"] = str(activity_id)
    return out


def register_pedal_dynamics(mcp, get_client: Callable[[], Any]) -> None:
    """``get_client()`` returns this server's ``GarminClient`` (the one with an async ``call``)."""

    @mcp.tool(name="get_pedal_dynamics")
    async def get_pedal_dynamics_tool(activity_id: str) -> dict:
        """Pedal metrics from a ride's original FIT file, which Garmin's JSON leaves out: left/right
        balance (power-weighted, from the session, and by power band), torque effectiveness and pedal
        smoothness per leg, cadence while pedalling, platform centre offset and power phase when the
        pedals record them (Garmin Rally only), and the sensors paired for the ride (power meter, chest
        strap or wrist heart rate, battery status, firmware).

        Args:
            activity_id: The ride's numeric Garmin ID, as returned by ``get_recent_activities``.
        """
        return await fetch_pedal_dynamics(get_client(), activity_id)
