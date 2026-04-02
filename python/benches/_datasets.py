"""
Utilities for loading real benchmark series from official public datasets.

The timed benchmark path should never hit the network directly. Download and
cache any source data first, then benchmark against the local snapshot.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_USER_AGENT = "interlib-real-bench/0.1 (opensource benchmark tooling)"
CACHE_DIR = Path(__file__).resolve().parent / "data_cache"


@dataclass(slots=True)
class RealSeries:
    name: str
    source: str
    source_url: str
    x: list[float]
    y: list[float]
    x_unit: str
    y_unit: str
    description: str


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _fetch_json(url: str, cache_name: str, *, refresh: bool, headers: dict[str, str] | None = None) -> dict[str, Any]:
    cache_path = _ensure_cache_dir() / cache_name

    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    request_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read().decode("utf-8")

    cache_path.write_text(payload, encoding="utf-8")
    return json.loads(payload)


def _iso_to_unix_seconds(timestamp: str) -> float:
    normalized = timestamp.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _dedupe_sorted_pairs(pairs: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    pairs.sort(key=lambda item: item[0])

    xs: list[float] = []
    ys: list[float] = []

    for x_val, y_val in pairs:
        if xs and math.isclose(x_val, xs[-1], rel_tol=0.0, abs_tol=1e-12):
            ys[-1] = y_val
        else:
            xs.append(x_val)
            ys.append(y_val)

    return xs, ys


def load_noaa_observations(
    *,
    station_id: str = "KSFO",
    field: str = "temperature",
    limit: int = 240,
    refresh: bool = False,
) -> RealSeries:
    """
    Load a 1D time series from NOAA/NWS station observations.

    Source documentation:
    https://www.weather.gov/documentation/services-web-api
    """

    field_map = {
        "temperature": ("temperature", "degC"),
        "dewpoint": ("dewpoint", "degC"),
        "wind_speed": ("windSpeed", "m/s"),
        "pressure": ("barometricPressure", "Pa"),
    }
    if field not in field_map:
        allowed = ", ".join(sorted(field_map))
        raise ValueError(f"Unknown NOAA field '{field}'. Choose one of: {allowed}")

    property_name, y_unit = field_map[field]
    url = f"https://api.weather.gov/stations/{station_id}/observations?limit={limit}"
    cache_name = f"noaa_{station_id.lower()}_{field}_{limit}.json"
    payload = _fetch_json(url, cache_name, refresh=refresh)

    features = payload.get("features", [])
    pairs: list[tuple[float, float]] = []

    for feature in features:
        properties = feature.get("properties", {})
        timestamp = properties.get("timestamp")
        measurement = properties.get(property_name, {})
        value = measurement.get("value") if isinstance(measurement, dict) else None

        if timestamp is None or value is None:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            continue

        unix_time = _iso_to_unix_seconds(timestamp)
        pairs.append((unix_time, float(value)))

    xs, ys = _dedupe_sorted_pairs(pairs)
    if len(xs) < 8:
        raise ValueError(
            f"NOAA dataset for station '{station_id}' and field '{field}' has only {len(xs)} usable points"
        )

    x0 = xs[0]
    elapsed_hours = [(value - x0) / 3600.0 for value in xs]

    return RealSeries(
        name=f"NOAA {station_id} {field}",
        source="NOAA / NWS API",
        source_url=url,
        x=elapsed_hours,
        y=ys,
        x_unit="hours since first observation",
        y_unit=y_unit,
        description=(
            f"Station observations for {station_id} using '{property_name}' from the "
            "National Weather Service API."
        ),
    )


def _extract_horizons_result_text(payload: dict[str, Any]) -> str:
    result = payload.get("result")
    if not isinstance(result, str):
        raise ValueError("NASA Horizons response did not contain a text result block")
    return result


def _extract_horizons_data_lines(result_text: str) -> list[str]:
    start_marker = "$$SOE"
    end_marker = "$$EOE"

    start_index = result_text.find(start_marker)
    end_index = result_text.find(end_marker)
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        raise ValueError("NASA Horizons response did not contain a $$SOE/$$EOE data block")

    data_block = result_text[start_index + len(start_marker):end_index]
    return [line.strip() for line in data_block.splitlines() if line.strip()]


def load_nasa_horizons_vectors(
    *,
    command: str = "499",
    axis: str = "x",
    center: str = "500@0",
    start_time: str = "2026-01-01",
    stop_time: str = "2026-02-01",
    step_size: str = "1 d",
    refresh: bool = False,
) -> RealSeries:
    """
    Load a trajectory component from NASA/JPL Horizons vector ephemerides.

    Documentation:
    https://ssd-api.jpl.nasa.gov/doc/horizons.html
    """

    axis_to_index = {"x": 2, "y": 3, "z": 4}
    if axis not in axis_to_index:
        raise ValueError("NASA axis must be one of: x, y, z")

    query = {
        "format": "json",
        "COMMAND": f"'{command}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": f"'{center}'",
        "START_TIME": f"'{start_time}'",
        "STOP_TIME": f"'{stop_time}'",
        "STEP_SIZE": f"'{step_size}'",
        "VEC_TABLE": "'1'",
        "CSV_FORMAT": "'YES'",
        "OUT_UNITS": "'AU-D'",
    }

    encoded = urllib.parse.urlencode(query, safe="'@")
    url = f"https://ssd.jpl.nasa.gov/api/horizons.api?{encoded}"
    cache_name = (
        f"nasa_horizons_{command.replace('/', '_')}_{axis}_{start_time}_{stop_time}_{step_size.replace(' ', '_')}.json"
    )
    payload = _fetch_json(url, cache_name, refresh=refresh)
    result_text = _extract_horizons_result_text(payload)
    lines = _extract_horizons_data_lines(result_text)

    pairs: list[tuple[float, float]] = []
    axis_index = axis_to_index[axis]

    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) <= axis_index:
            continue

        try:
            julian_day = float(parts[0])
            axis_value = float(parts[axis_index])
        except ValueError:
            continue

        if math.isfinite(julian_day) and math.isfinite(axis_value):
            pairs.append((julian_day, axis_value))

    xs, ys = _dedupe_sorted_pairs(pairs)
    if len(xs) < 8:
        raise ValueError(
            f"NASA Horizons dataset for command '{command}' axis '{axis}' has only {len(xs)} usable points"
        )

    x0 = xs[0]
    elapsed_days = [value - x0 for value in xs]

    return RealSeries(
        name=f"NASA Horizons {command} {axis}(t)",
        source="NASA / JPL Horizons API",
        source_url=url,
        x=elapsed_days,
        y=ys,
        x_unit="days since first ephemeris sample",
        y_unit="AU",
        description=(
            f"Cartesian {axis}-component from a Horizons VECTORS table for COMMAND={command} "
            f"relative to CENTER={center}."
        ),
    )
