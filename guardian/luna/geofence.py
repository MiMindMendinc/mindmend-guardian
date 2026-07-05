"""Geofence helpers for Luna location checks."""

from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return radius_km * c


def is_out_of_bounds(
    lat: float,
    lon: float,
    *,
    safe_lat: float,
    safe_lon: float,
    radius_km: float,
) -> bool:
    try:
        lat_value = float(lat)
        lon_value = float(lon)
        distance = haversine(lat_value, lon_value, safe_lat, safe_lon)
        return distance > radius_km
    except (TypeError, ValueError):
        return False
