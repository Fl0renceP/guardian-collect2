"""Configuration loading for the behavioural analysis module.

Everything tunable lives in `config.yaml`. This module turns that file into
typed objects and — importantly — fails loudly on a mistyped key.

That last point is not pedantry. A plain dict returns `None` for a typo, and a
threshold that silently becomes `None` either crashes deep inside a heuristic or,
worse, compares as falsy and quietly disables a safeguard. `_Section` raises
instead, so a bad config is caught at load time rather than in front of judges.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "config.yaml"

# Zone types the heuristics understand. A zone with any other type loads fine
# but no heuristic will look at it, so we warn rather than accept silently.
KNOWN_ZONE_TYPES = {
    "property_boundary",
    "gate",
    "vertical_structure",
    "vehicle_zone",
    "street_frontage",
    "exempt",
}

# Zone types that count as "the property" for approach/loitering purposes.
PROPERTY_ZONE_TYPES = {"property_boundary", "gate", "street_frontage"}


class ConfigError(ValueError):
    """Raised when config.yaml is missing a key or holds a nonsense value."""


class _Section:
    """Dict wrapper with attribute access that raises on unknown keys."""

    def __init__(self, name: str, data: Dict[str, Any]):
        self._name = name
        self._data = data or {}

    def __getattr__(self, key: str) -> Any:
        # Only called when normal attribute lookup fails, so _name/_data are safe.
        try:
            value = self._data[key]
        except KeyError as exc:
            raise ConfigError(
                f"config.yaml: '{self._name}' has no setting '{key}'. "
                f"Available: {sorted(self._data)}"
            ) from exc
        if isinstance(value, dict):
            return _Section(f"{self._name}.{key}", value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Explicit optional lookup, for genuinely optional settings."""
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:
        return f"<config {self._name}: {sorted(self._data)}>"


@dataclass(frozen=True)
class Zone:
    """A camera-relative region of interest.

    Coordinates are normalised to the frame (0..1 on both axes), so one config
    survives a resolution change and can be authored off a screenshot.
    """

    id: str
    type: str
    risk: float
    polygon: Tuple[Tuple[float, float], ...]

    @property
    def is_exempt(self) -> bool:
        """Somewhere it is normal to stand still — loitering never fires here."""
        return self.type == "exempt"

    @property
    def is_property(self) -> bool:
        return self.type in PROPERTY_ZONE_TYPES


@dataclass
class Settings:
    """Loaded configuration. Sections mirror config.yaml's top-level keys."""

    pipeline: _Section
    heuristics: _Section
    fusion: _Section
    output: _Section
    audit: _Section
    zones: List[Zone] = field(default_factory=list)
    source_path: Path = DEFAULT_CONFIG_PATH

    def zones_of_type(self, *types: str) -> List[Zone]:
        wanted = set(types)
        return [z for z in self.zones if z.type in wanted]

    @property
    def exempt_zones(self) -> List[Zone]:
        return [z for z in self.zones if z.is_exempt]

    @property
    def property_zones(self) -> List[Zone]:
        return [z for z in self.zones if z.is_property]

    def zone_by_id(self, zone_id: str) -> Zone | None:
        return next((z for z in self.zones if z.id == zone_id), None)

    def heuristic_weight(self, name: str) -> float:
        """Fusion weight for a heuristic, by its registered name."""
        weights = self.fusion.weights
        return float(weights.get(name, 0.0))


def _parse_polygon(raw: Sequence[Sequence[float]], zone_id: str) -> Tuple[Tuple[float, float], ...]:
    if not raw or len(raw) < 3:
        raise ConfigError(f"Zone '{zone_id}' needs at least 3 polygon points, got {len(raw or [])}.")

    points = []
    for point in raw:
        if len(point) != 2:
            raise ConfigError(f"Zone '{zone_id}' has a polygon point that is not [x, y]: {point!r}")
        x, y = float(point[0]), float(point[1])
        # Normalised coordinates only. Catching this here beats debugging why a
        # zone drawn in pixel coordinates never matches anything.
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ConfigError(
                f"Zone '{zone_id}' point {point!r} is outside 0..1. Zone polygons are "
                f"normalised to the frame, not pixel coordinates — divide by frame "
                f"width/height."
            )
        points.append((x, y))
    return tuple(points)


def _parse_zones(raw_zones: Sequence[Dict[str, Any]]) -> List[Zone]:
    zones: List[Zone] = []
    seen = set()
    for raw in raw_zones or []:
        zone_id = str(raw.get("id") or "").strip()
        if not zone_id:
            raise ConfigError("Every zone needs an 'id'.")
        if zone_id in seen:
            raise ConfigError(f"Duplicate zone id '{zone_id}'.")
        seen.add(zone_id)

        zone_type = str(raw.get("type") or "").strip()
        if zone_type not in KNOWN_ZONE_TYPES:
            raise ConfigError(
                f"Zone '{zone_id}' has type '{zone_type}', which no heuristic reads. "
                f"Known types: {sorted(KNOWN_ZONE_TYPES)}"
            )

        risk = float(raw.get("risk", 0.0))
        if not 0.0 <= risk <= 1.0:
            raise ConfigError(f"Zone '{zone_id}' risk must be 0..1, got {risk}.")

        zones.append(
            Zone(
                id=zone_id,
                type=zone_type,
                risk=risk,
                polygon=_parse_polygon(raw.get("polygon"), zone_id),
            )
        )
    return zones


def _validate_fusion(fusion: _Section) -> None:
    """Catch fusion weights that would let a composite score exceed 1.0.

    The composite is clamped anyway, but a set of weights summing past 1.0 means
    the formula no longer says what the README says it says, and the demo
    explanation stops being true.
    """
    total = (
        float(fusion.behaviour_weight)
        + float(fusion.face_weight)
        + float(fusion.agreement_weight)
    )
    if abs(total - 1.0) > 0.01:
        raise ConfigError(
            f"fusion.behaviour_weight + face_weight + agreement_weight must sum to 1.0, "
            f"got {total:.3f}. These three weights are what the composite score means; "
            f"if they do not sum to 1 the README's explanation is wrong."
        )

    for key in ("review_threshold", "behaviour_only_review_threshold", "face_trust_threshold"):
        value = float(getattr(fusion, key))
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"fusion.{key} must be 0..1, got {value}.")


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    """Load and validate config.yaml.

    `path` may also be supplied via the BEHAVIOUR_CONFIG environment variable,
    which is how a second camera's zone set gets swapped in without a code change.
    """
    config_path = Path(path or os.environ.get("BEHAVIOUR_CONFIG") or DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    missing = [k for k in ("pipeline", "heuristics", "fusion", "output", "audit") if k not in raw]
    if missing:
        raise ConfigError(f"config.yaml is missing required section(s): {missing}")

    settings = Settings(
        pipeline=_Section("pipeline", raw["pipeline"]),
        heuristics=_Section("heuristics", raw["heuristics"]),
        fusion=_Section("fusion", raw["fusion"]),
        output=_Section("output", raw["output"]),
        audit=_Section("audit", raw["audit"]),
        zones=_parse_zones(raw.get("zones", [])),
        source_path=config_path,
    )

    _validate_fusion(settings.fusion)
    return settings
