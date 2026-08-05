"""Zone geometry — polygons drawn on the camera's view.

Zones are authored in NORMALISED frame coordinates (0..1 on both axes) so one
config survives a resolution change and can be drawn off a screenshot. All
*distance* maths, however, happens in PIXELS and is then divided by the person's
own body height.

That two-step matters. Normalised coordinates are not a metric space when the
frame is not square: 0.1 horizontally on a 848x480 frame is 85px, while 0.1
vertically is 48px. Measuring "distance to the gate" in normalised units would
silently stretch every threshold by the aspect ratio. So: author normalised,
measure in pixels, express in body heights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from settings import Zone

Point = Tuple[float, float]


@dataclass
class PixelZone:
    """A zone projected onto a specific frame size."""

    zone: Zone
    polygon: Tuple[Point, ...]

    @property
    def id(self) -> str:
        return self.zone.id

    @property
    def type(self) -> str:
        return self.zone.type

    @property
    def risk(self) -> float:
        return self.zone.risk

    @property
    def is_exempt(self) -> bool:
        return self.zone.is_exempt

    @property
    def centroid(self) -> Point:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)

    def distance(self, point: Point) -> float:
        """Pixels from `point` to this zone. Zero if inside."""
        return distance_to_polygon(point, self.polygon)


class ZoneIndex:
    """Zones projected onto one frame size, with lookups by type.

    Built once per frame size rather than per frame — the projection is pure
    arithmetic but there is no reason to redo it 900 times.
    """

    def __init__(self, zones: Sequence[Zone], frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.zones: List[PixelZone] = [
            PixelZone(
                zone=zone,
                polygon=tuple(
                    (x * frame_width, y * frame_height) for x, y in zone.polygon
                ),
            )
            for zone in zones
        ]
        self._by_type: Dict[str, List[PixelZone]] = {}
        for pixel_zone in self.zones:
            self._by_type.setdefault(pixel_zone.type, []).append(pixel_zone)

    def matches(self, frame_width: int, frame_height: int) -> bool:
        return self.frame_width == frame_width and self.frame_height == frame_height

    def of_type(self, *types: str) -> List[PixelZone]:
        out: List[PixelZone] = []
        for zone_type in types:
            out.extend(self._by_type.get(zone_type, []))
        return out

    def containing(self, point: Point) -> List[PixelZone]:
        return [z for z in self.zones if z.contains(point)]

    def exempt_containing(self, point: Point) -> Optional[PixelZone]:
        """The exempt zone this point sits in, if any.

        Its existence is the loitering bias guard: somewhere it is normal to
        stand still — a taxi rank, a bus stop, a bench, a shop queue.
        """
        return next((z for z in self.zones if z.is_exempt and z.contains(point)), None)

    def nearest(self, point: Point, *types: str) -> Tuple[Optional[PixelZone], float]:
        """Closest zone of the given types, and the pixel distance to it."""
        candidates = self.of_type(*types) if types else self.zones
        best: Optional[PixelZone] = None
        best_distance = math.inf
        for zone in candidates:
            distance = zone.distance(point)
            if distance < best_distance:
                best, best_distance = zone, distance
        return best, best_distance

    def risk_at(self, point: Point) -> float:
        """Highest claims-derived risk among the zones containing this point.

        Exempt zones do not contribute risk — the whole point of marking a bus
        stop is that being there means nothing.
        """
        risks = [z.risk for z in self.containing(point) if not z.is_exempt]
        return max(risks) if risks else 0.0


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Standard ray-casting test."""
    x, y = point
    inside = False
    count = len(polygon)
    j = count - 1
    for i in range(count):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def distance_point_to_segment(point: Point, a: Point, b: Point) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return math.hypot(px - ax, py - ay)

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_polygon(point: Point, polygon: Sequence[Point]) -> float:
    """Pixels to the nearest polygon edge. Zero if the point is inside."""
    if point_in_polygon(point, polygon):
        return 0.0
    count = len(polygon)
    return min(
        distance_point_to_segment(point, polygon[i], polygon[(i + 1) % count])
        for i in range(count)
    )


def bbox_horizontal_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """Horizontal pixel gap between two boxes; 0 when they overlap in x.

    Used for person-to-vehicle proximity. Horizontal only, because a camera
    looking down a driveway puts a person standing beside a car at a large
    *vertical* pixel offset that says nothing about how far apart they are.
    """
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    if ax2 < bx1:
        return bx1 - ax2
    if bx2 < ax1:
        return ax1 - bx2
    return 0.0


def bbox_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """Shortest pixel gap between two boxes in both axes; 0 when they overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return math.hypot(dx, dy)


def zone_config_snippet(zone_id: str, zone_type: str, polygon: Iterable[Point], risk: float = 0.5) -> str:
    """Render a zone as pasteable config.yaml. Used by tools/draw_zones.py."""
    points = ", ".join(f"[{x:.3f}, {y:.3f}]" for x, y in polygon)
    return (
        f"  - id: \"{zone_id}\"\n"
        f"    type: \"{zone_type}\"\n"
        f"    risk: {risk}\n"
        f"    polygon: [{points}]\n"
    )
