"""Zone authoring tool — click a polygon onto a real frame, get pasteable config.

The default zones in config.yaml describe an imaginary property camera. Real
footage never matches them, so this draws a frame from YOUR video and lets you
click the actual gate, wall, driveway and pavement.

    python tools/draw_zones.py --source "../behaviour analysis/clip.mp4"
    python tools/draw_zones.py --source clip.mp4 --at 12.5      # a later frame

Controls:
    left click      add a point
    right click     undo the last point
    ENTER           finish this zone, choose its type, start the next
    t               cycle the zone type for the polygon in progress
    s               save the YAML snippet to zones_snippet.yaml
    r               reset everything
    q / ESC         quit (prints the snippet)

Output is normalised 0..1 coordinates, so the zones keep working if the camera
resolution changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from debug_overlay import ZONE_COLOURS  # noqa: E402
from frame_ingest import open_source  # noqa: E402
from zones import zone_config_snippet  # noqa: E402

ZONE_TYPES = [
    "property_boundary",
    "gate",
    "vertical_structure",
    "vehicle_zone",
    "street_frontage",
    "exempt",
]

HELP_LINES = [
    "click: add point   right-click: undo   ENTER: finish zone",
    "t: change type   s: save snippet   r: reset   q: quit",
]


class ZoneDrawer:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.height, self.width = frame.shape[:2]
        self.current: list[tuple[int, int]] = []
        self.zones: list[tuple[str, str, list[tuple[int, int]]]] = []
        self.type_index = 0

    @property
    def current_type(self) -> str:
        return ZONE_TYPES[self.type_index % len(ZONE_TYPES)]

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.current:
            self.current.pop()

    def finish_zone(self) -> None:
        if len(self.current) < 3:
            print("A zone needs at least 3 points.")
            return
        zone_type = self.current_type
        zone_id = f"{zone_type}_{sum(1 for z in self.zones if z[1] == zone_type) + 1}"
        self.zones.append((zone_id, zone_type, list(self.current)))
        print(f"Added '{zone_id}' ({zone_type}) with {len(self.current)} points.")
        self.current.clear()

    def snippet(self) -> str:
        if not self.zones:
            return "zones: []\n"
        out = ["zones:"]
        for zone_id, zone_type, points in self.zones:
            normalised = [(x / self.width, y / self.height) for x, y in points]
            risk = 0.0 if zone_type == "exempt" else 0.5
            out.append(zone_config_snippet(zone_id, zone_type, normalised, risk).rstrip("\n"))
        return "\n".join(out) + "\n"

    def render(self) -> np.ndarray:
        canvas = self.frame.copy()

        for zone_id, zone_type, points in self.zones:
            colour = ZONE_COLOURS.get(zone_type, (180, 180, 180))
            array = np.array(points, dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [array], colour)
            cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
            cv2.polylines(canvas, [array], True, colour, 2)
            cv2.putText(canvas, zone_id, tuple(array[0] + np.array([4, 16])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

        colour = ZONE_COLOURS.get(self.current_type, (255, 255, 255))
        for i, point in enumerate(self.current):
            cv2.circle(canvas, point, 4, colour, -1)
            if i:
                cv2.line(canvas, self.current[i - 1], point, colour, 1)
        if len(self.current) > 2:
            cv2.line(canvas, self.current[-1], self.current[0], colour, 1, cv2.LINE_AA)

        banner = [f"drawing: {self.current_type}   zones so far: {len(self.zones)}", *HELP_LINES]
        cv2.rectangle(canvas, (0, 0), (min(self.width, 520), 16 + 20 * len(banner)), (25, 25, 25), -1)
        for i, line in enumerate(banner):
            cv2.putText(canvas, line, (10, 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (230, 230, 230), 1, cv2.LINE_AA)
        return canvas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Draw zone polygons on a video frame.")
    parser.add_argument("--source", "-s", required=True, help="Video file or camera index.")
    parser.add_argument("--at", type=float, default=0.0, help="Seconds into the video to grab.")
    parser.add_argument("--out", default="zones_snippet.yaml", help="Where to save the snippet.")
    args = parser.parse_args(argv)

    source = int(args.source) if args.source.isdigit() else args.source
    capture, info = open_source(source)
    if args.at > 0 and not info.is_live:
        capture.set(cv2.CAP_PROP_POS_MSEC, args.at * 1000.0)

    ok, frame = capture.read()
    capture.release()
    if not ok:
        print(f"Could not read a frame from {args.source}", file=sys.stderr)
        return 1

    print(f"{info.describe()}\nDraw your zones. Coordinates are saved normalised (0..1).")

    drawer = ZoneDrawer(frame)
    window = "Draw zones — ENTER finishes a zone, q quits"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, drawer.on_mouse)

    while True:
        cv2.imshow(window, drawer.render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (13, 10):
            drawer.finish_zone()
        elif key == ord("t"):
            drawer.type_index += 1
        elif key == ord("r"):
            drawer.zones.clear()
            drawer.current.clear()
        elif key == ord("s"):
            Path(args.out).write_text(drawer.snippet(), encoding="utf-8")
            print(f"Saved to {args.out}")

    cv2.destroyAllWindows()

    snippet = drawer.snippet()
    print("\nPaste this into config.yaml, replacing the `zones:` block:\n")
    print(snippet)
    if drawer.zones:
        Path(args.out).write_text(snippet, encoding="utf-8")
        print(f"(also written to {args.out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
