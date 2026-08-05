"""Tests for the optional frame context on /api/v1/scan-face.

The rule these all serve: a malformed or missing face box must never break a
face scan. The match is still valid without it — only the behavioural join is
lost — so every bad input degrades to None rather than raising.

    python tests/test_frame_context.py
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.frame_context import (  # noqa: E402
    DEFAULT_CAMERA_ID,
    build_frame_context,
    face_box_centre,
    normalise_face_box,
    parse_face_box,
)


# --- parsing --------------------------------------------------------------
def test_parses_the_json_string_the_browser_sends():
    box = parse_face_box('{"x": 120, "y": 80, "w": 64, "h": 72}')
    assert box == {"x": 120.0, "y": 80.0, "w": 64.0, "h": 72.0}


def test_accepts_an_already_decoded_dict():
    assert parse_face_box({"x": 1, "y": 2, "w": 3, "h": 4})["w"] == 3.0


def test_absent_box_is_simply_absent():
    assert parse_face_box(None) is None
    assert parse_face_box("") is None


def test_malformed_input_never_raises():
    """Every one of these must degrade to None, not blow up the scan."""
    for bad in (
        "not json",
        "[1,2,3]",
        "null",
        '{"x": 1}',                              # incomplete
        '{"x": "a", "y": 2, "w": 3, "h": 4}',    # non-numeric
        '{"x": 1, "y": 2, "w": 0, "h": 4}',      # no area
        '{"x": 1, "y": 2, "w": -5, "h": 4}',     # negative
        '{"x": null, "y": 2, "w": 3, "h": 4}',
    ):
        assert parse_face_box(bad) is None, f"{bad!r} should have been rejected"


def test_nan_and_infinity_are_rejected():
    """float('nan') parses fine and then poisons every later comparison."""
    assert parse_face_box('{"x": NaN, "y": 2, "w": 3, "h": 4}') is None
    assert parse_face_box('{"x": Infinity, "y": 2, "w": 3, "h": 4}') is None


# --- normalising ----------------------------------------------------------
def test_normalises_against_the_source_frame():
    box = {"x": 480.0, "y": 360.0, "w": 96.0, "h": 72.0}
    assert normalise_face_box(box, 960, 720) == {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}


def test_no_frame_size_means_no_normalised_box():
    """Better no box than one normalised against a guessed denominator — the
    latter looks usable and points at the wrong part of the picture."""
    box = {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}
    assert normalise_face_box(box, None, None) is None
    assert normalise_face_box(box, 960, None) is None
    assert normalise_face_box(box, 0, 0) is None


def test_boxes_overhanging_the_frame_are_clamped_not_rejected():
    """Detectors routinely return a box a few pixels off the edge."""
    box = {"x": -10.0, "y": -8.0, "w": 100.0, "h": 100.0}
    result = normalise_face_box(box, 200, 200)
    assert result["x"] == 0.0 and result["y"] == 0.0
    assert 0 < result["w"] <= 1.0 and 0 < result["h"] <= 1.0


def test_a_normalised_box_never_extends_past_the_frame():
    box = {"x": 180.0, "y": 180.0, "w": 100.0, "h": 100.0}
    result = normalise_face_box(box, 200, 200)
    assert result["x"] + result["w"] <= 1.0 + 1e-9
    assert result["y"] + result["h"] <= 1.0 + 1e-9


def test_centre_is_the_middle_of_the_box():
    """The join tests the centre, not a corner: on a turned head a corner of the
    face box can fall outside the body box while the centre stays on the person."""
    centre = face_box_centre({"x": 0.4, "y": 0.2, "w": 0.2, "h": 0.4})
    assert centre == {"x": 0.5, "y": 0.4}
    assert face_box_centre(None) is None


# --- the request-level helper ---------------------------------------------
def test_full_context_from_a_browser_style_form():
    context = build_frame_context({
        "face_box": '{"x": 480, "y": 360, "w": 96, "h": 72}',
        "frame_width": "960",
        "frame_height": "720",
        "camera_id": "gate_cam_01",
        "captured_at": "2026-08-05T14:22:07.000Z",
    })

    assert context["face_box_normalised"] == {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}
    assert context["face_box_centre"] == {"x": 0.55, "y": 0.55}
    assert context["frame_size"] == {"w": 960, "h": 720}
    assert context["camera_id"] == "gate_cam_01"
    assert context["captured_at"] == "2026-08-05T14:22:07.000Z"


def test_empty_form_keeps_todays_behaviour():
    """The endpoint must behave exactly as before for callers that send nothing."""
    context = build_frame_context({})
    assert context["face_box"] is None
    assert context["face_box_normalised"] is None
    assert context["face_box_centre"] is None
    assert context["frame_size"] is None
    assert context["captured_at"] is None
    assert context["camera_id"] == DEFAULT_CAMERA_ID


def test_box_without_frame_size_yields_no_normalised_box():
    context = build_frame_context({"face_box": '{"x": 1, "y": 2, "w": 3, "h": 4}'})
    assert context["face_box"] is not None       # kept for debugging
    assert context["face_box_normalised"] is None  # but not usable for the join
    assert context["face_box_centre"] is None


def test_absurd_frame_dimensions_are_rejected():
    for width in ("0", "-960", "abc", "999999999"):
        context = build_frame_context({
            "face_box": '{"x": 1, "y": 2, "w": 3, "h": 4}',
            "frame_width": width,
            "frame_height": "720",
        })
        assert context["frame_size"] is None, f"frame_width={width} should be rejected"


def test_camera_id_is_bounded_and_trimmed():
    assert build_frame_context({"camera_id": "  gate_01  "})["camera_id"] == "gate_01"
    assert build_frame_context({"camera_id": ""})["camera_id"] == DEFAULT_CAMERA_ID
    assert len(build_frame_context({"camera_id": "x" * 500})["camera_id"]) <= 64


def test_context_carries_no_identity_fields():
    """This helper handles geometry. Identity stays in the recognition module."""
    context = build_frame_context({
        "face_box": '{"x": 1, "y": 2, "w": 3, "h": 4}',
        "frame_width": "100",
        "frame_height": "100",
        "full_name": "should be ignored",
        "person_id": "12",
    })
    assert "full_name" not in context
    assert "person_id" not in context


def _run_all():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, function in tests:
        try:
            function()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
