"""Guardian Collective — behavioural risk analysis.

A second, independent signal alongside facial recognition and LPR. It watches
body posture and movement, scores how unusual they are, and fuses that with the
face module's confidence to produce one composite score.

It reports MOVEMENT, never IDENTITY, and it never takes an action: the only
decision it makes is whether a human should look.

    from behavioural_analysis import BehaviouralPipeline, load_settings

    pipeline = BehaviouralPipeline(load_settings())
    for result in pipeline.run("clip.mp4"):
        for event in result.events:
            ...
"""

import sys
from pathlib import Path

# The submodules import each other flat (`from settings import ...`), matching
# how backend/services and backend/routes are written. Putting this directory on
# the path lets the package be imported from backend/ as well as run from here.
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from api_output import build_event, format_for_console, push_to_flask_api  # noqa: E402
from audit_log import AuditLog  # noqa: E402
from heuristics import HEURISTIC_NAMES, HeuristicResult, evaluate_all  # noqa: E402
from pipeline import BehaviouralPipeline, FrameResult  # noqa: E402
from risk_fusion import FaceSignal, face_signal_from_recognition, score_event  # noqa: E402
from settings import Settings, load_settings  # noqa: E402

__all__ = [
    "AuditLog",
    "BehaviouralPipeline",
    "FaceSignal",
    "FrameResult",
    "HEURISTIC_NAMES",
    "HeuristicResult",
    "Settings",
    "build_event",
    "evaluate_all",
    "face_signal_from_recognition",
    "format_for_console",
    "load_settings",
    "push_to_flask_api",
    "score_event",
]
