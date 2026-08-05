"""CLI entry point — run the behavioural analysis over a video or camera.

    python main.py --source "../behaviour analysis/clip.mp4" --show
    python main.py --source 0 --show                      # webcam
    python main.py --source clip.mp4 --save-json out.json

Run it from this directory (backend/behavioural_analysis/).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow both `python main.py` from here and `python -m behavioural_analysis.main`
# from backend/ — the modules import each other flat, like backend/services does.
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import api_output  # noqa: E402
from pipeline import BehaviouralPipeline, static_face_provider  # noqa: E402
from settings import ConfigError, load_settings  # noqa: E402

logger = logging.getLogger("behavioural_analysis")

BANNER = """
 Guardian Collective — Behavioural Analysis
 -------------------------------------------------------------------------
 A SECOND, INDEPENDENT SIGNAL alongside facial recognition and LPR.
 It reports movement, not identity. It never takes an action on its own:
 the only thing it can decide is whether a HUMAN should look.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="behavioural_analysis",
        description="Detect suspicious movement patterns and score them for human review.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", "-s", required=True,
        help="Video file path, or a camera index such as 0.",
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to config.yaml (default: the one beside this script).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open the OpenCV debug window (boxes, skeletons, zones, scores).",
    )
    parser.add_argument(
        "--save-json", default=None,
        help="Write all events from the run to this JSON file.",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=None,
        help="Stop after this many seconds of footage.",
    )
    parser.add_argument(
        "--start-seconds", type=float, default=0.0,
        help="Skip to this point in the video before starting.",
    )
    parser.add_argument(
        "--stride", type=int, default=None,
        help="Process every Nth frame (overrides config). Raise it if playback lags.",
    )
    parser.add_argument(
        "--push", action="store_true",
        help="Call push_to_flask_api for each event. Without output.flask_api_url set "
             "in config.yaml this only logs what it WOULD post — the Flask side is a stub.",
    )
    parser.add_argument(
        "--face-confidence", type=float, default=None,
        help="DEMO AID: a fixed facial-match confidence (0..1) for every track, so the "
             "fusion rules can be shown without running the face module. Clearly marked "
             "as operator-supplied in the output.",
    )
    parser.add_argument(
        "--face-label", default=None,
        choices=["offender", "suspect", "verified"],
        help="DEMO AID: the label accompanying --face-confidence. 'verified' demonstrates "
             "a known resident being damped down.",
    )
    parser.add_argument(
        "--hold", type=float, default=20.0,
        help="With --show, keep the last frame on screen for this many seconds "
             "after the video ends (press any key to close sooner). 0 closes "
             "immediately. Without it the window vanishes the moment the clip ends.",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="At the end, summarise why heuristics did NOT fire. This is the "
             "threshold-tuning view: it turns 'nothing detected' into 'the dwell "
             "was 3.4s against a 45s threshold'.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Only print the event summary, not each event as it happens.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Debug logging.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Ultralytics and MediaPipe are chatty at INFO.
    for noisy in ("ultralytics", "mediapipe", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    print(BANNER)

    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    print(f" config : {settings.source_path}")
    print(f" zones  : {len(settings.zones)} configured "
          f"({sum(1 for z in settings.zones if z.is_exempt)} exempt)")
    print(f" review : composite >= {settings.fusion.review_threshold}")

    face_provider = static_face_provider(args.face_confidence, args.face_label)
    if face_provider is not None:
        print(f" face   : OPERATOR-SUPPLIED confidence={args.face_confidence} "
              f"label={args.face_label} (demo aid, not the real face module)")
    else:
        print(" face   : none — behavioural signal only")
    print()

    source: str | int = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    show = args.show
    window = "Guardian Collective — behavioural analysis (q to quit)"
    if show:
        import cv2
        import debug_overlay

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    pipeline = BehaviouralPipeline(
        settings, face_provider=face_provider, explain_misses=args.explain
    )
    latest_event = None
    interrupted = False

    # Body positions, batched. They are published so a face match can be
    # attached to a body — the half of the join only this module can supply.
    # Batched rather than sent per frame because at a few frames a second that
    # would be a request every 300ms for data nobody reads in real time.
    events_url = settings.output.get("flask_api_url")
    tracks_url = events_url.replace("/events", "/tracks") if events_url else None
    snapshot_buffer = []
    last_flush = [0.0]
    FLUSH_SECONDS = 2.0

    # Rolling footage buffer. A flag fires at the END of the behaviour, so
    # frames are held continuously and the clip is cut from BEFORE the trigger.
    clip_cfg = settings.__dict__.get("clips") or settings.pipeline.get("clips")
    clips = None
    if clip_cfg is None:
        try:
            import yaml as _yaml

            with open(settings.source_path, "r", encoding="utf-8") as handle:
                clip_cfg = (_yaml.safe_load(handle) or {}).get("clips")
        except Exception:
            clip_cfg = None

    if clip_cfg and clip_cfg.get("enabled", True):
        from clip_buffer import ClipBuffer

        clips = ClipBuffer(
            pre_seconds=float(clip_cfg.get("pre_seconds", 12.0)),
            post_seconds=float(clip_cfg.get("post_seconds", 5.0)),
            jpeg_quality=int(clip_cfg.get("jpeg_quality", 70)),
            max_bytes=int(clip_cfg.get("max_buffer_mb", 96)) * 1024 * 1024,
            output_dir=MODULE_DIR / str(clip_cfg.get("output_dir", "clips")),
            fps=float(clip_cfg.get("playback_fps", 6.0)),
        )

    # event_id -> review_id, learned from the ingest response, so a clip can be
    # attached to the right review once its trailing seconds have been captured.
    review_ids = {}
    clip_url_base = events_url.rsplit("/events", 1)[0] if events_url else None

    def upload_clip(clip_id, path):
        review_id = review_ids.get(clip_id)
        if not (args.push and clip_url_base and review_id):
            return
        try:
            import requests

            with open(path, "rb") as handle:
                response = requests.post(
                    f"{clip_url_base}/review-queue/{review_id}/clip",
                    files={"file": (path.name, handle.read(), "video/mp4")},
                    timeout=30,
                )
            if response.ok:
                print(f" Clip for {review_id} uploaded ({path.stat().st_size // 1024}KB).")
            else:
                logger.warning("Clip upload for %s failed: %s", review_id, response.text[:200])
        except Exception as exc:
            # The clip is still on disk; the review stands without it.
            logger.warning("Clip upload for %s failed: %s", review_id, exc)

    # Body-position pushes go to a background thread. Posting them inline cost
    # about a quarter of the frame rate — the analysis would sit waiting on an
    # HTTP round trip while the camera moved on. Nothing downstream needs the
    # positions synchronously; they are read later, when a face scan arrives.
    from concurrent.futures import ThreadPoolExecutor

    pusher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="track-push")

    def flush_snapshots(force=False):
        if not (args.push and tracks_url and snapshot_buffer):
            return
        now = snapshot_buffer[-1].get("_t", 0.0)
        if not force and (now - last_flush[0]) < FLUSH_SECONDS:
            return
        batch = [{k: v for k, v in s.items() if k != "_t"} for s in snapshot_buffer]
        snapshot_buffer.clear()
        last_flush[0] = now
        pusher.submit(
            api_output.push_tracks_to_flask_api,
            batch,
            camera_id=settings.output.location_zone_id,
            url=tracks_url,
            timeout=float(settings.output.get("api_timeout_seconds", 5.0)),
        )

    def on_frame(result) -> bool:
        nonlocal latest_event

        if clips is not None:
            clips.add(result.frame.timestamp, result.frame.image)
            for clip_id, path in clips.tick(result.frame.timestamp):
                upload_clip(clip_id, path)

        if args.push and tracks_url:
            snapshot = api_output.track_snapshot(
                result, pipeline.event_timestamp(result.frame.timestamp)
            )
            if snapshot:
                snapshot["_t"] = result.frame.timestamp
                snapshot_buffer.append(snapshot)
                flush_snapshots()

        for event in result.events:
            if not args.quiet:
                print(api_output.format_for_console(event))

            # Only flags that reached a human get footage written. Everything
            # else stays in the memory ring buffer and is never saved anywhere.
            if clips is not None and event.get("requires_human_review"):
                clips.request(
                    event["event_id"].replace("/", "-").replace(":", "-"),
                    result.frame.timestamp,
                    label=event["track_id"],
                )

            if args.push:
                outcome = api_output.push_to_flask_api(
                    event,
                    url=settings.output.get("flask_api_url"),
                    timeout=float(settings.output.get("api_timeout_seconds", 5.0)),
                    dry_run=not settings.output.get("flask_api_url"),
                )
                review_id = (outcome or {}).get("review_id")
                if review_id:
                    review_ids[event["event_id"].replace("/", "-").replace(":", "-")] = review_id
            latest_event = event

        if show:
            import cv2
            import debug_overlay

            canvas = debug_overlay.render_frame(
                result.frame.image,
                zone_index=pipeline._zones_for(result.frame.width, result.frame.height),
                detections=result.detections,
                poses=result.poses,
                scores=result.scores,
                triggered=result.triggered,
                timestamp=result.frame.timestamp,
                frame_index=result.frame.index,
                events_so_far=len(pipeline.events),
                review_threshold=float(settings.fusion.review_threshold),
                latest_event=latest_event,
                note="behaviour only — no identity data in this module",
            )
            cv2.imshow(window, canvas)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return False
        return True

    try:
        pipeline.warm_up()
        for _ in pipeline.run(
            source,
            max_seconds=args.max_seconds,
            start_seconds=args.start_seconds,
            stride=args.stride,
            on_frame=on_frame,
        ):
            pass
        flush_snapshots(force=True)
        # A flag in the final seconds still gets whatever footage exists.
        if clips is not None:
            for clip_id, path in clips.flush():
                upload_clip(clip_id, path)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted.")
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1
    finally:
        # Let queued position pushes finish before the process exits.
        pusher.shutdown(wait=True)
        if show:
            import cv2

            # Hold the final frame so it can actually be looked at — otherwise
            # the window disappears the instant the clip ends.
            if args.hold > 0 and not interrupted:
                print(f"\n Holding the window for {args.hold:.0f}s — press any key to close.")
                cv2.waitKey(int(args.hold * 1000))
            cv2.destroyAllWindows()
            cv2.waitKey(1)      # let Win32 actually dispose of the window
        pipeline.close()

    events = pipeline.events
    review = [e for e in events if e["requires_human_review"]]

    print()
    print("-" * 78)
    print(f" {len(events)} event(s) detected"
          + (" (run interrupted)" if interrupted else "")
          + f", {len(review)} flagged for human review.")
    if settings.audit.get("enabled", True):
        print(f" Audit trail: {settings.audit.get('sqlite_path')} "
              f"and {settings.audit.get('jsonl_path')} (in {MODULE_DIR}).")
    print("-" * 78)

    if args.explain and pipeline.miss_reasons:
        print()
        print(" Why heuristics did not fire (most common first):")
        for (name, reason), count in pipeline.miss_reasons.most_common(12):
            print(f"   {count:5d}x  {name}")
            print(f"          {reason[:110]}")
        print()
        print(" Use this to tune config.yaml — every number above comes from it.")

    if args.save_json:
        api_output.write_events(events, args.save_json)
        print(f" Events written to {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
