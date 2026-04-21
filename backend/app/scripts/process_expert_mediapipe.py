"""One-time offline expert MediaPipe preprocessing.

This is the MAIN way expert reference videos are preprocessed. It must
NEVER be invoked from the learner compare / evaluation path at runtime —
it is a manual, offline tool.

Typical usage from the ``backend/`` folder:

    python -m app.scripts.process_expert_mediapipe \
        --expert_code <expert_video_uuid>

    python -m app.scripts.process_expert_mediapipe \
        --chapter_id <chapter_uuid> --overwrite

    python -m app.scripts.process_expert_mediapipe \
        --expert_code <expert_video_uuid> \
        --video_path C:/path/to/raw/expert.mp4

Positional semantics:
    --expert_code     The expert Video row's UUID (canonical expert
                      reference identifier in this codebase). Alias for
                      ``--expert_video_id``.
    --chapter_id      Alternative lookup: resolve the expert video from
                      the chapter's expert reference row.
    --video_path      Optional override for the source video path. When
                      omitted, the path already stored on the Video row
                      is used.
    --overwrite       Rerun MediaPipe and replace any previously saved
                      outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional
from uuid import UUID

from app.core.database import SessionLocal
from app.services.expert_mediapipe_service import (
    ExpertMediaPipeError,
    ExpertNotFoundError,
    register_expert_mediapipe_reference,
)


logger = logging.getLogger(__name__)


def _parse_uuid(value: Optional[str], *, field: str) -> Optional[UUID]:
    if value is None:
        return None
    try:
        return UUID(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid {field} (must be a UUID): {value!r} ({exc})")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="process_expert_mediapipe",
        description=(
            "Run MediaPipe preprocessing on an expert reference video once "
            "and save the outputs under storage/expert/mediapipe/<expert_code>/."
        ),
    )
    parser.add_argument(
        "--expert_code",
        dest="expert_code",
        default=None,
        help="Expert Video row UUID (canonical expert reference identifier).",
    )
    parser.add_argument(
        "--expert_video_id",
        dest="expert_video_id",
        default=None,
        help="Alias for --expert_code.",
    )
    parser.add_argument(
        "--chapter_id",
        dest="chapter_id",
        default=None,
        help="Chapter UUID (resolve expert via chapter's expert reference row).",
    )
    parser.add_argument(
        "--video_path",
        dest="video_path",
        default=None,
        help=(
            "Optional override for the source video path. Absolute path, "
            "CWD-relative path, or storage-relative key."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun MediaPipe and replace any previously saved outputs.",
    )
    parser.add_argument(
        "--max_num_hands",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="Logging level (DEBUG / INFO / WARNING / ERROR).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    expert_code = args.expert_code or args.expert_video_id
    if expert_code is None and args.chapter_id is None:
        parser.error("Provide --expert_code (or --expert_video_id) or --chapter_id.")

    expert_video_id = _parse_uuid(expert_code, field="--expert_code")
    chapter_id = _parse_uuid(args.chapter_id, field="--chapter_id")

    video_path_arg = args.video_path
    if video_path_arg:
        # Let the service do final resolution; here we only normalise
        # "~" and quotes so the service sees a clean string/Path.
        video_path_arg = str(Path(video_path_arg).expanduser())

    session = SessionLocal()
    try:
        reference = register_expert_mediapipe_reference(
            session,
            expert_video_id=expert_video_id,
            chapter_id=chapter_id,
            video_path=video_path_arg,
            overwrite=args.overwrite,
            max_num_hands=args.max_num_hands,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
    except ExpertNotFoundError as exc:
        print(f"[FAIL] Expert not found: {exc}", file=sys.stderr)
        return 2
    except ExpertMediaPipeError as exc:
        print(f"[FAIL] Expert MediaPipe preprocessing failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"[FAIL] Unexpected error: {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()

    status_verb = "Reused existing" if reference.reused_existing else "Processed"
    print(f"[OK] {status_verb} expert MediaPipe reference for {reference.expert_code}.")
    print(
        json.dumps(
            reference.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
