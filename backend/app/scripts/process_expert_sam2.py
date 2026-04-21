"""One-time offline expert SAM 2 preprocessing.

This is the MAIN way expert reference videos are preprocessed with SAM 2.
It must NEVER be invoked from the learner compare / evaluation path at
runtime — it is a manual, offline tool.

SAM 2 is initialized automatically from the expert's MediaPipe features,
so the MediaPipe expert flow must run FIRST. A typical end-to-end admin
session looks like:

    # 1) MediaPipe
    python -m app.scripts.process_expert_mediapipe --expert_code <uuid>

    # 2) SAM 2
    python -m app.scripts.process_expert_sam2 --expert_code <uuid>

    python -m app.scripts.process_expert_sam2 \
        --chapter_id <chapter_uuid> --overwrite

    python -m app.scripts.process_expert_sam2 \
        --expert_code <uuid> \
        --video_path C:/path/to/raw/expert.mp4

Positional semantics:
    --expert_code     The expert Video row's UUID (canonical expert
                      reference identifier in this codebase). Alias for
                      ``--expert_video_id``.
    --chapter_id      Alternative lookup: resolve the expert video from
                      the chapter's expert reference row.
    --video_path      Optional override for the source video path. When
                      omitted, the MediaPipe expert ``source.mp4`` is
                      used if present, otherwise the path stored on the
                      Video row.
    --overwrite       Rerun SAM 2 and replace any previously saved outputs.
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
from app.services.expert_sam2_service import (
    ExpertMediaPipeMissingError,
    ExpertNotFoundError,
    ExpertSam2Error,
    register_expert_sam2_reference,
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
        prog="process_expert_sam2",
        description=(
            "Run SAM 2 preprocessing on an expert reference video once and "
            "save the outputs under storage/expert/sam2/<expert_code>/. "
            "Requires that process_expert_mediapipe was run first."
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
        help="Rerun SAM 2 and replace any previously saved outputs.",
    )
    parser.add_argument(
        "--no_annotation",
        action="store_true",
        help="Skip rendering the annotated.mp4 overlay video.",
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
        video_path_arg = str(Path(video_path_arg).expanduser())

    session = SessionLocal()
    try:
        reference = register_expert_sam2_reference(
            session,
            expert_video_id=expert_video_id,
            chapter_id=chapter_id,
            video_path=video_path_arg,
            overwrite=args.overwrite,
            render_annotation=not args.no_annotation,
        )
    except ExpertNotFoundError as exc:
        print(f"[FAIL] Expert not found: {exc}", file=sys.stderr)
        return 2
    except ExpertMediaPipeMissingError as exc:
        print(
            f"[FAIL] MediaPipe expert reference is required first: {exc}",
            file=sys.stderr,
        )
        return 3
    except ExpertSam2Error as exc:
        print(f"[FAIL] Expert SAM 2 preprocessing failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"[FAIL] Unexpected error: {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()

    status_verb = "Reused existing" if reference.reused_existing else "Processed"
    print(f"[OK] {status_verb} expert SAM 2 reference for {reference.expert_code}.")
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
