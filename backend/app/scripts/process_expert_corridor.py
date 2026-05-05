"""CLI script: build expert corridor from a smoothed trajectory.

Usage (from the backend/ directory with the venv active):

    python -m app.scripts.process_expert_corridor \\
        --expert_code straight_line_v1 \\
        --margin_px 40

The script locates:
    backend/storage/outputs/sam2_yolo/experts/<expert_code>/trajectory_smoothed.json

and writes:
    corridor.json
    corridor_preview.png
    corridor_overlay.mp4

in the same directory.

Optional flags:
    --overwrite    Re-run even if corridor.json already exists.
    --margin_px    Corridor half-width in pixels (default: 40).
    --storage_root Override storage root (default: auto-detected).
"""

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build expert corridor from trajectory_smoothed.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--expert_code", required=True, help="Expert code directory name")
    p.add_argument("--margin_px", type=int, default=40, help="Corridor half-width (px)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing corridor.json")
    p.add_argument(
        "--storage_root",
        default=None,
        help=(
            "Root storage path.  Defaults to <backend_root>/storage where "
            "<backend_root> is the parent of this file's package."
        ),
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve paths
    backend_root = Path(__file__).resolve().parents[2]  # …/backend
    storage_root = Path(args.storage_root) if args.storage_root else backend_root / "storage"
    expert_dir = storage_root / "outputs" / "sam2_yolo" / "experts" / args.expert_code
    smoothed_path = expert_dir / "trajectory_smoothed.json"

    if not smoothed_path.is_file():
        print(
            f"[ERROR] trajectory_smoothed.json not found at: {smoothed_path}\n"
            "        Run the SAM2/YOLO offline pipeline first:\n"
            "          python -m app.scripts.process_expert_sam2_yolo "
            f"--expert_code {args.expert_code} ...",
            file=sys.stderr,
        )
        sys.exit(1)

    corridor_path = expert_dir / "corridor.json"
    if corridor_path.is_file() and not args.overwrite:
        print(
            f"[SKIP] corridor.json already exists at {corridor_path}.\n"
            "       Pass --overwrite to regenerate."
        )
        sys.exit(0)

    from app.services.sam2_yolo.corridor import build_expert_corridor  # noqa: PLC0415

    print(f"[corridor] Building corridor for expert: {args.expert_code}")
    print(f"[corridor] Input : {smoothed_path}")
    print(f"[corridor] Output: {expert_dir}")
    print(f"[corridor] margin_px = {args.margin_px}")
    print()

    result = build_expert_corridor(
        smoothed_trajectory_path=str(smoothed_path),
        output_dir=str(expert_dir),
        margin_px=args.margin_px,
    )

    usable = result.get("usable_for_learner_comparison", False)
    print()
    print(f"[corridor] Done.  usable_for_learner_comparison = {usable}")


if __name__ == "__main__":
    main()
