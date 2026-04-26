from __future__ import annotations

import json
from pathlib import Path

from .schemas import RawOpticalFlowResult, SummaryOpticalFlowResult


def ensure_output_dir(output_dir: str | Path) -> Path:
    """
    Create the output directory if it does not exist.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_output_paths(
    output_dir: str | Path,
    run_id: str,
) -> tuple[Path, Path]:
    """
    Build the raw and summary JSON output paths for one run.
    """
    output_path = ensure_output_dir(output_dir)

    raw_path = output_path / f"optical_flow_{run_id}_raw.json"
    summary_path = output_path / f"optical_flow_{run_id}_summary.json"

    return raw_path, summary_path


def save_raw_result(
    raw_result: RawOpticalFlowResult,
    output_dir: str | Path,
) -> Path:
    """
    Save the raw optical flow result JSON.
    """
    raw_path, _ = build_output_paths(output_dir, raw_result.run.run_id)

    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(
            raw_result.model_dump(mode="json"),
            f,
            indent=2,
            ensure_ascii=False,
        )

    return raw_path


def save_summary_result(
    summary_result: SummaryOpticalFlowResult,
    output_dir: str | Path,
) -> Path:
    """
    Save the summary optical flow result JSON.
    """
    _, summary_path = build_output_paths(output_dir, summary_result.run.run_id)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary_result.model_dump(mode="json"),
            f,
            indent=2,
            ensure_ascii=False,
        )

    return summary_path


def save_optical_flow_results(
    raw_result: RawOpticalFlowResult,
    summary_result: SummaryOpticalFlowResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Save both raw and summary JSON outputs.

    Returns:
        (raw_json_path, summary_json_path)
    """
    raw_path = save_raw_result(raw_result, output_dir)
    summary_path = save_summary_result(summary_result, output_dir)
    return raw_path, summary_path
