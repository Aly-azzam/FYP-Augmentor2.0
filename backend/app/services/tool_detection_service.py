"""Tool Detection Service — detect craft tools in video frames.

Responsibilities:
- run tool detection model on frames
- output per-frame tool detections with class, confidence, bbox

Owner: Perception Engine (Person 1)
"""

from pathlib import Path


async def detect_tools(video_path: Path) -> list[dict]:
    """Detect tools across video frames.

    Returns list of per-frame tool detection dicts.
    """
    # TODO: implement tool detection model inference
    return []


async def detect_tools_in_frame(frame_data) -> list[dict]:
    """Detect tools in a single frame.

    Returns list of ToolDetection-compatible dicts.
    """
    # TODO: implement single-frame detection
    return []
