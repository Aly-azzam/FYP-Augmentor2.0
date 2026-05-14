"""VLM-powered coaching feedback generation.

Loads evaluation artefacts for a completed run, extracts learner and expert
frames at each error peak, sends them to the configured VLM, and saves the
resulting structured feedback to feedback.json.
"""

from __future__ import annotations

import base64
import bisect
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2

VLM_PROVIDER = "groq"  # options: "groq" | "anthropic"

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_STORAGE_ROOT = _BACKEND_ROOT / "storage"


# ── VLM dispatch ──────────────────────────────────────────────────────────────

def _call_vlm(messages: list) -> str:
    if VLM_PROVIDER == "groq":
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        r = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_tokens=1000,
            temperature=0.2,
            seed=42,
        )
        return r.choices[0].message.content

    elif VLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        r = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=messages,
        )
        return r.content[0].text

    raise ValueError(f"Unknown VLM_PROVIDER: {VLM_PROVIDER}")


# ── Frame helpers ─────────────────────────────────────────────────────────────

def _read_frame_b64(video_path: str, frame_index: int, max_width: int = 1280) -> str | None:
    """Seek to frame_index in video, resize to ≤max_width, return base64 JPEG."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        ret, frame = cap.read()
        if not ret:
            return None
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    finally:
        cap.release()


def _find_expert_frame(
    peak_frame: int,
    dtw_learner_frames: list[int],
    dtw_lookup: dict[int, int],
) -> int:
    """Return expert frame index aligned to peak_frame via DTW; fallback to peak_frame."""
    if not dtw_learner_frames:
        return peak_frame
    pos = bisect.bisect_left(dtw_learner_frames, peak_frame)
    candidates: list[int] = []
    if pos < len(dtw_learner_frames):
        candidates.append(dtw_learner_frames[pos])
    if pos > 0:
        candidates.append(dtw_learner_frames[pos - 1])
    closest = min(candidates, key=lambda x: abs(x - peak_frame))
    return dtw_lookup.get(closest, peak_frame)


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_feedback(run_id: str, expert_id: str, output_dir: str) -> str:
    """Generate VLM feedback for a run. Returns path to saved feedback.json."""
    vlm_dir = Path(output_dir).parent / "vlm"
    vlm_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = vlm_dir / "feedback.json"

    eval_dir = _STORAGE_ROOT / "evaluation" / run_id

    # ── Load unified errors ───────────────────────────────────────────────────
    with open(eval_dir / "score" / "unified_errors.json", encoding="utf-8") as fh:
        unified = json.load(fh)
    all_errors: list[dict] = unified.get("all_errors", [])

    # ── Load angle errors for peak_frame lookup ───────────────────────────────
    with open(eval_dir / "angle" / "angle_errors.json", encoding="utf-8") as fh:
        angle_data = json.load(fh)
    angle_peak_lookup: dict[tuple[int, int], int] = {}
    angle_event_lookup: dict[tuple[int, int], dict] = {}
    for evt in angle_data.get("error_events", []):
        fs, fe = evt["frame_start"], evt["frame_end"]
        angle_peak_lookup[(fs, fe)] = evt.get(
            "peak_frame", (fs + fe) // 2
        )
        angle_event_lookup[(fs, fe)] = evt

    # ── Load DTW alignment for expert frame mapping ───────────────────────────
    with open(eval_dir / "angle" / "dtw_alignment.json", encoding="utf-8") as fh:
        dtw_data = json.load(fh)
    dtw_lookup: dict[int, int] = {}
    for m in dtw_data.get("matches", []):
        li = m.get("learner_frame_index")
        ei = m.get("expert_frame_index")
        if li is not None and ei is not None:
            dtw_lookup[int(li)] = int(ei)
    dtw_learner_frames = sorted(dtw_lookup.keys())

    # ── Load expert video path ────────────────────────────────────────────────
    expert_meta_path = (
        _STORAGE_ROOT / "outputs" / "sam2_yolo" / "experts" / expert_id / "metadata.json"
    )
    with open(expert_meta_path, encoding="utf-8") as fh:
        expert_meta = json.load(fh)
    expert_video_path: str = expert_meta["source_video_path"]

    # ── Learner visualization video ───────────────────────────────────────────
    vis_video_path = str(eval_dir / "visualization" / "visualization.mp4")

    # ── Task description ──────────────────────────────────────────────────────
    task_description = "straight-line cutting task"
    summary_path = eval_dir / "score" / "run_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as fh:
            task_description = json.load(fh).get("task_description", task_description)

    # ── Sort errors by frame_start ────────────────────────────────────────────
    errors_sorted = sorted(all_errors, key=lambda e: e.get("frame_start", 0))

    # ── Build VLM message content ─────────────────────────────────────────────
    system_content = (
        'You are "Your Crafting Coach" — a warm, casual, encouraging coach who gives honest '
        "feedback to learners practicing hands-on crafting skills. You explain mistakes clearly "
        "like a friend who really knows what they're talking about. You use emojis naturally. "
        "You never sound robotic or like a report. You always end on a motivating note."
    )

    user_content: list[dict] = []

    user_content.append({
        "type": "text",
        "text": (
            f"Here are the results from a learner's attempt at a {task_description}.\n\n"
            "For each error moment I'm showing you two frames side by side — "
            "left is the learner's frame with visual overlays showing what went wrong, "
            "right is the expert at the same moment.\n\n"
        ),
    })

    for n, error in enumerate(errors_sorted, 1):
        error_type = error.get("error_type", "unknown")
        frame_start = error.get("frame_start", 0)
        frame_end = error.get("frame_end", 0)

        if error_type == "angle":
            peak_frame = angle_peak_lookup.get(
                (frame_start, frame_end), (frame_start + frame_end) // 2
            )
            peak_deg = error.get("peak_angle_diff_deg") or 0.0
            correction = (
                f"The learner's blade was {peak_deg:.0f}° off from the expert angle. "
                "Look at the two lines in the image — the white line is the learner, the cyan line is the expert. "
                "Describe which direction (clockwise or counter-clockwise, left or right) the learner needed to "
                "rotate their blade to match the expert."
            )
            error_desc = f"Blade angle was {peak_deg:.0f}° off the expert angle\n{correction}"
            t_start = error.get("timestamp_start_sec", round(frame_start / 30, 1))
            t_end = error.get("timestamp_end_sec", round(frame_end / 30, 1))
            label = f"Angle error (from {t_start}s to {t_end}s)"
        else:
            peak_frame = (frame_start + frame_end) // 2
            deviation = error.get("peak_deviation_px") or 0.0
            drift_dir = "right" if deviation > 0 else "left"
            correct_dir = "left" if deviation > 0 else "right"
            error_desc = (
                f"Scissors drifted from the expert cutting path\n"
                f"The scissors drifted {drift_dir} of the expert path. To self-correct, "
                f"the learner needed to steer back {correct_dir} toward the cutting line."
            )
            t_start = round(frame_start / 30, 1)
            t_end = round(frame_end / 30, 1)
            label = f"Trajectory error (from {t_start}s to {t_end}s)"

        expert_frame = _find_expert_frame(peak_frame, dtw_learner_frames, dtw_lookup)

        learner_b64 = _read_frame_b64(vis_video_path, peak_frame)
        expert_b64 = _read_frame_b64(expert_video_path, expert_frame)

        user_content.append({"type": "text", "text": f"--- {label} ---\n"})

        if learner_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{learner_b64}"},
            })
        if expert_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{expert_b64}"},
            })

        user_content.append({"type": "text", "text": error_desc + "\n\n"})

    write_prompt = (
        'Write feedback as "Your Crafting Coach". You MUST follow this exact structure, no deviations:\n\n'
        "🎯 **Overall**\n"
        '[2-3 sentences about the full attempt — mention timing e.g. "in the first few seconds"]\n\n'
    )
    for n in range(1, len(errors_sorted) + 1):
        write_prompt += (
            f"⚠️ **Issue {n} — [short title you invent]**\n"
            '[2-3 sentences — describe what you see visually, reference the time range e.g. "around the 3-second mark". '
            "Compare learner frame to expert frame. Be specific. Include the specific correction needed in that moment "
            '(e.g. "you needed to rotate your blade X° to the right" or "you needed to steer back left to recover the line").]\n'
            "✅ **What to do next time:** [1-2 sentences of concrete advice.]\n\n"
        )
    write_prompt += (
        "🏆 **Priority fix:** [One sentence — the most important thing to fix.]\n"
        "💪 **What you did well:** [One genuine positive observation.]\n\n"
        "STRICT RULES:\n"
        '- Never say "frame", "error frame", "Error 1/2/3", "deviation", "DTW", "corridor", "pixel"\n'
        '- Always reference errors by time range e.g. "around 3s" or "between 2s and 5s"\n'
        "- Each Issue block must have exactly the ⚠️ header, 2-3 sentences, and ✅ line\n"
        "- Do not add any text before 🎯 or after the 💪 line\n"
        "- Keep total response under 400 words\n"
        '- Always address the learner directly as "you", never say "the learner"\n'
        "- For each issue, explain what specific correction was needed in that moment — make it feel like real-time coaching\n"
    )
    user_content.append({"type": "text", "text": write_prompt})

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # ── Call VLM ──────────────────────────────────────────────────────────────
    feedback_text = _call_vlm(messages)

    # ── Save feedback.json ────────────────────────────────────────────────────
    result = {
        "run_id": run_id,
        "task_description": task_description,
        "feedback": feedback_text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(feedback_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print(f"[VLM] Feedback saved to {feedback_path}")
    return str(feedback_path)
