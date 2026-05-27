"""File name constants and output JSON shape documentation for the angle pipeline.

JSON structures match yolo_scissors_test exactly.
"""

from __future__ import annotations

# ── Output file names ─────────────────────────────────────────────────────────

ANGLES_FILENAME = "angles.json"
YOLO_DETECTIONS_FILENAME = "yolo_detections.json"
DTW_ALIGNMENT_FILENAME = "dtw_alignment.json"
LEARNER_COMPARISON_FILENAME = "learner_angle_comparison.json"
RUN_SUMMARY_FILENAME = "run_summary.json"
DTW_PREVIEW_FILENAME = "dtw_aligned_preview.mp4"
LEARNER_COMPARISON_VIDEO_FILENAME = "learner_comparison_output.mp4"


# ── JSON shape documentation (as inline comments) ────────────────────────────
#
# angles.json (expert and learner):
# {
#   "video_type": "expert" | "learner",
#   "video_path": str,
#   "total_frames": int,
#   "fps": float,
#   "frames": [
#     {
#       "frame_index": int,
#       "timestamp_sec": float,
#       "detected": bool,
#       "valid_line": bool,
#       "confidence": float | null,
#       "bbox": {"x1","y1","x2","y2","width","height"} | null,
#       "line_center": {"x","y"} | null,
#       "line_angle": float | null,      # [0, 180) degrees
#       "line_start": {"x","y"} | null,
#       "line_end": {"x","y"} | null,
#       "line_source": "fitline"|"bbox_diagonal"|"previous_frame"|"none",
#       "fallback_used": bool
#     }
#   ]
# }
#
# yolo_detections.json (expert only, raw detections per frame):
# {
#   "video_path": str,
#   "total_frames": int,
#   "fps": float,
#   "frame_stride": int,
#   "confidence_threshold": float,
#   "frames": [
#     {
#       "frame_index": int,
#       "status": "detected"|"no_detection"|"reused"|"skipped",
#       "detections": [
#         {
#           "class": str,
#           "confidence": float,
#           "bbox": [x1, y1, x2, y2],
#           "width": float,
#           "height": float
#         }
#       ]
#     }
#   ]
# }
#
# dtw_alignment.json:
# {
#   "dtw_distance": float,
#   "normalized_distance": float,
#   "num_matches": int,
#   "expert_valid_frame_count": int,
#   "learner_valid_frame_count": int,
#   "matches": [
#     {
#       "dtw_step": int,
#       "expert_seq_index": int,
#       "learner_seq_index": int,
#       "expert_frame_index": int,
#       "learner_frame_index": int,
#       "expert_timestamp_sec": float,
#       "learner_timestamp_sec": float,
#       "expert_angle": float,
#       "learner_angle": float,
#       "angle_difference": float
#     }
#   ]
# }
#
# learner_angle_comparison.json:
# {
#   "video_type": "learner_comparison",
#   "reference_mode": "dynamic_dtw_expert_angle",
#   "frames": [
#     {
#       "frame_index": int,
#       "timestamp_sec": float,
#       "learner_detected": bool,
#       "learner_valid_line": bool,
#       "learner_angle": float | null,
#       "learner_line_center": {"x","y"} | null,
#       "matched_expert_frame_index": int | null,
#       "matched_expert_timestamp_sec": float | null,
#       "matched_expert_angle": float | null,
#       "match_source": "dtw_direct"|"nearest_previous"|"nearest_next"|null,
#       "angle_difference": float | null,
#       "error_level": "ok"|"medium"|"high"|"unknown"
#     }
#   ]
# }
#
# run_summary.json:
# {
#   "clip_id": str,
#   "expert_name": str,
#   "expert_total_frames": int,
#   "learner_total_frames": int,
#   "expert_valid_lines": int,
#   "learner_valid_lines": int,
#   "expert_valid_ratio": float,
#   "learner_valid_ratio": float,
#   "dtw_distance": float,
#   "normalized_dtw_distance": float,
#   "num_dtw_matches": int,
#   "mean_angle_difference": float | null,
#   "median_angle_difference": float | null,
#   "max_angle_difference": float | null,
#   "high_error_frame_count": int,
#   "medium_error_frame_count": int,
#   "ok_frame_count": int
# }
