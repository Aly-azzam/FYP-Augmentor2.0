"""Perception pipeline test placeholders.

These tests are intentionally minimal until the perception pipeline has real
video processing and landmark extraction logic.
"""

import unittest

from app.services.perception_service import run_perception_pipeline


class TestPerceptionPipeline(unittest.TestCase):
    """Basic placeholder tests for the Perception Engine module."""

    def test_run_perception_pipeline_placeholder(self) -> None:
        """Ensure the stub pipeline entry point returns an output contract."""
        output = run_perception_pipeline(video_path="dummy.mp4", video_id="vid-1")
        self.assertEqual(output.video_id, "vid-1")
        self.assertEqual(output.total_frames, 0)
        self.assertEqual(output.frames, [])


if __name__ == "__main__":
    unittest.main()

