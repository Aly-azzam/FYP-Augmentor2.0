from pathlib import Path

from app.services.sequence_service import SequenceService


def main() -> None:
    service = SequenceService()
    input_path = Path("storage/raw/fake_landmarks.json")

    landmark_data = service.load_landmark_sequence(input_path)
    sequence = service.build_motion_sequence(landmark_data)
    wrist_series = service.extract_landmark_series(sequence, "wrist")

    assert sequence.video_id == "learner_01"
    assert sequence.fps == 30.0
    assert sequence.sequence_length == 4
    assert sequence.tracked_landmarks == ["wrist", "index_tip", "thumb_tip"]
    assert sequence.frames[1].positions["thumb_tip"] == [0.0, 0.0, 0.0]
    assert sequence.frames[2].positions["thumb_tip"] == [0.0, 0.0, 0.0]
    assert len(wrist_series) == 4

    print("SequenceService Step 1 test passed.")
    print(f"video_id={sequence.video_id}")
    print(f"sequence_length={sequence.sequence_length}")
    print(f"tracked_landmarks={sequence.tracked_landmarks}")
    print(f"first_frame_positions={sequence.frames[0].positions}")


if __name__ == "__main__":
    main()
