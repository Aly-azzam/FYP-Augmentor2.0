"""Constants for the Motion Representation Engine."""

# Canonical coordinate convention: normalized MediaPipe-style
COORDINATE_SYSTEM = "normalized"

TIME_UNIT = "seconds"
ANGLE_UNIT = "degrees"

# Step 1 sequence-building defaults
DEFAULT_TRACKED_LANDMARKS = [
    "wrist",
    "index_tip",
    "thumb_tip",
]
DEFAULT_COORD_DIM = 3
DEFAULT_VISIBILITY_THRESHOLD = 0.5

# Smoothing / interpolation
MAX_INTERPOLATION_GAP_FRAMES = 5
VELOCITY_SMOOTHING_WINDOW = 3
