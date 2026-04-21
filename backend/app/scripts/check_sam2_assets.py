"""Read-only SAM2 asset diagnostic script.

Usage:
    python -m app.scripts.check_sam2_assets
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.sam2_constants import SAM2_DEFAULT_FRAME_STRIDE, get_sam2_asset_paths
from app.services.sam2.sam2_service import SAM2DependencyError, resolve_device_info


def _is_file(path: Path) -> bool:
    return path.expanduser().resolve().is_file()


def main() -> int:
    assets = get_sam2_asset_paths()

    checkpoint_exists = _is_file(assets.checkpoint_path)
    config_exists = _is_file(assets.config_path)

    try:
        device_info = resolve_device_info()
        selected_device = str(device_info["device"])
        gpu_name = device_info["gpu_name"]
    except SAM2DependencyError as exc:
        selected_device = "unavailable"
        gpu_name = None
        print(f"[WARN] Could not resolve SAM2 device: {exc}")

    payload = {
        "resolved_checkpoint_path": str(assets.checkpoint_path),
        "resolved_config_path": str(assets.config_path),
        "checkpoint_exists": checkpoint_exists,
        "config_exists": config_exists,
        "legacy_demo_checkpoint_rel": assets.legacy_checkpoint_rel,
        "legacy_demo_config_rel": assets.legacy_config_rel,
        "selected_device": selected_device,
        "gpu_name": gpu_name,
        "default_frame_stride": SAM2_DEFAULT_FRAME_STRIDE,
    }
    print(json.dumps(payload, indent=2))

    return 0 if checkpoint_exists and config_exists else 1


if __name__ == "__main__":
    raise SystemExit(main())
