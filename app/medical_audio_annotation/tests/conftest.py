"""Pytest configuration and sys.path setup for medical_audio_annotation tests."""

import sys
from pathlib import Path

# Ensure workspace root and package dir are in sys.path
package_dir = Path(__file__).resolve().parent.parent
app_dir = package_dir.parent
workspace_root = app_dir.parent

for path in [workspace_root, app_dir, package_dir]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
