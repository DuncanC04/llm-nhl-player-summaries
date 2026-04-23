#!/usr/bin/env python3
"""
Player Statistics Summary Generator — entry point.

Implementation lives in llm_training/player_summary/ (data, presets, training, inference).
"""

import io
import os
import sys
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Reconfigure stdout/stderr to UTF-8 on Windows consoles (default is cp1252).
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name)
    if hasattr(_stream, "buffer") and getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        setattr(sys, _stream_name, io.TextIOWrapper(_stream.buffer, encoding="utf-8", errors="replace"))

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_training.player_summary.cli import main

if __name__ == "__main__":
    main()
