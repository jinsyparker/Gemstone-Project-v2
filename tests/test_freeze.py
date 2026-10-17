"""Fails if the frozen split, baseline checkpoints, saved test predictions, benchmark table or recipe change."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import freeze  # noqa: E402


def test_frozen_files_unchanged():
    assert freeze.FREEZE.exists(), "run: python code/freeze.py make"
    assert freeze.check() == []
