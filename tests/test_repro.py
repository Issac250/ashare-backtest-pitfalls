"""Every chapter's reproduction must run and pass its own assert.

The scripts run one at a time, in a fresh interpreter each, so a failure
points at a single chapter and no two backtests share memory.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPRO = Path(__file__).resolve().parents[1] / "repro"
SCRIPTS = sorted(p for p in REPRO.glob("[0-9][0-9]_*.py"))


def test_twelve_chapters_have_a_repro():
    assert len(SCRIPTS) >= 12


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_repro_runs(script):
    proc = subprocess.run([sys.executable, str(script)], cwd=REPRO, capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, f"{script.name} failed:\n{proc.stdout}\n{proc.stderr}"
    assert "(wrong" in proc.stdout and "(right" in proc.stdout, "print the wrong and the right number"
