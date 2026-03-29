"""
AceFlow v1.0
Built on top of Ace-Step v1.5

Copyright (C) 2026 Marco Robustini [Marcopter]

This file is part of AceFlow.
AceFlow is licensed under the GNU General Public License v3.0 or later.

You may redistribute and/or modify this software under the terms
of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or any later version.

AceFlow is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .chord_synth import synthesize_reference_wav_bytes


def render_reference_wav_file(
    chords: list[str],
    output_path: str | Path,
    bpm: float = 120.0,
    beats_per_chord: int = 4,
    target_duration_sec: Optional[float] = None,
    renderer_preference: str = "soundfont",
) -> dict:
    """Render a chord-reference WAV to disk and return the render metadata."""
    wav_bytes, meta = synthesize_reference_wav_bytes(
        chords=chords,
        bpm=bpm,
        beats_per_chord=beats_per_chord,
        target_duration_sec=target_duration_sec,
        renderer_preference=renderer_preference,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(wav_bytes)
    meta['output_path'] = str(out)
    meta['size_bytes'] = out.stat().st_size
    return meta
