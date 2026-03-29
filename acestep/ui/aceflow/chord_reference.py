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

"""Compatibility facade for AceFlow chord-reference helpers.

The original public API remains here for callers such as ``app.py``, while the
implementation is split by responsibility into parsing, voicing, synthesis, and
file-output modules.
"""

from .chord_file import render_reference_wav_file
from .chord_parser import ParsedChord, parse_chord_symbol
from .chord_synth import MAX_RENDER_DURATION_SEC, midi_to_freq, synthesize_reference_wav_bytes
from .chord_voicing import choose_voicing

__all__ = [
    'MAX_RENDER_DURATION_SEC',
    'ParsedChord',
    'choose_voicing',
    'midi_to_freq',
    'parse_chord_symbol',
    'render_reference_wav_file',
    'synthesize_reference_wav_bytes',
]
