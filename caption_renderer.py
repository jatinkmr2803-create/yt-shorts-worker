"""
Caption Renderer
────────────────
Generates styled ASS (Advanced SubStation Alpha) subtitle files
from word-level timestamps for burnt-in captions on vertical shorts.

Caption Style: Bold yellow/white text with black stroke, positioned
in the lower third of a 1080×1920 frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class CaptionStyle:
    """Configurable caption styling preset."""
    name: str
    font_name: str = "Arial"
    font_size: int = 58
    primary_color: str = "&H0000FFFF"     # Yellow (ASS BGR format)
    secondary_color: str = "&H00FFFFFF"   # White
    outline_color: str = "&H00000000"     # Black
    back_color: str = "&H80000000"        # Semi-transparent black
    bold: bool = True
    outline_width: int = 3
    shadow_depth: int = 1
    alignment: int = 2                     # Bottom-center
    margin_v: int = 140                    # Distance from bottom
    margin_l: int = 60
    margin_r: int = 60


# ── Preset Styles ─────────────────────────────────────────────

CAPTION_PRESETS: dict[str, CaptionStyle] = {
    "yellow_stroke": CaptionStyle(
        name="yellow_stroke",
        primary_color="&H0000FFFF",       # Bright yellow
        outline_color="&H00000000",       # Black stroke
        font_size=58,
        outline_width=3,
    ),
    "white_glow": CaptionStyle(
        name="white_glow",
        primary_color="&H00FFFFFF",       # White
        outline_color="&H00000000",       # Black stroke
        font_size=54,
        outline_width=4,
        shadow_depth=2,
    ),
    "highlighter": CaptionStyle(
        name="highlighter",
        primary_color="&H00FFFFFF",       # White text
        outline_color="&H000055FF",       # Orange-red outline
        back_color="&H800055FF",          # Semi-transparent orange bg
        font_size=60,
        outline_width=0,
        shadow_depth=0,
    ),
    "tiktok_bold": CaptionStyle(
        name="tiktok_bold",
        font_name="Impact",
        primary_color="&H00FFFFFF",       # White
        outline_color="&H00000000",       # Black
        font_size=62,
        outline_width=4,
        shadow_depth=0,
    ),
    "neon_green": CaptionStyle(
        name="neon_green",
        primary_color="&H0000FF00",       # Neon Green
        outline_color="&H00000000",       # Black stroke
        font_size=58,
        outline_width=4,
        shadow_depth=1,
    ),
    "hot_pink": CaptionStyle(
        name="hot_pink",
        primary_color="&H00FF00FF",       # Magenta / Hot Pink
        outline_color="&H00000000",       # Black stroke
        font_size=58,
        outline_width=4,
        shadow_depth=1,
    ),
}


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format: H:MM:SS.CC"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _build_ass_header(style: CaptionStyle, video_width: int = 1080, video_height: int = 1920) -> str:
    """Build the ASS file header with script info and style definitions."""
    bold_flag = -1 if style.bold else 0

    return f"""[Script Info]
Title: Viral Shorts Captions
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font_name},{style.font_size},{style.primary_color},{style.secondary_color},{style.outline_color},{style.back_color},{bold_flag},0,0,0,100,100,0,0,1,{style.outline_width},{style.shadow_depth},{style.alignment},{style.margin_l},{style.margin_r},{style.margin_v},1
Style: Highlight,{style.font_name},{int(style.font_size * 1.15)},{style.secondary_color},{style.primary_color},{style.outline_color},{style.back_color},{bold_flag},0,0,0,105,105,0,0,1,{style.outline_width + 1},{style.shadow_depth},{style.alignment},{style.margin_l},{style.margin_r},{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def generate_word_by_word_captions(
    words: list[dict],
    segment_start: float,
    segment_end: float,
    output_path: str,
    style_preset: str = "yellow_stroke",
    words_per_group: int = 3,
    highlight_current: bool = True,
) -> str:
    """
    Generate an ASS subtitle file with word-by-word captions.

    Words are grouped (e.g., 3 at a time) and displayed with the current
    word highlighted in a larger/different color for a dynamic viral look.

    Args:
        words: Word-level timestamps from Whisper [{'word': str, 'start': float, 'end': float}]
        segment_start: Start time of the segment in the source video
        segment_end: End time of the segment in the source video
        output_path: Where to write the .ass file
        style_preset: Caption style preset name
        words_per_group: Number of words shown at once
        highlight_current: Whether to highlight the currently spoken word

    Returns:
        Path to the generated .ass file
    """
    style = CAPTION_PRESETS.get(style_preset, CAPTION_PRESETS["yellow_stroke"])

    # Filter words within our segment
    segment_words = [
        w for w in words
        if w.get("start", 0) >= segment_start and w.get("end", 0) <= segment_end
    ]

    if not segment_words:
        # Fallback: if no word-level timestamps, create a single caption
        segment_words = [{"word": "...", "start": segment_start, "end": segment_end}]

    # Offset timestamps so segment starts at 0:00
    for w in segment_words:
        w["adj_start"] = w.get("start", segment_start) - segment_start
        w["adj_end"] = w.get("end", segment_end) - segment_start

    # Build ASS content
    ass_content = _build_ass_header(style)

    # Group words and create events
    events = []
    for i in range(0, len(segment_words), words_per_group):
        group = segment_words[i : i + words_per_group]

        group_start = group[0]["adj_start"]
        group_end = group[-1]["adj_end"]

        # Add a small padding to prevent gaps between groups
        group_end = min(group_end + 0.05, segment_end - segment_start)

        if highlight_current:
            # Show all words but highlight each one as it's spoken
            for j, word in enumerate(group):
                word_start = word["adj_start"]
                word_end = word["adj_end"]

                # Build the line with the current word highlighted
                parts = []
                for k, w in enumerate(group):
                    if k == j:
                        # Highlight current word: larger + different color
                        parts.append(
                            r"{\rHighlight}" + w["word"].strip() + r"{\rDefault}"
                        )
                    else:
                        parts.append(w["word"].strip())

                line_text = " ".join(parts)

                start_ts = _format_ass_time(word_start)
                end_ts = _format_ass_time(word_end)

                events.append(
                    f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{line_text}"
                )
        else:
            # Simple group display — all words appear at once
            line_text = " ".join(w["word"].strip() for w in group)
            start_ts = _format_ass_time(group_start)
            end_ts = _format_ass_time(group_end)

            events.append(
                f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{line_text}"
            )

    ass_content += "\n".join(events) + "\n"

    # Write to file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    return output_path


def generate_segment_captions(
    segments: list[dict],
    segment_start: float,
    segment_end: float,
    output_path: str,
    style_preset: str = "yellow_stroke",
) -> str:
    """
    Fallback: Generate ASS captions from segment-level (sentence) timestamps
    when word-level timestamps are unavailable.

    Args:
        segments: Segment-level timestamps from Whisper
        segment_start: Start time in source video
        segment_end: End time in source video
        output_path: Where to write the .ass file
        style_preset: Caption style preset name

    Returns:
        Path to the generated .ass file
    """
    style = CAPTION_PRESETS.get(style_preset, CAPTION_PRESETS["yellow_stroke"])

    clip_segments = [
        s for s in segments
        if s.get("end", 0) > segment_start and s.get("start", 0) < segment_end
    ]

    ass_content = _build_ass_header(style)
    events = []

    for seg in clip_segments:
        adj_start = max(seg["start"] - segment_start, 0)
        adj_end = min(seg["end"] - segment_start, segment_end - segment_start)
        text = seg.get("text", "").strip()

        if not text:
            continue

        # Split long sentences into shorter lines (max ~6 words per line)
        words = text.split()
        lines = []
        for k in range(0, len(words), 6):
            lines.append(" ".join(words[k : k + 6]))
        formatted_text = r"\N".join(lines)

        start_ts = _format_ass_time(adj_start)
        end_ts = _format_ass_time(adj_end)

        events.append(
            f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{formatted_text}"
        )

    ass_content += "\n".join(events) + "\n"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    return output_path
