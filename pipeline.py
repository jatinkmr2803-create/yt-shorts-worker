"""
Video Processing Pipeline (Optimized)
──────────────────────────────────────
Complete pipeline: Download → Transcribe → Detect Viral Segments → Render Shorts

This is the core processing engine that transforms a YouTube URL into
multiple 1080×1920 vertical shorts with burnt-in styled captions.

Performance optimizations:
  • Downloads at 1080p (for high quality vertical crops)
  • Extracts audio as OGG Opus (~0.12 MB/min vs 1.88 MB/min WAV)
  • Transcribes audio chunks in parallel via ThreadPoolExecutor
  • Renders all shorts in parallel via ThreadPoolExecutor
  • Uses FFmpeg 'ultrafast' preset for 5–8× faster encoding
  • Caches ffprobe results to avoid repeated calls
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import yt_dlp
from groq import Groq

try:
    from worker.caption_renderer import (
        generate_segment_captions,
        generate_word_by_word_captions,
    )
    from worker.viral_detector import ScoredSegment, detect_viral_segments
except ImportError:
    from caption_renderer import (
        generate_segment_captions,
        generate_word_by_word_captions,
    )
    from viral_detector import ScoredSegment, detect_viral_segments

# ── Configuration ─────────────────────────────────────────────

logger = logging.getLogger("pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
MAX_AUDIO_CHUNK_MB = 24  # Groq free tier limit is 25 MB
MAX_SHORTS_PER_JOB = int(os.getenv("MAX_SHORTS_PER_JOB", "5"))
MIN_SHORT_DURATION = float(os.getenv("MIN_SHORT_DURATION", "15"))
MAX_SHORT_DURATION = float(os.getenv("MAX_SHORT_DURATION", "60"))

# Vertical short dimensions
SHORT_WIDTH = 1080
SHORT_HEIGHT = 1920

# Parallel workers — capped to avoid overwhelming CPU / API rate limits
MAX_RENDER_WORKERS = int(os.getenv("MAX_RENDER_WORKERS", "4"))
MAX_TRANSCRIBE_WORKERS = int(os.getenv("MAX_TRANSCRIBE_WORKERS", "3"))


# ═══════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD VIDEO
# ═══════════════════════════════════════════════════════════════

def download_video(youtube_url: str, output_dir: str) -> dict[str, Any]:
    """
    Download a YouTube video at an enforced 480p quality level using yt-dlp.

    480p provides a strong balance of reasonable fidelity while guaranteeing extremely fast downloads.

    Returns:
        Dict with 'video_path', 'title', 'duration', and 'metadata'.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=480]+bestaudio/best[height<=480]/best",
        "extractor_args": {"youtube": ["player_client=android,ios"]},
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "writeinfojson": False,
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }

    # IMPORTANT: Check if user uploaded a cookies file specifically to bypass bot protection!
    for cookie_file in ["cookies.txt", "www.youtube.com_cookies.txt", "www.youtube.com_cookies"]:
        if os.path.exists(cookie_file):
            ydl_opts["cookiefile"] = cookie_file
            logger.info(f"🍪 Using provided {cookie_file} for bot protection bypass!")
            break

    logger.info(f"⬇ Downloading (480p): {youtube_url}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)

        if info is None:
            raise RuntimeError(f"Failed to extract info for {youtube_url}")

        video_id = info.get("id", "unknown")
        video_path = os.path.join(output_dir, f"{video_id}.mp4")

        # yt-dlp may save with different extension, find the actual file
        if not os.path.exists(video_path):
            for ext in ["mp4", "mkv", "webm"]:
                candidate = os.path.join(output_dir, f"{video_id}.{ext}")
                if os.path.exists(candidate):
                    video_path = candidate
                    break

        result = {
            "video_path": video_path,
            "title": info.get("title", "Untitled"),
            "duration": info.get("duration", 0),
            "metadata": {
                "uploader": info.get("uploader", ""),
                "view_count": info.get("view_count", 0),
                "like_count": info.get("like_count", 0),
                "description": (info.get("description", "") or "")[:500],
            },
        }

        logger.info(
            f"✓ Downloaded: {result['title']} "
            f"({result['duration']:.0f}s) → {video_path}"
        )
        return result


# ═══════════════════════════════════════════════════════════════
# STEP 2: EXTRACT & CHUNK AUDIO  (OGG Opus — ~200× smaller)
# ═══════════════════════════════════════════════════════════════

def extract_audio(video_path: str, output_dir: str | None = None) -> list[str]:
    """
    Extract audio as OGG Opus (mono, 16kbps) — dramatically smaller than WAV.

    WAV 16kHz mono = ~1.88 MB/min  →  ~13 min fills 25 MB
    OGG Opus 16kbps = ~0.12 MB/min →  ~200 min fits in 25 MB

    For most YouTube videos (<3 hrs), the entire audio fits in one file,
    eliminating the need for chunking entirely.

    Returns:
        List of audio chunk file paths.
    """
    if output_dir is None:
        output_dir = os.path.dirname(video_path)

    base_name = Path(video_path).stem
    full_audio_path = os.path.join(output_dir, f"{base_name}_audio.ogg")

    # Extract audio as Opus in OGG container — tiny file, Whisper supports it
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                     # No video
        "-acodec", "libopus",      # Opus codec
        "-b:a", "16k",             # 16kbps — minimal but fine for speech
        "-ar", "16000",            # 16kHz sample rate (Whisper optimal)
        "-ac", "1",                # Mono
        "-application", "voip",    # Optimize for speech
        full_audio_path,
    ]

    logger.info("🔊 Extracting audio (Opus)...")
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr[:500]}")

    # Check file size
    file_size_mb = os.path.getsize(full_audio_path) / (1024 * 1024)
    logger.info(f"✓ Audio extracted: {file_size_mb:.1f} MB (Opus)")

    if file_size_mb <= MAX_AUDIO_CHUNK_MB:
        return [full_audio_path]

    # ── Chunk the audio (rare — only for videos > ~3 hours) ──
    # Opus at 16kbps ≈ 0.12 MB/min
    duration_per_mb = 60 / 0.12  # ~500 seconds per MB
    chunk_duration = int(MAX_AUDIO_CHUNK_MB * duration_per_mb)

    # Get total duration
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        full_audio_path,
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    total_duration = float(probe_result.stdout.strip())

    num_chunks = math.ceil(total_duration / chunk_duration)
    chunk_paths = []

    logger.info(
        f"✂ Splitting audio into {num_chunks} chunks "
        f"({chunk_duration}s each)"
    )

    for i in range(num_chunks):
        start = i * chunk_duration
        chunk_path = os.path.join(output_dir, f"{base_name}_chunk_{i:03d}.ogg")

        chunk_cmd = [
            "ffmpeg", "-y",
            "-i", full_audio_path,
            "-ss", str(start),
            "-t", str(chunk_duration),
            "-acodec", "libopus",
            "-b:a", "16k",
            "-ar", "16000",
            "-ac", "1",
            "-application", "voip",
            chunk_path,
        ]
        subprocess.run(chunk_cmd, capture_output=True, text=True, timeout=120)
        chunk_paths.append(chunk_path)

    # Clean up full audio file
    os.remove(full_audio_path)

    return chunk_paths


# ═══════════════════════════════════════════════════════════════
# STEP 3: TRANSCRIBE WITH GROQ WHISPER  (Parallel)
# ═══════════════════════════════════════════════════════════════

def _transcribe_single_chunk(
    audio_path: str,
    chunk_index: int,
    total_chunks: int,
    offset: float,
) -> dict[str, Any]:
    """Transcribe one audio chunk via Groq API. Thread-safe."""
    client = Groq(api_key=GROQ_API_KEY)

    logger.info(
        f"🎤 Transcribing chunk {chunk_index + 1}/{total_chunks} "
        f"(offset: {offset:.1f}s)..."
    )

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), audio_file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language="en",
        )

    response_data = response

    # Extract segments with offset adjustment
    segments = []
    if hasattr(response_data, "segments") and response_data.segments:
        for seg in response_data.segments:
            seg_dict = {
                "start": (seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0)) + offset,
                "end": (seg.get("end", 0) if isinstance(seg, dict) else getattr(seg, "end", 0)) + offset,
                "text": seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", ""),
            }
            segments.append(seg_dict)

    # Extract words with offset adjustment
    words = []
    if hasattr(response_data, "words") and response_data.words:
        for w in response_data.words:
            word_dict = {
                "start": (w.get("start", 0) if isinstance(w, dict) else getattr(w, "start", 0)) + offset,
                "end": (w.get("end", 0) if isinstance(w, dict) else getattr(w, "end", 0)) + offset,
                "word": w.get("word", "") if isinstance(w, dict) else getattr(w, "word", ""),
            }
            words.append(word_dict)

    text = getattr(response_data, "text", "") or ""

    return {
        "index": chunk_index,
        "text": text,
        "segments": segments,
        "words": words,
    }


def transcribe_audio(
    audio_paths: list[str],
    time_offsets: list[float] | None = None,
) -> dict[str, Any]:
    """
    Transcribe audio using Groq's Whisper-large-v3 API.

    For multiple chunks, transcription runs in parallel using ThreadPoolExecutor
    (each chunk is an independent API call).

    Args:
        audio_paths: List of audio file paths (may be chunked).
        time_offsets: Cumulative time offset for each chunk (auto-calculated if None).

    Returns:
        Dict with 'text', 'segments', and 'words' (all timestamps adjusted).
    """
    # Pre-calculate offsets from audio durations if not provided
    if time_offsets is None:
        time_offsets = [0.0]
        for i, path in enumerate(audio_paths[:-1]):
            # Get duration via ffprobe
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            dur = float(probe_result.stdout.strip())
            time_offsets.append(time_offsets[-1] + dur)

    total = len(audio_paths)

    # Single chunk? Skip thread pool overhead
    if total == 1:
        result = _transcribe_single_chunk(audio_paths[0], 0, 1, time_offsets[0])
        logger.info(
            f"✓ Transcription complete: "
            f"{len(result['segments'])} segments, {len(result['words'])} words"
        )
        return {
            "text": result["text"],
            "segments": result["segments"],
            "words": result["words"],
        }

    # Multiple chunks → parallel transcription
    logger.info(f"⚡ Transcribing {total} chunks in parallel...")

    chunk_results = [None] * total

    with ThreadPoolExecutor(max_workers=min(MAX_TRANSCRIBE_WORKERS, total)) as executor:
        futures = {}
        for i, (path, offset) in enumerate(zip(audio_paths, time_offsets)):
            future = executor.submit(
                _transcribe_single_chunk, path, i, total, offset
            )
            futures[future] = i

        for future in as_completed(futures):
            result = future.result()
            chunk_results[result["index"]] = result

    # Merge results in order
    all_segments = []
    all_words = []
    full_text_parts = []

    for r in chunk_results:
        all_segments.extend(r["segments"])
        all_words.extend(r["words"])
        full_text_parts.append(r["text"])

    logger.info(
        f"✓ Transcription complete: "
        f"{len(all_segments)} segments, {len(all_words)} words"
    )

    return {
        "text": " ".join(full_text_parts),
        "segments": all_segments,
        "words": all_words,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 4: RENDER VERTICAL SHORTS  (Parallel + ultrafast)
# ═══════════════════════════════════════════════════════════════

def _get_video_dimensions(video_path: str) -> tuple[int, int]:
    """Get video width and height using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"])


# Module-level cache to avoid repeated ffprobe calls for the same video
_dimensions_cache: dict[str, tuple[int, int]] = {}


def _get_video_dimensions_cached(video_path: str) -> tuple[int, int]:
    """Cached wrapper — avoids calling ffprobe for the same file repeatedly."""
    if video_path not in _dimensions_cache:
        _dimensions_cache[video_path] = _get_video_dimensions(video_path)
    return _dimensions_cache[video_path]


def _build_smart_crop_filter(src_width: int, src_height: int) -> str:
    """
    Build FFmpeg video filter for smart center-crop to 1080×1920.

    Strategy:
    - If landscape: scale height to 1920, crop center width to 1080
    - If portrait/square: scale width to 1080, pad/crop height to 1920
    - Maintains aspect ratio, no stretching
    """
    src_aspect = src_width / src_height
    target_aspect = SHORT_WIDTH / SHORT_HEIGHT  # 0.5625

    if src_aspect > target_aspect:
        # Source is wider than target → scale by height, crop width
        # Scale so height = 1920, then crop center 1080px
        return (
            f"scale=-2:{SHORT_HEIGHT},"
            f"crop={SHORT_WIDTH}:{SHORT_HEIGHT}:(in_w-{SHORT_WIDTH})/2:0,"
            f"setsar=1"
        )
    elif src_aspect < target_aspect:
        # Source is narrower than target → scale by width, pad height
        return (
            f"scale={SHORT_WIDTH}:-2,"
            f"pad={SHORT_WIDTH}:{SHORT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1"
        )
    else:
        # Exact match — just scale
        return f"scale={SHORT_WIDTH}:{SHORT_HEIGHT},setsar=1"


def render_short(
    video_path: str,
    segment: ScoredSegment,
    words: list[dict],
    transcript_segments: list[dict],
    output_path: str,
    caption_style: str = "yellow_stroke",
) -> dict[str, Any]:
    """
    Render a single vertical short with burnt-in captions.

    Uses 'ultrafast' x264 preset for 5–8× faster encoding vs 'medium'.
    Quality difference is minimal for short-form social content.

    Args:
        video_path: Path to the source video file
        segment: ScoredSegment with start/end times
        words: All word-level timestamps from transcription
        transcript_segments: All segment-level timestamps
        output_path: Where to save the rendered short
        caption_style: Caption styling preset

    Returns:
        Dict with output metadata (path, duration, file_size).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Get source video dimensions (cached)
    src_width, src_height = _get_video_dimensions_cached(video_path)

    # Build crop/scale filter
    crop_filter = _build_smart_crop_filter(src_width, src_height)

    # Generate caption file
    ass_path = output_path.replace(".mp4", ".ass")

    if words:
        generate_word_by_word_captions(
            words=words,
            segment_start=segment.start_time,
            segment_end=segment.end_time,
            output_path=ass_path,
            style_preset=caption_style,
            words_per_group=3,
            highlight_current=True,
        )
    else:
        # Fallback to segment-level captions
        generate_segment_captions(
            segments=transcript_segments,
            segment_start=segment.start_time,
            segment_end=segment.end_time,
            output_path=ass_path,
            style_preset=caption_style,
        )

    # ── FFmpeg render command ──
    # Key speed optimizations:
    #   • -ss before -i = fast seek (no decode before start)
    #   • -preset ultrafast = 5–8× faster encoding
    #   • -tune fastdecode = optimized for quick decode
    #   • -threads 0 = auto-detect optimal thread count

    # Escape the ASS path for FFmpeg filter (handle Windows backslashes)
    ass_path_escaped = ass_path.replace("\\", "/").replace(":", "\\\\:")

    video_filter = (
        f"{crop_filter},"
        f"ass='{ass_path_escaped}'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(segment.start_time),
        "-to", str(segment.end_time),
        "-i", video_path,
        "-vf", video_filter,
        "-c:v", "libx264",
        "-preset", "ultrafast",        # ← 5–8× faster than 'medium'
        "-tune", "fastdecode",         # ← optimize for speed
        "-crf", "26",                  # ← slightly lower quality = faster
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-movflags", "+faststart",     # Enable streaming
        "-r", "30",                    # 30fps for shorts
        "-threads", "0",              # Auto-detect threads
        output_path,
    ]

    logger.info(
        f"🎬 Rendering short: {segment.start_time:.1f}s – "
        f"{segment.end_time:.1f}s (score: {segment.virality_score:.3f})"
    )

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg render failed: {result.stderr[:500]}"
        )

    # Get output file info
    file_size = os.path.getsize(output_path)

    # Clean up ASS file
    if os.path.exists(ass_path):
        os.remove(ass_path)

    output_info = {
        "path": output_path,
        "duration": segment.duration,
        "file_size_bytes": file_size,
        "width": SHORT_WIDTH,
        "height": SHORT_HEIGHT,
        "caption_style": caption_style,
        "virality_score": segment.virality_score,
    }

    logger.info(
        f"✓ Short rendered: {output_path} "
        f"({file_size / (1024 * 1024):.1f} MB)"
    )
    return output_info


def _render_short_task(
    video_path: str,
    segment: ScoredSegment,
    words: list[dict],
    transcript_segments: list[dict],
    output_path: str,
    caption_style: str,
) -> tuple[ScoredSegment, dict[str, Any] | None, str | None]:
    """
    Wrapper for parallel rendering. Returns (segment, result_or_None, error_or_None).
    """
    try:
        result = render_short(
            video_path=video_path,
            segment=segment,
            words=words,
            transcript_segments=transcript_segments,
            output_path=output_path,
            caption_style=caption_style,
        )
        return segment, result, None
    except Exception as e:
        logger.error(f"Failed to render short #{segment.rank}: {e}")
        return segment, None, str(e)


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATOR: Full Pipeline
# ═══════════════════════════════════════════════════════════════

def process_job(
    job_id: str,
    youtube_url: str,
    output_dir: str | None = None,
    supabase_client: Any = None,
    max_shorts: int | None = None,
    caption_style: str = "yellow_stroke",
    progress_callback: Any = None,
) -> dict[str, Any]:
    """
    Full pipeline orchestrator: Download → Transcribe → Detect → Render.

    All I/O-heavy steps (transcription, rendering) are parallelized.

    Args:
        job_id: Unique job identifier (from Supabase).
        youtube_url: YouTube video URL to process.
        output_dir: Directory for intermediate and output files.
        supabase_client: Supabase client for status updates (optional).
        max_shorts: Maximum number of shorts to generate.
        caption_style: Caption styling preset.
        progress_callback: Optional callable(**kwargs) for real-time progress updates.

    Returns:
        Dict with job results including paths to all generated shorts.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, job_id)
    if max_shorts is None:
        max_shorts = MAX_SHORTS_PER_JOB

    os.makedirs(output_dir, exist_ok=True)

    def _emit(**kwargs):
        """Emit progress to both Supabase and the in-memory callback."""
        if progress_callback:
            try:
                progress_callback(**kwargs)
            except Exception:
                pass

    def _update_status(status: str, **extra):
        """Update job status in Supabase (if client provided)."""
        if supabase_client:
            try:
                data = {"status": status, **extra}
                supabase_client.table("jobs").update(data).eq("id", job_id).execute()
            except Exception as e:
                logger.warning(f"Failed to update job status: {e}")

    try:
        _update_status("processing")

        # ── Step 1: Download ──
        logger.info(f"{'═' * 60}")
        logger.info(f"  STEP 1/4: Downloading video (480p)")
        logger.info(f"{'═' * 60}")

        _emit(
            status="processing",
            step="downloading",
            step_label="Downloading Video",
            step_detail="Fetching video at 480p...",
            progress=5,
        )

        download_result = download_video(youtube_url, output_dir)
        video_path = download_result["video_path"]

        _update_status("processing", video_title=download_result["title"],
                       video_duration=download_result["duration"])
        _emit(
            step="downloading",
            step_label="Downloading Video",
            step_detail=f"Downloaded: {download_result['title'][:50]}",
            progress=20,
            video_title=download_result["title"],
            video_duration=download_result["duration"],
        )

        # ── Step 2: Extract audio ──
        logger.info(f"{'═' * 60}")
        logger.info(f"  STEP 2/4: Extracting audio (Opus) & transcribing")
        logger.info(f"{'═' * 60}")

        _emit(
            step="extracting",
            step_label="Extracting Audio",
            step_detail="Converting audio to Opus format...",
            progress=25,
        )

        audio_chunks = extract_audio(video_path, output_dir)

        _emit(
            step="extracting",
            step_label="Extracting Audio",
            step_detail=f"Audio extracted ({len(audio_chunks)} chunk{'s' if len(audio_chunks) > 1 else ''})",
            progress=30,
        )

        # ── Step 2b: Transcribe ──
        _emit(
            step="transcribing",
            step_label="Transcribing Audio",
            step_detail="Sending to Groq Whisper API...",
            progress=35,
        )

        transcript = transcribe_audio(audio_chunks)

        # Clean up audio chunks
        for chunk in audio_chunks:
            if os.path.exists(chunk):
                os.remove(chunk)

        _emit(
            step="transcribing",
            step_label="Transcribing Audio",
            step_detail=f"Transcribed {len(transcript['words'])} words",
            progress=55,
        )

        # ── Step 3: Detect viral segments ──
        logger.info(f"{'═' * 60}")
        logger.info(f"  STEP 3/4: Detecting viral segments")
        logger.info(f"{'═' * 60}")

        _emit(
            step="detecting",
            step_label="Detecting Viral Segments",
            step_detail="Analyzing keywords, sentiment & engagement...",
            progress=60,
        )

        viral_segments = detect_viral_segments(
            segments=transcript["segments"],
            words=transcript["words"],
            min_duration=MIN_SHORT_DURATION,
            max_duration=MAX_SHORT_DURATION,
            max_segments=max_shorts,
        )

        if not viral_segments:
            logger.warning("⚠ No viral segments detected. Creating fallback segment.")
            fallback_end = min(30.0, download_result["duration"])
            viral_segments = [
                ScoredSegment(
                    start_time=0.0,
                    end_time=fallback_end,
                    text=transcript["text"][:200],
                    virality_score=0.1,
                    rank=1,
                )
            ]

        _update_status("processing", segments_found=len(viral_segments))
        _emit(
            step="detecting",
            step_label="Detecting Viral Segments",
            step_detail=f"Found {len(viral_segments)} viral segment{'s' if len(viral_segments) != 1 else ''}",
            progress=70,
            segments_found=len(viral_segments),
        )

        logger.info(f"🎯 Found {len(viral_segments)} viral segments:")
        for seg in viral_segments:
            logger.info(
                f"   #{seg.rank} [{seg.start_time:.1f}s – {seg.end_time:.1f}s] "
                f"score={seg.virality_score:.3f} | "
                f"keywords={seg.keyword_hits[:5]}"
            )

        # ── Step 4: Render shorts (PARALLEL) ──
        logger.info(f"{'═' * 60}")
        logger.info(f"  STEP 4/4: Rendering {len(viral_segments)} shorts in parallel")
        logger.info(f"{'═' * 60}")

        _emit(
            step="rendering",
            step_label="Rendering Shorts",
            step_detail=f"Encoding {len(viral_segments)} vertical shorts...",
            progress=72,
        )

        # Pre-cache video dimensions before spawning threads
        _get_video_dimensions_cached(video_path)

        # Build render tasks
        render_tasks = []
        for segment in viral_segments:
            short_filename = f"short_{segment.rank:02d}_{uuid.uuid4().hex[:8]}.mp4"
            short_path = os.path.join(output_dir, short_filename)
            render_tasks.append((segment, short_path, short_filename))

        rendered_shorts = []
        num_workers = min(MAX_RENDER_WORKERS, len(render_tasks))
        total_renders = len(render_tasks)
        completed_renders = 0

        logger.info(f"⚡ Launching {num_workers} render workers...")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for segment, short_path, short_filename in render_tasks:
                future = executor.submit(
                    _render_short_task,
                    video_path=video_path,
                    segment=segment,
                    words=transcript["words"],
                    transcript_segments=transcript["segments"],
                    output_path=short_path,
                    caption_style=caption_style,
                )
                futures[future] = (segment, short_path, short_filename)

            for future in as_completed(futures):
                segment, short_path, short_filename = futures[future]
                seg_obj, short_info, error = future.result()
                completed_renders += 1

                # Update render progress (72% → 95%)
                render_progress = 72 + int((completed_renders / total_renders) * 23)
                _emit(
                    step="rendering",
                    step_label="Rendering Shorts",
                    step_detail=f"Rendered {completed_renders}/{total_renders} shorts",
                    progress=render_progress,
                    shorts_created=completed_renders,
                )

                if short_info is None:
                    continue

                short_info["segment"] = {
                    "start_time": seg_obj.start_time,
                    "end_time": seg_obj.end_time,
                    "text": seg_obj.text[:300],
                    "virality_score": seg_obj.virality_score,
                    "keyword_hits": seg_obj.keyword_hits,
                    "sentiment_data": seg_obj.sentiment_data,
                    "rank": seg_obj.rank,
                }
                rendered_shorts.append(short_info)

                # Upload to Supabase Storage if client is available
                if supabase_client:
                    try:
                        storage_path = f"{job_id}/{short_filename}"
                        with open(short_path, "rb") as f:
                            supabase_client.storage.from_("shorts").upload(
                                storage_path, f,
                                {"content-type": "video/mp4"}
                            )
                        short_info["storage_path"] = storage_path

                        public_url = supabase_client.storage.from_("shorts").get_public_url(storage_path)
                        short_info["public_url"] = public_url

                        supabase_client.table("segments").insert({
                            "job_id": job_id,
                            "start_time": seg_obj.start_time,
                            "end_time": seg_obj.end_time,
                            "transcript_text": seg_obj.text[:2000],
                            "virality_score": seg_obj.virality_score,
                            "keyword_hits": json.dumps(seg_obj.keyword_hits),
                            "sentiment_data": json.dumps(seg_obj.sentiment_data),
                            "rank": seg_obj.rank,
                        }).execute()

                        supabase_client.table("shorts").insert({
                            "job_id": job_id,
                            "segment_id": job_id,
                            "storage_path": storage_path,
                            "public_url": public_url,
                            "file_size_bytes": short_info["file_size_bytes"],
                            "duration": short_info["duration"],
                            "caption_style": caption_style,
                        }).execute()
                    except Exception as e:
                        logger.error(f"Failed to upload short to Supabase: {e}")

        # Sort by rank for consistent output order
        rendered_shorts.sort(key=lambda s: s["segment"]["rank"])

        # ── Finalize ──
        final_result = {
            "job_id": job_id,
            "youtube_url": youtube_url,
            "video_title": download_result["title"],
            "video_duration": download_result["duration"],
            "segments_found": len(viral_segments),
            "shorts_created": len(rendered_shorts),
            "shorts": rendered_shorts,
            "transcript_preview": transcript["text"][:500],
        }

        _update_status(
            "completed",
            shorts_created=len(rendered_shorts),
        )

        logger.info(f"{'═' * 60}")
        logger.info(f"  ✅ JOB COMPLETE: {len(rendered_shorts)} shorts created")
        logger.info(f"{'═' * 60}")

        return final_result

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        _update_status("failed", error_message=str(e)[:1000])
        _emit(
            status="failed",
            step="error",
            step_label="Failed",
            step_detail=str(e)[:200],
        )
        raise


# ═══════════════════════════════════════════════════════════════
# CLI: Run pipeline directly
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <youtube_url> [output_dir]")
        print("Example: python pipeline.py https://youtu.be/dQw4w9WgXcQ ./my_shorts")
        sys.exit(1)

    url = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    job = str(uuid.uuid4())

    print(f"\n🎬 Starting pipeline for: {url}")
    print(f"   Job ID: {job}\n")

    result = process_job(job_id=job, youtube_url=url, output_dir=out_dir)

    print(f"\n{'─' * 60}")
    print(f"📊 Results:")
    print(f"   Title:    {result['video_title']}")
    print(f"   Duration: {result['video_duration']:.0f}s")
    print(f"   Segments: {result['segments_found']}")
    print(f"   Shorts:   {result['shorts_created']}")
    print(f"{'─' * 60}")

    for s in result["shorts"]:
        print(f"   📹 {s['path']}")
        print(f"      Score: {s['virality_score']:.3f} | Duration: {s['duration']:.1f}s")
        print(f"      Keywords: {s['segment']['keyword_hits'][:5]}")
