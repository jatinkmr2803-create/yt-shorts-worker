"""
FastAPI Worker — YouTube Shorts Generator
──────────────────────────────────────────
Background job poller + API endpoints for the video processing worker.

Features:
  • In-memory job store for standalone mode (no Supabase required)
  • Granular step-by-step progress tracking via progress callback
  • Serves the frontend as static files from /
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from supabase import Client, create_client

try:
    from worker.pipeline import process_job
except ImportError:
    from pipeline import process_job

logger = logging.getLogger("worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))
AUTO_DELETE_DAYS = int(os.getenv("AUTO_DELETE_DAYS", "1"))


# ══════════════════════════════════════════════════════════════
# IN-MEMORY JOB STORE  (standalone mode — no Supabase needed)
# ══════════════════════════════════════════════════════════════

JOB_STORE: dict[str, dict[str, Any]] = {}


def _create_job_entry(job_id: str, youtube_url: str) -> dict:
    """Create a fresh job entry in the in-memory store."""
    entry = {
        "id": job_id,
        "youtube_url": youtube_url,
        "status": "pending",
        "step": "queued",
        "step_label": "Queued",
        "step_detail": "Waiting for worker...",
        "progress": 0,
        "video_title": None,
        "video_duration": None,
        "segments_found": None,
        "shorts_created": None,
        "shorts": [],
        "error_message": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "steps_completed": [],
    }
    JOB_STORE[job_id] = entry
    return entry


def _progress_callback(job_id: str, **kwargs):
    """
    Called by the pipeline at each major step.
    Updates the in-memory job store with granular progress.
    """
    if job_id not in JOB_STORE:
        return
    entry = JOB_STORE[job_id]
    for key, value in kwargs.items():
        if key in entry:
            entry[key] = value
    # Track completed steps
    step = kwargs.get("step")
    if step and step not in [s["id"] for s in entry.get("steps_completed", [])]:
        entry.setdefault("steps_completed", []).append({
            "id": step,
            "label": kwargs.get("step_label", step),
            "time": datetime.now(timezone.utc).isoformat(),
        })


# ── Supabase Client ──────────────────────────────────────────

def get_supabase() -> Client | None:
    """Create Supabase client (returns None if not configured)."""
    if (
        not SUPABASE_URL
        or not SUPABASE_SERVICE_KEY
        or "your-project" in SUPABASE_URL
        or SUPABASE_SERVICE_KEY == "eyJhbGciOiJIUzI1NiIs..."
        or len(SUPABASE_SERVICE_KEY) < 30
    ):
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception:
        return None


# ── Background Job Poller ─────────────────────────────────────

async def poll_for_jobs():
    """
    Continuously poll Supabase for pending jobs and process them.
    Runs as a background task within the FastAPI lifespan.
    """
    supabase = get_supabase()
    if not supabase:
        logger.info("📴 Job polling disabled (no Supabase config — standalone mode)")
        return

    logger.info(f"🔄 Job poller started (interval: {POLL_INTERVAL}s)")

    while True:
        try:
            # Fetch the oldest pending job
            result = (
                supabase.table("jobs")
                .select("*")
                .eq("status", "pending")
                .order("created_at", desc=False)
                .limit(1)
                .execute()
            )

            if result.data:
                job = result.data[0]
                job_id = job["id"]
                youtube_url = job["youtube_url"]

                logger.info(f"📋 Picked up job: {job_id}")
                logger.info(f"   URL: {youtube_url}")

                try:
                    await asyncio.to_thread(
                        process_job,
                        job_id=job_id,
                        youtube_url=youtube_url,
                        supabase_client=supabase,
                    )
                except Exception as e:
                    logger.error(f"❌ Job {job_id} failed: {e}")

        except Exception as e:
            logger.error(f"❌ Poller error: {e}")

        await asyncio.sleep(POLL_INTERVAL)


# ── Auto Cleanup Task ─────────────────────────────────────────

async def auto_cleanup():
    """Periodically deletes old jobs and shorts from Supabase."""
    supabase = get_supabase()
    if not supabase or AUTO_DELETE_DAYS <= 0:
        return

    logger.info(f"🧹 Auto-cleanup enabled ({AUTO_DELETE_DAYS} days)")

    while True:
        try:
            # Check once an hour
            await asyncio.sleep(3600)

            cutoff = datetime.now(timezone.utc) - timedelta(days=AUTO_DELETE_DAYS)

            res = (
                supabase.table("jobs")
                .select("id")
                .lt("created_at", cutoff.isoformat())
                .execute()
            )

            if res.data:
                logger.info(f"🗑️ Found {len(res.data)} old jobs to delete")
                for job in res.data:
                    job_id = job["id"]

                    # Remove from storage
                    s_res = supabase.table("shorts").select("storage_path").eq("job_id", job_id).execute()
                    if s_res.data:
                        paths = [s["storage_path"] for s in s_res.data]
                        try:
                            supabase.storage.from_("shorts").remove(paths)
                        except Exception as storage_err:
                            logger.error(f"Failed to delete storage files for job {job_id}: {storage_err}")

                    # Delete from DB (cascades)
                    supabase.table("jobs").delete().eq("id", job_id).execute()

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# ── FastAPI Lifespan ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background tasks on startup."""
    poller_task = asyncio.create_task(poll_for_jobs())
    cleanup_task = asyncio.create_task(auto_cleanup())
    logger.info("🚀 Worker started")
    yield
    poller_task.cancel()
    cleanup_task.cancel()
    logger.info("👋 Worker shutting down")


# ── FastAPI App ───────────────────────────────────────────────

app = FastAPI(
    title="YouTube Shorts Worker",
    description="Video processing worker for the YouTube → Viral Shorts pipeline",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Models ────────────────────────────────────────────

class ProcessRequest(BaseModel):
    youtube_url: str
    caption_style: str = "yellow_stroke"
    max_shorts: int = 5


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


# ── API Endpoints ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    supabase = get_supabase()
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "supabase_connected": supabase is not None,
        "poll_interval": POLL_INTERVAL,
    }


@app.post("/process", response_model=JobResponse)
async def trigger_processing(request: ProcessRequest):
    """
    Manually trigger video processing.
    Creates a job in Supabase or the in-memory store.
    """
    supabase = get_supabase()
    job_id = str(uuid.uuid4())

    if supabase:
        # Insert job into Supabase (the poller will pick it up)
        try:
            supabase.table("jobs").insert({
                "id": job_id,
                "youtube_url": request.youtube_url,
                "status": "pending",
            }).execute()

            return JobResponse(
                job_id=job_id,
                status="pending",
                message="Job queued for processing",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create job: {str(e)}",
            )
    else:
        # Standalone mode: create in-memory entry + process in background
        _create_job_entry(job_id, request.youtube_url)

        def _run():
            try:
                _progress_callback(
                    job_id,
                    status="processing",
                    step="starting",
                    step_label="Starting",
                    step_detail="Initializing pipeline...",
                    progress=2,
                    started_at=datetime.now(timezone.utc).isoformat(),
                )
                result = process_job(
                    job_id=job_id,
                    youtube_url=request.youtube_url,
                    caption_style=request.caption_style,
                    max_shorts=request.max_shorts,
                    progress_callback=lambda **kw: _progress_callback(job_id, **kw),
                )
                _progress_callback(
                    job_id,
                    status="completed",
                    step="done",
                    step_label="Complete",
                    step_detail=f"{result['shorts_created']} shorts created!",
                    progress=100,
                    shorts=result.get("shorts", []),
                    shorts_created=result.get("shorts_created", 0),
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
            except Exception as e:
                _progress_callback(
                    job_id,
                    status="failed",
                    step="error",
                    step_label="Failed",
                    step_detail=str(e)[:200],
                    error_message=str(e)[:1000],
                )

        asyncio.create_task(asyncio.to_thread(_run))

        return JobResponse(
            job_id=job_id,
            status="processing",
            message="Processing started",
        )


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status and results of a specific job.
    Works in both Supabase and standalone (in-memory) mode.
    """
    # Check in-memory store first (standalone mode)
    if job_id in JOB_STORE:
        return JOB_STORE[job_id]

    # Fallback to Supabase
    supabase = get_supabase()
    if not supabase:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        result = (
            supabase.table("jobs")
            .select("*")
            .eq("id", job_id)
            .single()
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="Job not found")

        job = result.data

        # If completed, also fetch shorts
        shorts = []
        if job["status"] == "completed":
            shorts_result = (
                supabase.table("shorts")
                .select("*")
                .eq("job_id", job_id)
                .execute()
            )
            shorts = shorts_result.data or []

        return {
            **job,
            "shorts": shorts,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve Output Directory (For Previews/Downloads) ───────────

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount(
    "/output",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="output",
)


# ── Serve Frontend ────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.is_dir():
    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="frontend",
    )


# ═══════════════════════════════════════════════════════════════
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ═══════════════════════════════════════════════════════════════
