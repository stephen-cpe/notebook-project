"""Background job workers — thin wrappers that run in daemon threads.

Each worker receives a notebook_id and the Flask app object so it can push
an application context for DB access. Workers handle their own error logging
and status updates.

Media jobs (audio/video) carry a ``generation`` token: the route increments
the notebook's ``audio_generation``/``video_generation`` on every launch and
delete. A worker persists its result only when its generation still matches;
otherwise the result is stale (superseded by a relaunch, or resurrected
after a delete) and is discarded. Each job also gets a unique ``job_id``
so concurrent jobs never share temporary file names.
"""

from __future__ import annotations

import logging
import threading
import uuid

logger = logging.getLogger(__name__)

#: Statuses that mean a media job may still be running.
TRANSIENT_MEDIA_STATUSES = ("queued", "scripting", "synthesizing")


def _new_job_id() -> str:
    """Return a short unique id for isolating a job's temp files."""
    return uuid.uuid4().hex[:8]


def _mark_crashed(
    notebook_id: int,
    app: object,
    kind: str,
    generation: int | None,
) -> None:
    """Mark a crashed media job failed (stale-aware)."""
    try:
        with app.app_context():  # type: ignore[attr-defined]
            from src.extensions import db
            from src.models import Notebook

            nb = db.session.get(Notebook, notebook_id)
            if nb is None:
                return
            current = nb.audio_generation if kind == "audio" else nb.video_generation
            if generation is not None and current != generation:
                logger.info(
                    "%s job crashed for notebook %d but generation %d is stale "
                    "(current %d); leaving state alone.",
                    kind.capitalize(),
                    notebook_id,
                    generation,
                    current,
                )
                return
            if kind == "audio":
                nb.audio_status = "failed"
                nb.audio_error = "Generation crashed unexpectedly."
            else:
                nb.video_status = "failed"
                nb.video_error = "Generation crashed unexpectedly."
            db.session.commit()
    except Exception:
        logger.exception("Failed to set %s status to failed for notebook %d", kind, notebook_id)


def launch_audio_job(
    notebook_id: int,
    app: object,
    topic: str = "",
    speaker_a: str = "Ava",
    speaker_b: str = "Andrew",
    generation: int | None = None,
    job_id: str | None = None,
) -> str:
    """Launch a background thread to generate an Audio Overview.

    Returns the job id. ``generation`` should be the notebook's current
    ``audio_generation`` so a superseded job's result is discarded.
    """

    jid = job_id or _new_job_id()

    def _run() -> None:
        try:
            with app.app_context():  # type: ignore[attr-defined]
                from src.services.audio_service import generate_audio_for_notebook

                result = generate_audio_for_notebook(
                    notebook_id,
                    topic=topic,
                    speaker_a=speaker_a,
                    speaker_b=speaker_b,
                    job_id=jid,
                    generation=generation,
                )
                if result is None:
                    logger.error("Audio job returned None for notebook %d", notebook_id)
                else:
                    logger.info(
                        "Audio job finished: notebook=%d status=%s",
                        notebook_id,
                        result.status,
                    )
        except Exception:
            logger.exception("Audio job crashed for notebook %d", notebook_id)
            _mark_crashed(notebook_id, app, "audio", generation)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jid


def launch_summary_job(notebook_id: int, app: object) -> None:
    """Launch a background thread to regenerate the notebook summary."""

    def _run() -> None:
        try:
            with app.app_context():  # type: ignore[attr-defined]
                from src.extensions import db
                from src.models import Notebook
                from src.services.summary_service import SummaryService

                nb = db.session.get(Notebook, notebook_id)
                if nb is None:
                    logger.error("Summary job: notebook %d not found", notebook_id)
                    return

                svc = SummaryService()
                result = svc.generate_summary(nb)
                if result is None:
                    logger.error("Summary job failed for notebook %d", notebook_id)
                else:
                    logger.info(
                        "Summary job finished: notebook=%d skipped=%s",
                        notebook_id,
                        result.skipped,
                    )
        except Exception:
            logger.exception("Summary job crashed for notebook %d", notebook_id)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def launch_video_job(
    notebook_id: int,
    app: object,
    topic: str = "",
    speaker: str = "Ava",
    generation: int | None = None,
    job_id: str | None = None,
) -> str:
    """Launch a background thread to generate a Video Overview.

    Returns the job id. ``generation`` should be the notebook's current
    ``video_generation`` so a superseded job's result is discarded.
    """

    jid = job_id or _new_job_id()

    def _run() -> None:
        try:
            with app.app_context():  # type: ignore[attr-defined]
                from src.services.video_service import generate_video_for_notebook

                result = generate_video_for_notebook(
                    notebook_id, topic=topic, speaker=speaker, job_id=jid, generation=generation
                )
                if result is None:
                    logger.error("Video job returned None for notebook %d", notebook_id)
                else:
                    logger.info(
                        "Video job finished: notebook=%d status=%s",
                        notebook_id,
                        result.status,
                    )
        except Exception:
            logger.exception("Video job crashed for notebook %d", notebook_id)
            _mark_crashed(notebook_id, app, "video", generation)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jid


def recover_interrupted_media_jobs(app: object) -> int:
    """Mark transient media jobs as failed after a process restart.

    Daemon threads do not survive restarts, so any notebook left in a
    transient (queued/scripting/synthesizing) media status can never
    complete. Reset them to failed with an explanatory error so the UI
    never shows a permanently "processing" state.

    Returns the number of notebooks recovered. Must be called with an app
    context available (it pushes its own).
    """
    with app.app_context():  # type: ignore[attr-defined]
        from src.extensions import db
        from src.models import Notebook

        recovered = 0
        notebooks = (
            db.session.query(Notebook)
            .filter(
                Notebook.audio_status.in_(TRANSIENT_MEDIA_STATUSES)
                | Notebook.video_status.in_(TRANSIENT_MEDIA_STATUSES)
            )
            .all()
        )
        for nb in notebooks:
            if nb.audio_status in TRANSIENT_MEDIA_STATUSES:
                nb.audio_status = "failed"
                nb.audio_error = "Generation was interrupted by a restart; please retry."
                recovered += 1
            if nb.video_status in TRANSIENT_MEDIA_STATUSES:
                nb.video_status = "failed"
                nb.video_error = "Generation was interrupted by a restart; please retry."
                recovered += 1
        if recovered:
            db.session.commit()
            logger.info("Recovered %d interrupted media job(s) after restart", recovered)
        return recovered
