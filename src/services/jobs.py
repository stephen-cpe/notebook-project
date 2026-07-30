"""Background job workers — thin wrappers that run in daemon threads.

Each worker receives a notebook_id and the Flask app object so it can push
an application context for DB access. Workers handle their own error logging
and status updates.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


def launch_audio_job(
    notebook_id: int,
    app: object,
    topic: str = "",
    speaker_a: str = "Ava",
    speaker_b: str = "Andrew",
) -> None:
    """Launch a background thread to generate an Audio Overview."""

    def _run() -> None:
        try:
            with app.app_context():  # type: ignore[attr-defined]
                from src.services.audio_service import generate_audio_for_notebook

                result = generate_audio_for_notebook(
                    notebook_id, topic=topic, speaker_a=speaker_a, speaker_b=speaker_b
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
            try:
                with app.app_context():  # type: ignore[attr-defined]
                    from src.extensions import db
                    from src.models import Notebook

                    nb = db.session.get(Notebook, notebook_id)
                    if nb is not None:
                        nb.audio_status = "failed"
                        db.session.commit()
            except Exception:
                logger.exception(
                    "Failed to set audio status to failed for notebook %d", notebook_id
                )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


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


def launch_video_job(notebook_id: int, app: object, topic: str = "", speaker: str = "Ava") -> None:
    """Launch a background thread to generate a Video Overview."""

    def _run() -> None:
        try:
            with app.app_context():  # type: ignore[attr-defined]
                from src.services.video_service import generate_video_for_notebook

                result = generate_video_for_notebook(notebook_id, topic=topic, speaker=speaker)
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
            try:
                with app.app_context():  # type: ignore[attr-defined]
                    from src.extensions import db
                    from src.models import Notebook

                    nb = db.session.get(Notebook, notebook_id)
                    if nb is not None:
                        nb.video_status = "failed"
                        db.session.commit()
            except Exception:
                logger.exception(
                    "Failed to set video status to failed for notebook %d", notebook_id
                )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
