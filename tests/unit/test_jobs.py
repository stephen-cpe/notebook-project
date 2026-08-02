"""Unit tests for src.services.jobs.

The job workers run background tasks in daemon threads. Unit tests verify
that the launch functions start threads and handle errors gracefully.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.services.jobs import (
    launch_audio_job,
    launch_summary_job,
    launch_video_job,
)


class TestLaunchAudioJob:
    def test_launches_thread(self, app: object) -> None:
        with patch("src.services.jobs.threading.Thread") as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            launch_audio_job(1, app)

            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
            _, kwargs = mock_thread.call_args
            assert kwargs.get("daemon") is True
            assert callable(kwargs["target"])

    def test_thread_runs_audio_generation(self, app: object) -> None:
        with app.app_context():
            from src.extensions import db
            from src.models import Notebook, User
            from src.services.auth_service import hash_password

            u = User(username="jobtest", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Audio Job Test")
            db.session.add(nb)
            db.session.commit()
            nb_id = nb.id

        # jobs.py lazy-imports generate_audio_for_notebook at runtime, so the
        # mock must be placed on the audio_service module.
        with patch("src.services.audio_service.generate_audio_for_notebook") as mock_gen:
            mock_gen.return_value = MagicMock(status="ready")

            launch_audio_job(nb_id, app)
            time.sleep(0.2)

            mock_gen.assert_called_once_with(nb_id, topic="", speaker_a="Ava", speaker_b="Andrew")

    def test_handles_audio_generation_error(
        self, app: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        caplog.set_level(logging.ERROR)

        with app.app_context():
            from src.extensions import db
            from src.models import Notebook, User
            from src.services.auth_service import hash_password

            u = User(username="jobtest2", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Audio Job Test 2")
            db.session.add(nb)
            db.session.commit()
            nb_id = nb.id

        with patch(
            "src.services.audio_service.generate_audio_for_notebook", side_effect=Exception("boom")
        ):
            launch_audio_job(nb_id, app)
            time.sleep(0.2)

        assert any("Audio job crashed" in r.message for r in caplog.records)

        # The notebook's audio_status was set to failed by the error handler.
        with app.app_context():
            from src.extensions import db
            from src.models import Notebook

            nb = db.session.get(Notebook, nb_id)
            assert nb is not None
            assert nb.audio_status == "failed"


class TestLaunchSummaryJob:
    def test_launches_thread(self, app: object) -> None:
        with patch("src.services.jobs.threading.Thread") as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            launch_summary_job(1, app)

            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()

    def test_thread_runs_summary_generation(self, app: object) -> None:
        with app.app_context():
            from src.extensions import db
            from src.models import Notebook, User
            from src.services.auth_service import hash_password

            u = User(username="jobtest3", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Summary Job Test")
            db.session.add(nb)
            db.session.commit()
            nb_id = nb.id

        with patch("src.services.summary_service.SummaryService") as mock_svc_class:
            mock_svc = MagicMock()
            mock_svc.generate_summary.return_value = MagicMock(skipped=False)
            mock_svc_class.return_value = mock_svc

            launch_summary_job(nb_id, app)
            time.sleep(0.2)

            mock_svc.generate_summary.assert_called_once()

    def test_handles_summary_error(self, app: object, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        caplog.set_level(logging.ERROR)

        with app.app_context():
            from src.extensions import db
            from src.models import Notebook, User
            from src.services.auth_service import hash_password

            u = User(username="jobtest4", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Summary Job Test 2")
            db.session.add(nb)
            db.session.commit()
            nb_id = nb.id

        with patch(
            "src.services.summary_service.SummaryService", side_effect=Exception("summary boom")
        ):
            launch_summary_job(nb_id, app)
            time.sleep(0.2)

        assert any("Summary job crashed" in r.message for r in caplog.records)

    def test_missing_notebook_logs_error(
        self, app: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        caplog.set_level(logging.ERROR)

        with patch("src.services.summary_service.SummaryService") as mock_svc_class:
            launch_summary_job(99999, app)
            time.sleep(0.2)

        assert any("not found" in r.message for r in caplog.records)
        mock_svc_class.assert_not_called()


class TestLaunchVideoJob:
    def test_launches_thread(self, app: object) -> None:
        with patch("src.services.jobs.threading.Thread") as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            launch_video_job(1, app)

            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()

    def test_thread_runs_video_generation(self, app: object) -> None:
        with app.app_context():
            from src.extensions import db
            from src.models import Notebook, User
            from src.services.auth_service import hash_password

            u = User(username="jobtest5", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Video Job Test")
            db.session.add(nb)
            db.session.commit()
            nb_id = nb.id

        with patch("src.services.video_service.generate_video_for_notebook") as mock_gen:
            mock_gen.return_value = MagicMock(status="ready")

            launch_video_job(nb_id, app, topic="AI", speaker="Ava")
            time.sleep(0.2)

            mock_gen.assert_called_once_with(nb_id, topic="AI", speaker="Ava")

    def test_handles_video_error(self, app: object, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        caplog.set_level(logging.ERROR)

        with app.app_context():
            from src.extensions import db
            from src.models import Notebook, User
            from src.services.auth_service import hash_password

            u = User(username="jobtest6", password_hash=hash_password("pw"))
            db.session.add(u)
            db.session.commit()
            nb = Notebook(user_id=u.id, name="Video Job Test 2")
            db.session.add(nb)
            db.session.commit()
            nb_id = nb.id

        with patch(
            "src.services.video_service.generate_video_for_notebook",
            side_effect=Exception("video boom"),
        ):
            launch_video_job(nb_id, app)
            time.sleep(0.2)

        assert any("Video job crashed" in r.message for r in caplog.records)

        # The notebook's video_status was set to failed by the error handler.
        with app.app_context():
            from src.extensions import db
            from src.models import Notebook

            nb = db.session.get(Notebook, nb_id)
            assert nb is not None
            assert nb.video_status == "failed"
