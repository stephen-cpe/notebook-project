"""Typed exception hierarchy for the service layer.

All application errors derive from ``NotebookError`` so routes can catch a
single base type. Each subclass carries enough context for a useful HTTP
response without leaking internals.
"""

from __future__ import annotations


class NotebookError(Exception):
    """Base class for all notebook-project domain errors."""


class AuthError(NotebookError):
    """Authentication failure (bad credentials, duplicate username, etc.)."""


class DuplicateUsernameError(AuthError):
    """A signup attempt used an already-taken username."""


class InvalidCredentialsError(AuthError):
    """A login attempt used an unknown username or wrong password."""


class AIServiceError(NotebookError):
    """An AI/LLM/embedding/OCR call failed."""


class AIModelUnavailableError(AIServiceError):
    """The configured model could not be reached."""


class AITimeoutError(AIServiceError):
    """An AI call timed out."""


class IngestionError(NotebookError):
    """Source ingestion failed."""


class RetrievalError(NotebookError):
    """RAG retrieval failed."""


class AudioError(NotebookError):
    """Audio Overview generation failed."""


class VideoError(NotebookError):
    """Video Overview generation failed."""


class VoiceError(NotebookError):
    """Voice conversation pipeline failed (non-fatal turn error)."""


class SpeechToTextError(VoiceError):
    """Speech-to-text transcription failed."""


class AudioTooLongError(SpeechToTextError):
    """The submitted audio exceeded the configured maximum duration/size."""


class UnsupportedAudioFormatError(SpeechToTextError):
    """The submitted audio could not be decoded to PCM."""


class TextToSpeechError(VoiceError):
    """Text-to-speech synthesis failed."""


class SummaryError(NotebookError):
    """Summary generation failed."""


class JobError(NotebookError):
    """A background job could not be launched or is in a bad state."""
