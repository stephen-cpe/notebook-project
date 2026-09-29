"""Answer difficulty control (ported from study-and-learn prompts)."""

from __future__ import annotations

DIFFICULTY_LEVELS = ("Easy", "Normal", "Hard")

DIFFICULTY_INSTRUCTIONS = {
    "Easy": (
        "Explain like the reader is 10-11 years old: lead with a familiar "
        "analogy, define every term on first use, prefer short sentences."
    ),
    "Normal": (
        "Explain like the reader is 12-13 years old: define terms on first "
        "use, keep an enthusiastic expert tone."
    ),
    "Hard": (
        "Explain like the reader is 14-15 years old: use full vocabulary, "
        "assume basic background, go deeper on mechanisms and connections."
    ),
}


def difficulty_instruction(level: str) -> str:
    """Return the instruction block for a difficulty level (safe default)."""
    return DIFFICULTY_INSTRUCTIONS.get(level, DIFFICULTY_INSTRUCTIONS["Normal"])


def normalize_difficulty(level: str) -> str:
    """Normalize a user-supplied difficulty to a known level."""
    text = (level or "").strip().capitalize()
    return text if text in DIFFICULTY_LEVELS else "Normal"
