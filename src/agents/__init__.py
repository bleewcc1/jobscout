"""JobScout sub-agents + skill loading.

COURSE CONCEPT (Agent Skills / progressive disclosure): each skill's
metadata (frontmatter description) is cheap and always known; the full
body is loaded into an agent's system prompt only when that agent actually
runs. One skill, one job.
"""

from __future__ import annotations

import os
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"

# Override with GEMINI_MODEL for a different Gemini model.
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def thinking_kwargs() -> dict:
    """Keep the agent call sites provider-neutral."""
    return {}


def load_skill(name: str) -> str:
    """Return the SKILL.md body (frontmatter stripped) for injection into
    an agent's system prompt at invocation time — progressive disclosure."""
    path = SKILLS_DIR / name / "SKILL.md"
    text = path.read_text()
    if text.startswith("---"):
        # strip YAML frontmatter: body starts after the second '---'
        _, _, body = text.split("---", 2)
        return body.strip()
    return text.strip()
