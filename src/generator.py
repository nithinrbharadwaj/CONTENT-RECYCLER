"""
generator.py
============
Uses an LLM (OpenAI gpt-4o-mini or Google Gemini) to rewrite/recycle a
social media post for a new target platform.

Supported providers (set via LLM_PROVIDER env var):
    "openai"  → uses OPENAI_API_KEY  (default)
    "gemini"  → uses GOOGLE_API_KEY
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Any, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
OPENAI_MODEL: str = "gpt-4o-mini"
GEMINI_MODEL: str = "gemini-2.0-flash-lite"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
RECYCLE_PROMPT_TEMPLATE = """\
You are an expert Content Strategist. Your job is to RECYCLE an old social media post.

ORIGINAL POST (from {source_platform}, posted {original_date}):
"{original_text}"

TASK:
Rewrite this post for {target_platform} in {target_year}.
- Keep the core message and brand voice intact
- Update any outdated references or statistics
- Optimize for {target_platform} best practices (length, hashtags, tone)
- Do NOT copy the original. Make it feel fresh.

OUTPUT: Only the final post. No explanations.\
"""


def _build_prompt(
    original_text: str,
    source_platform: str,
    original_date: str,
    target_platform: str,
    target_year: int,
) -> str:
    return RECYCLE_PROMPT_TEMPLATE.format(
        source_platform=source_platform,
        original_date=original_date,
        original_text=original_text,
        target_platform=target_platform,
        target_year=target_year,
    )


# ---------------------------------------------------------------------------
# Provider: OpenAI
# ---------------------------------------------------------------------------

def _generate_openai(prompt: str) -> Tuple[str, Dict[str, int]]:
    """Call OpenAI chat completion and return (text, usage_dict)."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai package not installed. Run: pip install openai") from e

    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set in environment / .env file.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=512,
    )
    elapsed = time.perf_counter() - t0

    text = response.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "latency_s": round(elapsed, 2),
        "model": OPENAI_MODEL,
    }
    log.info("OpenAI usage: %s", usage)
    return text, usage


# ---------------------------------------------------------------------------
# Provider: Gemini
# ---------------------------------------------------------------------------

def _generate_gemini(prompt: str) -> Tuple[str, Dict[str, int]]:
    """Call Google Gemini and return (text, usage_dict)."""
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(
            "google-genai package not installed. Run: pip install google-genai"
        ) from e

    if not GOOGLE_API_KEY:
        raise EnvironmentError("GOOGLE_API_KEY is not set in environment / .env file.")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    elapsed = time.perf_counter() - t0

    text = response.text.strip()
    usage = {
        "prompt_tokens": len(prompt.split()),
        "completion_tokens": len(text.split()),
        "total_tokens": len(prompt.split()) + len(text.split()),
        "latency_s": round(elapsed, 2),
        "model": GEMINI_MODEL,
    }
    log.info("Gemini usage: %s", usage)
    return text, usage


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recycle_post(
    original_text: str,
    source_platform: str,
    original_date: str,
    target_platform: str,
    target_year: Optional[int] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Rewrite `original_text` for `target_platform` using an LLM.

    Parameters
    ----------
    original_text    : The source social media post text
    source_platform  : Original platform (e.g., "LinkedIn")
    original_date    : Date string of original post (e.g., "2022-03-10")
    target_platform  : Desired target platform (e.g., "Twitter")
    target_year      : Year to write for (defaults to current year)
    provider         : "openai" or "gemini" (defaults to LLM_PROVIDER env var)

    Returns
    -------
    dict with keys: recycled_text, original_text, source_platform,
                    target_platform, usage
    """
    import datetime

    if target_year is None:
        target_year = datetime.date.today().year

    prov = (provider or LLM_PROVIDER).lower()
    prompt = _build_prompt(
        original_text=original_text,
        source_platform=source_platform,
        original_date=original_date,
        target_platform=target_platform,
        target_year=target_year,
    )

    log.info("Generating recycled post via %s…", prov)

    if prov == "openai":
        recycled_text, usage = _generate_openai(prompt)
    elif prov == "gemini":
        recycled_text, usage = _generate_gemini(prompt)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{prov}'. Use 'openai' or 'gemini'.")

    return {
        "recycled_text": recycled_text,
        "original_text": original_text,
        "source_platform": source_platform,
        "target_platform": target_platform,
        "original_date": original_date,
        "target_year": target_year,
        "provider": prov,
        "usage": usage,
    }
