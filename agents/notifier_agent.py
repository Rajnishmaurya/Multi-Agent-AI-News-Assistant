"""
Notifier Agent for the AI News Assistant.

Expected shared-state keys:
    topic: str
    summary_points: list[str] or list[dict]
    sources: list[str] or list[dict]
    email_text: str       (written by this agent)
    output_file: str      (written by this agent)
    error: str | None     (written by this agent)

This agent generates an email using OpenRouter, then saves it under output/.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def _topic_slug(topic: str) -> str:
    """Convert a topic into a filesystem-safe filename slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return (slug[:60].rstrip("-") or "news")


def _extract_url(item: Any) -> str | None:
    """Extract a URL from a source item while avoiding fabricated links."""
    if isinstance(item, str):
        candidate = item.strip()
    elif isinstance(item, dict):
        candidate = str(item.get("url", "")).strip()
    else:
        return None

    parsed = urlparse(candidate)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return candidate
    return None


def _normalise_points(summary_points: list[Any]) -> list[str]:
    """Normalise summary points to plain text."""
    points = []
    for item in summary_points:
        if isinstance(item, str):
            point = item.strip()
        elif isinstance(item, dict):
            point = str(
                item.get("point")
                or item.get("text")
                or item.get("summary")
                or ""
            ).strip()
        else:
            point = str(item).strip()

        if point:
            points.append(point)

    return points


def _collect_urls(summary_points: list[Any], sources: list[Any]) -> list[str]:
    """Collect only URLs supplied by the upstream agents, removing duplicates."""
    urls = []

    for source in sources:
        url = _extract_url(source)
        if url:
            urls.append(url)

    # Also support source URLs attached directly to individual summary points.
    for point in summary_points:
        if isinstance(point, dict):
            point_sources = point.get("sources", [])
            if isinstance(point_sources, (str, dict)):
                point_sources = [point_sources]
            for source in point_sources:
                url = _extract_url(source)
                if url:
                    urls.append(url)

            for key in ("url", "source_url"):
                url = _extract_url(point.get(key, ""))
                if url:
                    urls.append(url)
