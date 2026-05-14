#!/usr/bin/env python3
"""CLI tool for wiki revision history: log, show, diff."""

import argparse
import difflib
import sys

from storage.database import get_connection
from storage.models import (
    get_page_versions,
    get_page_version,
    get_page_by_title,
)


def format_log(title: str, versions: list[dict]) -> str:
    """Format version history as a readable table."""
    lines = [f"Revision history for: {title}", ""]
    for v in versions:
        ts = v.get("created_at", "unknown")[:16]
        q = v.get("source_question", "") or "未知来源"
        lines.append(f"  v{v['version']}  |  {ts}  |  {q}")
    return "\n".join(lines)


def format_show(version: dict) -> str:
    """Format a single version's full content."""
    lines = [
        f"{version['title']} (v{version['version']})",
        f"    Created: {version.get('created_at', 'unknown')}",
        f"    Source:  {version.get('source_question', '未知来源')}",
        "",
        version.get("content", ""),
    ]
    return "\n".join(lines)


def format_diff(title: str, v1: int, v2: int, old_content: str, new_content: str) -> str:
    """Generate unified diff between two versions."""
    if old_content == new_content:
        return f"No differences between v{v1} and v{v2}"

    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"{title} v{v1}",
        tofile=f"{title} v{v2}",
    )
    return "".join(diff)


def handle_log(title: str) -> str:
    """Handle 'log' subcommand."""
    page = get_page_by_title(title)
    if not page:
        return f"Page '{title}' not found"
    versions = get_page_versions(page["id"])
    if not versions:
        return f"No version history for '{title}'"
    return format_log(title, versions)


def handle_show(title: str, version_num: int | None) -> str:
    """Handle 'show' subcommand."""
    page = get_page_by_title(title)
    if not page:
        return f"Page '{title}' not found"

    if version_num is None:
        versions = get_page_versions(page["id"], limit=1)
        if not versions:
            return f"No versions for '{title}'"
        version_num = versions[0]["version"]

    version = get_page_version(page["id"], version_num)
    if not version:
        return f"Version v{version_num} not found for '{title}'"
    return format_show(version)


def handle_diff(title: str, v1: int, v2: int) -> str:
    """Handle 'diff' subcommand."""
    page = get_page_by_title(title)
    if not page:
        return f"Page '{title}' not found"

    old_ver = get_page_version(page["id"], v1)
    new_ver = get_page_version(page["id"], v2)
    if not old_ver:
        return f"Version v{v1} not found for '{title}'"
    if not new_ver:
        return f"Version v{v2} not found for '{title}'"

    return format_diff(title, v1, v2, old_ver["content"], new_ver["content"])


def main():
    parser = argparse.ArgumentParser(description="Wiki revision history")
    sub = parser.add_subparsers(dest="command", required=True)

    log_p = sub.add_parser("log", help="Show version history")
    log_p.add_argument("title", help="Page title")

    show_p = sub.add_parser("show", help="Show version content")
    show_p.add_argument("title", help="Page title")
    show_p.add_argument("version", nargs="?", type=int, default=None,
                        help="Version number (default: latest)")

    diff_p = sub.add_parser("diff", help="Diff two versions")
    diff_p.add_argument("title", help="Page title")
    diff_p.add_argument("v1", type=int, help="First version")
    diff_p.add_argument("v2", type=int, help="Second version")

    args = parser.parse_args()

    if args.command == "log":
        print(handle_log(args.title))
    elif args.command == "show":
        print(handle_show(args.title, args.version))
    elif args.command == "diff":
        print(handle_diff(args.title, args.v1, args.v2))


if __name__ == "__main__":
    main()
