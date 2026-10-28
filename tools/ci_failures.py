"""Turn the failed tests in a pytest JUnit report into GitHub annotations and a job summary.

GitHub shows a CI job's log only to signed-in users, but annotations are public (on the
run page and through the API), so this makes a red build readable from anywhere.

    python tools/ci_failures.py pytest.xml
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

TAIL_LINES = 30  # the end of a traceback holds the assertion and the line that failed


def failures(report: Path) -> Iterator[tuple[str, str]]:
    """(test name, end of its traceback) for every failed or erroring test."""
    for case in ET.parse(report).iter("testcase"):
        for problem in case:
            if problem.tag in ("failure", "error"):
                name = f"{case.get('classname')}::{case.get('name')}"
                text = (problem.text or problem.get("message") or "").strip()
                yield name, "\n".join(text.splitlines()[-TAIL_LINES:])


def annotation(name: str, text: str) -> str:
    """A GitHub ``::error`` command; its text must be escaped to stay on one line."""

    def escape(s: str) -> str:
        return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")

    title = escape(name).replace(":", "%3A").replace(",", "%2C")
    return f"::error title={title}::{escape(text)}"


def main() -> int:
    found = list(failures(Path(sys.argv[1])))
    for name, text in found:
        print(annotation(name, text))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary and found:
        with open(summary, "a", encoding="utf-8") as out:
            out.write(f"### {len(found)} failed test(s)\n\n")
            for name, text in found:
                out.write(f"<details><summary>{name}</summary>\n\n```\n{text}\n```\n</details>\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
