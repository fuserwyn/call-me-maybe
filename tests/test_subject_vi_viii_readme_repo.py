"""Subject VI (README requirements) and VIII (submission contents)."""

from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_SECTIONS = [
    "Description",
    "Instructions",
    "Resources",
    "Algorithm explanation",
    "Design decisions",
    "Performance analysis",
    "Challenges faced",
    "Testing strategy",
    "Example usage",
]


def readme() -> str:
    """Return README.md as text."""
    return (ROOT / "README.md").read_text(encoding="utf-8")


def section(text: str, title: str) -> str:
    """Body of the ``## title`` section, up to the next heading of that level."""
    match = re.search(
        rf"^(#+)\s*{re.escape(title)}\s*$", text, re.MULTILINE | re.IGNORECASE
    )
    if match is None:
        return ""
    level = len(match.group(1))
    rest = text[match.end():]
    end = re.search(rf"^#{{1,{level}}}\s", rest, re.MULTILINE)
    return rest[: end.start()] if end else rest


class ReadmeTest(unittest.TestCase):
    """VI Readme Requirements."""

    def test_readme_at_repo_root(self) -> None:
        """README.md exists at the root of the Git repository."""
        self.assertTrue((ROOT / "README.md").is_file())
        self.assertTrue((ROOT / ".git").exists())

    def test_first_line_is_the_italic_42_credit(self) -> None:
        """First line: *This project has been created as part of the 42
        curriculum by <login>[, <login>...].*"""
        first = readme().splitlines()[0]
        self.assertRegex(
            first,
            r"^[*_]This project has been created as part of the 42 curriculum "
            r"by [\w-]+(, [\w-]+)*\.[*_]$",
        )

    def test_required_sections_present(self) -> None:
        """Every section from chapter VI has a heading."""
        text = readme()
        missing = [
            title for title in REQUIRED_SECTIONS
            if not re.search(rf"^#+\s*{re.escape(title)}\s*$", text,
                             re.MULTILINE | re.IGNORECASE)
        ]
        self.assertEqual(missing, [])

    def test_sections_are_not_empty(self) -> None:
        """Each required section has real content."""
        text = readme()
        thin = [
            title for title in REQUIRED_SECTIONS
            if len(section(text, title).split()) < 15
        ]
        self.assertEqual(thin, [])

    def test_resources_list_references_and_ai_usage(self) -> None:
        """Resources: classic references with links + how AI was used."""
        body = section(readme(), "Resources")
        self.assertGreaterEqual(len(re.findall(r"https?://", body)), 2)
        self.assertRegex(body, r"\bAI\b")
        self.assertRegex(body.lower(), r"(used|use) .*(for|to)")

    def test_written_in_english(self) -> None:
        """No Cyrillic text in the README."""
        self.assertIsNone(re.search(r"[Ѐ-ӿ]", readme()))

    def test_example_usage_shows_the_run_command(self) -> None:
        """Example usage shows how to run the program."""
        self.assertIn("uv run python -m src", readme())


class SubmissionTest(unittest.TestCase):
    """VIII Submission and peer review."""

    tracked: list[str]

    @classmethod
    def setUpClass(cls) -> None:
        """List the files Git will submit."""
        if shutil.which("git") is None:
            raise unittest.SkipTest("git is not on PATH")
        result = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
            check=False,
        )
        cls.tracked = result.stdout.splitlines()

    def test_required_files_are_committed(self) -> None:
        """src/, pyproject.toml, uv.lock, llm_sdk/, data/input/, README.md."""
        required = [
            "src/__main__.py",
            "pyproject.toml",
            "uv.lock",
            "llm_sdk/llm_sdk/__init__.py",
            "llm_sdk/pyproject.toml",
            "data/input/function_calling_tests.json",
            "data/input/functions_definition.json",
            "README.md",
            "Makefile",
            ".gitignore",
        ]
        self.assertEqual([p for p in required if p not in self.tracked], [])

    def test_output_directory_is_not_committed(self) -> None:
        """The output/ directory must not be in the repository."""
        leaked = [
            path for path in self.tracked
            if path.startswith(("data/output/", "output/"))
        ]
        self.assertEqual(leaked, [])

    def test_no_generated_artifacts_committed(self) -> None:
        """No caches, venvs or model weights in Git."""
        junk = re.compile(
            r"(__pycache__|\.mypy_cache|\.venv/|\.pyc$|\.safetensors$|\.bin$)"
        )
        self.assertEqual([p for p in self.tracked if junk.search(p)], [])


if __name__ == "__main__":
    unittest.main()
