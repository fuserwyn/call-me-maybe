"""Subject chapter IV: general rules, Makefile, guidelines, extra requirements."""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

MYPY_FLAGS = [
    "--warn-return-any",
    "--warn-unused-ignores",
    "--ignore-missing-imports",
    "--disallow-untyped-defs",
    "--check-untyped-defs",
]
FORBIDDEN_PACKAGES = {
    "dspy",
    "torch",
    "transformers",
    "huggingface_hub",
    "outlines",
    "guidance",
    "lmformatenforcer",
    "jsonformer",
    "xgrammar",
    "llama_cpp",
    "vllm",
    "tokenizers",
    "sentencepiece",
    "accelerate",
}


def source_files() -> list[Path]:
    """Return every Python file under src/."""
    return sorted(SRC.rglob("*.py"))


def parse(path: Path) -> ast.Module:
    """Parse one source file."""
    with path.open(encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=str(path))


def rel(path: Path) -> str:
    """Path relative to the repo root, for messages."""
    return str(path.relative_to(ROOT))


def run(command: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    """Run a command from the repo root and capture its output."""
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def module_available(name: str) -> bool:
    """Whether ``python -m name`` can run in this interpreter."""
    probe = run([sys.executable, "-c", f"import {name}"], timeout=60)
    return probe.returncode == 0


def makefile_rules() -> dict[str, str]:
    """Map each Makefile target to its recipe text (continuations joined)."""
    rules: dict[str, str] = {}
    current: str | None = None
    with (ROOT / "Makefile").open(encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"^([A-Za-z][\w-]*)\s*:(?!=)", line)
            if match:
                current = match.group(1)
                rules[current] = ""
            elif current and line.startswith("\t"):
                rules[current] += line.strip().rstrip("\\") + " "
    return rules


class GeneralRulesTest(unittest.TestCase):
    """IV.1 General Rules."""

    def test_python_310_or_later(self) -> None:
        """The project declares and runs on Python 3.10+."""
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'requires-python\s*=\s*">=\s*3\.(\d+)', text)
        self.assertIsNotNone(match, "requires-python is missing in pyproject.toml")
        assert match is not None
        self.assertGreaterEqual(int(match.group(1)), 10)
        self.assertGreaterEqual(sys.version_info[:2], (3, 10))

    def test_flake8_passes(self) -> None:
        """``flake8 .`` reports nothing."""
        if not module_available("flake8"):
            self.fail("flake8 is not installed in the project environment")
        result = run([sys.executable, "-m", "flake8", "."])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mypy_passes_with_subject_flags(self) -> None:
        """``mypy .`` with the flags from IV.2 reports nothing."""
        if not module_available("mypy"):
            self.fail("mypy is not installed in the project environment")
        result = run([sys.executable, "-m", "mypy", ".", *MYPY_FLAGS])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_every_function_is_fully_annotated(self) -> None:
        """All parameters and return values in src/ carry type hints."""
        missing: list[str] = []
        for path in source_files():
            for node in ast.walk(parse(path)):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                where = f"{rel(path)}:{node.lineno} {node.name}"
                if node.returns is None:
                    missing.append(f"{where} -> return")
                args = node.args
                every = args.posonlyargs + args.args + args.kwonlyargs
                for extra in (args.vararg, args.kwarg):
                    if extra is not None:
                        every.append(extra)
                for arg in every:
                    if arg.annotation is None and arg.arg not in ("self", "cls"):
                        missing.append(f"{where} ({arg.arg})")
        self.assertEqual(missing, [], "missing type hints")

    def test_every_module_class_and_function_has_docstring(self) -> None:
        """PEP 257 docstrings on modules, classes and functions (nested too)."""
        missing: list[str] = []
        for path in source_files():
            tree = parse(path)
            if ast.get_docstring(tree) is None:
                missing.append(f"{rel(path)} (module)")
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ) and ast.get_docstring(node) is None:
                    missing.append(f"{rel(path)}:{node.lineno} {node.name}")
        self.assertEqual(missing, [], "missing docstrings")

    def test_files_are_opened_with_context_managers(self) -> None:
        """Every ``open()`` / ``Path.open()`` call is a ``with`` item."""
        leaks: list[str] = []
        for path in source_files():
            tree = parse(path)
            managed = {
                id(item.context_expr)
                for node in ast.walk(tree)
                if isinstance(node, (ast.With, ast.AsyncWith))
                for item in node.items
            }
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute)
                    else ""
                )
                if name == "open" and id(node) not in managed:
                    leaks.append(f"{rel(path)}:{node.lineno}")
        self.assertEqual(leaks, [], "open() outside a with-statement")

    def test_no_bare_except(self) -> None:
        """Errors are caught by type, never with a bare ``except:``."""
        bare = [
            f"{rel(path)}:{node.lineno}"
            for path in source_files()
            for node in ast.walk(parse(path))
            if isinstance(node, ast.ExceptHandler) and node.type is None
        ]
        self.assertEqual(bare, [])


class MakefileTest(unittest.TestCase):
    """IV.2 Makefile."""

    rules: dict[str, str]

    @classmethod
    def setUpClass(cls) -> None:
        """Parse the Makefile once."""
        cls.rules = makefile_rules()

    def test_required_rules_exist(self) -> None:
        """install, run, debug, clean and lint are all defined."""
        required = {"install", "run", "debug", "clean", "lint"}
        self.assertEqual(required - set(self.rules), set())

    def test_install_uses_a_package_manager(self) -> None:
        """install runs pip / uv / pipx."""
        self.assertRegex(self.rules["install"], r"\b(uv|pip|pipx)\b")

    def test_run_executes_the_main_module(self) -> None:
        """run starts ``python -m src``."""
        self.assertIn("python -m src", self.rules["run"])

    def test_debug_uses_pdb(self) -> None:
        """debug goes through Python's built-in debugger."""
        self.assertIn("pdb", self.rules["debug"])

    def test_clean_removes_caches(self) -> None:
        """clean deletes __pycache__ and .mypy_cache."""
        self.assertIn("__pycache__", self.rules["clean"])
        self.assertIn(".mypy_cache", self.rules["clean"])

    def test_lint_uses_the_mandatory_flags(self) -> None:
        """lint runs ``flake8 .`` and ``mypy .`` with every required flag."""
        recipe = self.rules["lint"]
        self.assertIn("flake8 .", recipe)
        self.assertIn("mypy .", recipe)
        for flag in MYPY_FLAGS:
            self.assertIn(flag, recipe)

    def test_optional_lint_strict_rule(self) -> None:
        """(optional) lint-strict runs ``mypy . --strict``."""
        self.assertIn("lint-strict", self.rules)
        self.assertIn("mypy . --strict", self.rules["lint-strict"])

    def test_optional_lint_strict_passes(self) -> None:
        """(optional) ``mypy . --strict`` reports nothing."""
        if not module_available("mypy"):
            self.fail("mypy is not installed in the project environment")
        result = run([sys.executable, "-m", "mypy", ".", "--strict"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class AdditionalGuidelinesTest(unittest.TestCase):
    """IV.3 Additional Guidelines."""

    def test_gitignore_excludes_python_artifacts_and_output(self) -> None:
        """.gitignore covers caches, the venv and generated output."""
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ("__pycache__", ".venv", ".mypy_cache", "data/output"):
            self.assertIn(pattern, text)


class AdditionalRequirementsTest(unittest.TestCase):
    """IV.3.1 Additional Requirements."""

    def test_all_classes_use_pydantic(self) -> None:
        """Every class in src/ is a pydantic model (directly or inherited)."""
        bases: dict[str, list[str]] = {}
        where: dict[str, str] = {}
        for path in source_files():
            for node in ast.walk(parse(path)):
                if isinstance(node, ast.ClassDef):
                    bases[node.name] = [
                        base.id if isinstance(base, ast.Name)
                        else base.attr if isinstance(base, ast.Attribute)
                        else ""
                        for base in node.bases
                    ]
                    where[node.name] = f"{rel(path)}:{node.lineno}"
        pydantic = {"BaseModel", "RootModel"}
        changed = True
        while changed:
            changed = False
            for name, parents in bases.items():
                if name not in pydantic and pydantic.intersection(parents):
                    pydantic.add(name)
                    changed = True
        offenders = sorted(f"{where[name]} {name}" for name in bases
                           if name not in pydantic)
        self.assertEqual(offenders, [], "classes that are not pydantic models")

    def _third_party_imports(self) -> dict[str, list[str]]:
        """Top-level imported package -> places it is imported in src/."""
        found: dict[str, list[str]] = {}
        for path in source_files():
            for node in ast.walk(parse(path)):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    top = name.split(".")[0]
                    if top and top not in sys.stdlib_module_names and top != "src":
                        found.setdefault(top, []).append(f"{rel(path)}:{node.lineno}")
        return found

    def test_no_forbidden_libraries(self) -> None:
        """No dspy / torch / transformers / huggingface / outlines & co."""
        used = self._third_party_imports()
        self.assertEqual(
            {name: used[name] for name in FORBIDDEN_PACKAGES & set(used)}, {}
        )

    def test_only_numpy_pydantic_and_llm_sdk(self) -> None:
        """Strict reading: third-party code is limited to what IV.3.1 names."""
        used = self._third_party_imports()
        allowed = {"numpy", "pydantic", "llm_sdk"}
        self.assertEqual(
            {name: places for name, places in used.items() if name not in allowed},
            {},
        )

    def test_dependencies_declared_for_uv(self) -> None:
        """pyproject pins numpy, pydantic and the local llm_sdk copy."""
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertRegex(text, r'"numpy[>=<~ \d.]*"')
        self.assertRegex(text, r'"pydantic[>=<~ \d.]*"')
        self.assertRegex(text, r'llm-sdk\s*=\s*\{\s*path\s*=\s*"\./llm_sdk"')
        self.assertTrue((ROOT / "uv.lock").is_file())

    def test_llm_sdk_copied_next_to_src(self) -> None:
        """llm_sdk/ sits in the same directory as src/."""
        self.assertTrue((ROOT / "src").is_dir())
        self.assertTrue((ROOT / "llm_sdk" / "llm_sdk" / "__init__.py").is_file())

    def test_uv_sync_keeps_the_lint_tools(self) -> None:
        """After the reviewer's plain ``uv sync``, ``make lint`` still works."""
        if shutil.which("uv") is None:
            self.skipTest("uv is not on PATH")
        result = run(["uv", "sync", "--dry-run"], timeout=300)
        output = result.stdout + result.stderr
        removed = re.findall(r"^\s*-\s*(flake8|mypy)==", output, re.MULTILINE)
        self.assertEqual(
            removed, [], "plain `uv sync` uninstalls the lint tools:\n" + output
        )

    def test_uv_lock_matches_pyproject(self) -> None:
        """uv.lock is up to date, so ``uv sync`` installs exactly it."""
        if shutil.which("uv") is None:
            self.skipTest("uv is not on PATH")
        result = run(["uv", "lock", "--check"], timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_no_private_llm_sdk_members(self) -> None:
        """Only public SDK names are imported or touched on the model."""
        problems: list[str] = []
        for path in source_files():
            for node in ast.walk(parse(path)):
                if isinstance(node, ast.ImportFrom) and node.module == "llm_sdk":
                    problems.extend(
                        f"{rel(path)}:{node.lineno} imports {alias.name}"
                        for alias in node.names
                        if alias.name.startswith("_")
                    )
                if not isinstance(node, ast.Attribute):
                    continue
                owner = node.value
                on_model = (
                    isinstance(owner, ast.Name) and owner.id == "model"
                ) or (isinstance(owner, ast.Attribute) and owner.attr == "_model")
                if on_model and node.attr.startswith("_"):
                    problems.append(f"{rel(path)}:{node.lineno} .{node.attr}")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
