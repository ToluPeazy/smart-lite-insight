"""Tests that runtime imports are backed by declared dependencies.

`src/agent.py` imported `requests` while only a transitive dependency supplied
it, which breaks as soon as that transitive edge changes.
"""

import ast
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FIRST_PARTY = {"src", "seed", "dashboard", "scripts", "tests"}


def declared_dependencies() -> set[str]:
    """Distribution names declared in [project].dependencies, normalised."""
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    names = set()
    for spec in pyproject["project"]["dependencies"]:
        name = spec.split(";")[0].strip()
        for separator in ("[", ">", "<", "=", "!", "~", " "):
            name = name.split(separator)[0]
        names.add(name.replace("_", "-").lower())

    return names


def top_level_imports(path: Path) -> set[str]:
    """Top-level module names imported by a Python file."""
    tree = ast.parse(path.read_text())
    modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])

    return modules


# Import name → distribution name, where they differ.
IMPORT_TO_DISTRIBUTION = {
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "yaml": "pyyaml",
}


class TestDeclaredDependencies:
    def test_requests_is_declared(self):
        assert "requests" in declared_dependencies()

    def test_every_third_party_import_in_src_is_declared(self):
        declared = declared_dependencies()
        missing = {}

        for path in sorted((PROJECT_ROOT / "src").glob("*.py")):
            for module in top_level_imports(path):
                if module in sys.stdlib_module_names or module in FIRST_PARTY:
                    continue
                distribution = IMPORT_TO_DISTRIBUTION.get(module, module)
                if distribution.replace("_", "-").lower() not in declared:
                    missing.setdefault(path.name, set()).add(module)

        assert missing == {}
