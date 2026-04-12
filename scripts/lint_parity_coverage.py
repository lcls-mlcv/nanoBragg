"""
Lint parity_cases.yaml against spec-style AT identifiers in the repo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

_AT_ID = re.compile(r"AT-PARALLEL-\d{3}")


@dataclass
class LintResult:
    passed: bool
    severity: str
    message: str


class ParityCoverageLinter:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.results: List[LintResult] = []

    def _extract_yaml_ats(self, data: Dict[str, Any]) -> Set[str]:
        cases = data.get("cases") or []
        ids = set()
        for c in cases:
            cid = c.get("id")
            if cid:
                ids.add(str(cid))
        return ids

    def _extract_spec_ats(self) -> Set[str]:
        found: Set[str] = set()
        tests_glob = self.repo_root.glob("tests/test_at_parallel_*.py")
        for path in tests_glob:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            found.update(_AT_ID.findall(text))
        for md in self.repo_root.glob("README*.md"):
            try:
                text = md.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            found.update(_AT_ID.findall(text))
        docs = self.repo_root / "docs"
        if docs.is_dir():
            for md in docs.rglob("*.md"):
                try:
                    text = md.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                found.update(_AT_ID.findall(text))
        specs = self.repo_root / "specs"
        if specs.is_dir():
            for md in specs.rglob("*.md"):
                try:
                    text = md.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                found.update(_AT_ID.findall(text))
        return found

    def _validate_cases(self, data: Dict[str, Any]) -> None:
        cases = data.get("cases")
        if not isinstance(cases, list):
            self.results.append(
                LintResult(False, "ERROR", "parity_cases.yaml: missing or invalid 'cases' list")
            )
            return
        for i, c in enumerate(cases):
            if not isinstance(c, dict):
                self.results.append(
                    LintResult(False, "ERROR", f"cases[{i}] is not an object")
                )
                continue
            for key in ("id", "description", "base_args", "thresholds", "runs"):
                if key not in c:
                    self.results.append(
                        LintResult(
                            False,
                            "ERROR",
                            f"case index {i} missing required key {key!r}",
                        )
                    )

    def run_all_checks(self) -> bool:
        self.results = []
        yaml_path = self.repo_root / "tests" / "parity_cases.yaml"
        if not yaml_path.exists():
            self.results.append(
                LintResult(
                    False,
                    "ERROR",
                    "tests/parity_cases.yaml not found",
                )
            )
            return False

        try:
            with open(yaml_path, encoding="utf-8") as f:
                raw = f.read()
            data = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            self.results.append(
                LintResult(False, "ERROR", f"YAML parse error: {e}")
            )
            return False
        except OSError as e:
            self.results.append(
                LintResult(False, "ERROR", f"Could not read parity_cases.yaml: {e}")
            )
            return False

        if not isinstance(data, dict):
            self.results.append(
                LintResult(False, "ERROR", "parity_cases.yaml root must be a mapping")
            )
            return False

        self._validate_cases(data)
        errors = [r for r in self.results if not r.passed and r.severity == "ERROR"]
        if errors:
            return False

        yaml_ats = self._extract_yaml_ats(data)
        self.results.append(
            LintResult(
                True,
                "INFO",
                f"AT cases in parity_cases.yaml: found {len(yaml_ats)} cases",
            )
        )
        return True
