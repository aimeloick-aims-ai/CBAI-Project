"""Record software verification for the exact source snapshot, including dirty files."""

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "reports/software_verification"
    output.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "pyproject.toml"]
    for folder in ("src", "scripts", "tests", "experiments", "examples"):
        paths.extend((ROOT / folder).rglob("*.py"))
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)
    }
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    commands = [
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--junitxml=reports/software_verification/junit.xml",
        ],
    ]
    results = []
    for i, command in enumerate(commands):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        (output / f"check_{i}.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        results.append({"command": command, "returncode": result.returncode})
        print(result.stdout, flush=True)
    unchanged = all(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
        for name, sha in hashes.items()
    )
    passed = unchanged and all(r["returncode"] == 0 for r in results)
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "base_git_head": head,
        "working_tree_status_before_checks": dirty,
        "scope": "exact source snapshot; not a clean HEAD claim if working tree dirty",
        "source_sha256": hashes,
        "sources_unchanged_during_checks": unchanged,
        "checks": results,
        "passed": passed,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
