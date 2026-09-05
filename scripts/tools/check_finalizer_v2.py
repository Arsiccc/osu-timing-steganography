"""Regression checks for bounded v2 finalizer input selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from scripts.analysis.finalize_research_results_v2 import (
    APPROVED_BRANCH_MANIFESTS,
    ROOT,
    audit_manifests,
    approved_manifest_paths,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(directory: Path, payload_name: str = "payload.txt") -> None:
    directory.mkdir(parents=True)
    payload = directory / payload_name
    payload.write_text("frozen payload\n", encoding="utf-8")
    (directory / "result_hashes.json").write_text(
        json.dumps({"files": {payload_name: digest(payload)}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def check_synthetic_boundary() -> None:
    with tempfile.TemporaryDirectory(prefix="finalizer-v2-test-") as name:
        root = Path(name)
        approved = "results/approved_branch/result_hashes.json"
        write_manifest(root / "results" / "approved_branch")

        write_manifest(root / "results" / "final_audit_v1")
        write_manifest(root / "results" / "release_audit_v1")
        write_manifest(root / "temporary_output" / "final_audit_v2")

        selected = approved_manifest_paths(root, (approved,))
        assert selected == (root / approved,)
        first = audit_manifests(root, (approved,))
        second = audit_manifests(root, (approved,))
        assert first == second
        assert len(first) == 1
        assert first[0]["branch"] == "approved_branch"
        assert first[0]["all_hashes_valid"] is True


def check_repository_inputs() -> None:
    selected = approved_manifest_paths()
    relative = tuple(path.relative_to(ROOT).as_posix() for path in selected)
    assert relative == APPROVED_BRANCH_MANIFESTS
    rows = audit_manifests()
    assert len(rows) == 8
    assert all(row["all_hashes_valid"] for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-repository-inputs", action="store_true")
    args = parser.parse_args()
    check_synthetic_boundary()
    if args.check_repository_inputs:
        check_repository_inputs()
    print("FINALIZER V2 CHECK OK")


if __name__ == "__main__":
    main()

