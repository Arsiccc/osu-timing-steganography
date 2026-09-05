"""Write final diagnostics and SHA-256 manifest for host-aware PN results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from osu_stego.paths import RESULTS_DIR


OUTPUT = RESULTS_DIR / "host_aware_pn_selection_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    tables = {
        name: pd.read_csv(output / name)
        for name in (
            "completed_units.csv", "candidate_scores.csv", "physical_results.csv",
            "bit_results.csv", "timing_features.csv", "index_known_results.csv",
            "blind_decoder_results.csv", "blind_candidate_diagnostics.csv",
        )
    }
    diagnostics = [
        {"check": "completed_units", "value": len(tables["completed_units.csv"]), "expected": 1500},
        {"check": "candidate_scores", "value": len(tables["candidate_scores.csv"]), "expected": 12000},
        {"check": "index_known_conditions", "value": len(tables["index_known_results.csv"]), "expected": 6000},
        {"check": "blind_conditions", "value": len(tables["blind_decoder_results.csv"]), "expected": 6000},
        {"check": "blind_hypotheses", "value": len(tables["blind_candidate_diagnostics.csv"]), "expected": 22500},
        {"check": "deduplicated_physical_runs", "value": len(tables["physical_results.csv"]), "expected": "variable"},
        {"check": "bit_rows", "value": len(tables["bit_results.csv"]), "expected": int(tables["physical_results.csv"]["coded_bits"].sum())},
        {"check": "timing_feature_rows", "value": len(tables["timing_features.csv"]), "expected": 2 * len(tables["physical_results.csv"])},
        {"check": "duplicate_units", "value": int(tables["completed_units.csv"]["unit_id"].duplicated().sum()), "expected": 0},
        {"check": "duplicate_physical_ids", "value": int(tables["physical_results.csv"]["physical_id"].duplicated().sum()), "expected": 0},
        {"check": "wrong_accept_index_known", "value": int(tables["index_known_results.csv"]["wrong_accept"].sum()), "expected": "observed"},
        {"check": "wrong_accept_blind", "value": int(tables["blind_decoder_results.csv"]["wrong_accept"].sum()), "expected": "observed"},
    ]
    pd.DataFrame(diagnostics).to_csv(output / "diagnostics.csv", index=False)
    manifest = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "result_hashes.json":
            manifest[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (output / "result_hashes.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"hashed_artifacts={len(manifest)}")


if __name__ == "__main__":
    main()
