"""Freeze the K=1 message-content reanalysis before new outcome summaries."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy
import pandas
import scipy
import sklearn

from osu_stego.paths import CONFIG_DIR, RESULTS_DIR
from scripts.experiments.run_adaptive_alpha_validation import file_sha256


OUTPUT = RESULTS_DIR / "message_content_robustness_v1"
SOURCE = RESULTS_DIR / "host_aware_pn_selection_v1"


def main() -> None:
    lock_path = OUTPUT / "experiment_lock.json"
    if lock_path.exists():
        raise FileExistsError("Message-content experiment lock already exists.")
    source_artifacts = (
        "result_hashes.json", "experiment_lock.json", "message_families.json",
        "pn_candidates.json", "holdout_selection.csv", "completed_units.csv",
        "index_known_results.csv", "physical_results.csv", "bit_results.csv",
        "timing_features.csv",
    )
    analysis_inputs = (
        "source_artifact_audit.md", "message_payload_manifest.csv",
        "duplicate_message_audit.csv", "fold_manifest.csv", "detector_config.json",
    )
    sources = {
        "preparation": Path("scripts/preprocessing/prepare_message_content_robustness.py"),
        "freeze": Path(__file__),
        "analysis": Path("scripts/analysis/analyze_message_content_robustness.py"),
        "sanity": Path("scripts/tools/check_message_content_robustness.py"),
        "result_audit": Path("scripts/tools/check_message_content_results.py"),
        "finalizer": Path("scripts/analysis/finalize_message_content_robustness.py"),
        "message_generator": Path("osu_stego/stego/host_aware_pn.py"),
        "ecc": Path("osu_stego/stego/ecc.py"),
        "integrity": Path("osu_stego/stego/integrity.py"),
        "adaptive_alpha": Path("osu_stego/stego/adaptive_alpha.py"),
        "payload_layout": Path("osu_stego/stego/payload_layout.py"),
        "pn_generator": Path("osu_stego/stego/pn_sequence.py"),
        "writer": Path("osu_stego/parsing/osr_writer.py"),
        "matcher": Path("osu_stego/matching/matcher.py"),
        "timing_features": Path("osu_stego/analysis/timing_features.py"),
        "classifier_families": Path("osu_stego/analysis/classifier_families.py"),
    }
    for path in [*(SOURCE / name for name in source_artifacts), *(OUTPUT / name for name in analysis_inputs), *sources.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    holdout = pandas.read_csv(SOURCE / "holdout_selection.csv")
    candidate = json.loads((SOURCE / "pn_candidates.json").read_text())["candidates"][0]
    source_lock = json.loads((SOURCE / "experiment_lock.json").read_text())
    lock = {
        "experiment_version": "message-content-robustness-v1",
        "lock_created_before_message_specific_outcome_summaries": True,
        "pre_outcome_amendments": [
            "After a failed zero-output analysis attempt, force useful_message and coded_message CSV columns to str; pandas had inferred short binary strings as integers. No outcome artifact had been written or inspected.",
            "After a second zero-output attempt, remove an unnecessary payload-manifest merge in pooled analyses; K=1 already carries encoded_sha256 and the merge had only created suffixed duplicate columns. No outcome artifact had been written or inspected."
        ],
        "scientific_status": "post-hoc robustness reanalysis of consumed physical artifacts",
        "reuse_filter": "index_known_results.csv where integer K == 1",
        "physical_replay_generation_authorized": False,
        "fixed_physical_configuration": {
            "adaptive_alpha_policy": "sender-local-v2",
            "placement": "distributed", "N": 8, "payload_fraction": 0.075,
            "ecc": "SECDED(8,4)", "integrity": "frozen confidence rejection",
            "pn_candidate_index": int(candidate["index"]),
            "pn_key_id": candidate["key_id"],
            "pn_effective_seed_32": int(candidate["effective_seed_32"]),
            "layout_seeds": [0, 1, 2],
        },
        "message_families": ["all_zero", "all_one", "alternating", "random_a", "random_b"],
        "reference_message": "random_a",
        "replay_ids": sorted(holdout.replay_file.astype(str).tolist()),
        "beatmap_ids": sorted(holdout.beatmap_hash.astype(str).unique().tolist()),
        "expected": {
            "replays": 100, "maps": 14, "messages": 5, "layouts": 3,
            "conditions": 1500,
        },
        "bootstrap": {"unit": "replay_file", "iterations": 2000, "seed": 20260903},
        "source_host_aware_lock_sha256": file_sha256(SOURCE / "experiment_lock.json"),
        "source_host_aware_result_manifest_sha256": file_sha256(SOURCE / "result_hashes.json"),
        "source_artifact_sha256": {
            name: file_sha256(SOURCE / name) for name in source_artifacts
        },
        "analysis_input_sha256": {
            name: file_sha256(OUTPUT / name) for name in analysis_inputs
        },
        "frozen_component_sha256": {
            "adaptive_policy": source_lock["adaptive_policy_sha256"],
            "message_families": source_lock["message_families_sha256"],
            "pn_candidates": source_lock["pn_candidates_sha256"],
            "integrity_rule": source_lock["integrity_rule_sha256"],
            "map_offsets": source_lock["map_offsets_sha256"],
            "feature_schema": source_lock["feature_schema_sha256"],
        },
        "source_code_sha256": {name: file_sha256(path) for name, path in sources.items()},
        "versions": {
            "python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"message_content_lock_sha256={file_sha256(lock_path)}")


if __name__ == "__main__":
    main()
