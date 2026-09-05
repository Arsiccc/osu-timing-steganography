"""Freeze the classifier-family experiment lock before alternative outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from osu_stego.paths import CONFIG_DIR, RESULTS_DIR
from scripts.experiments.run_adaptive_alpha_validation import file_sha256


OUTPUT = RESULTS_DIR / "classifier_family_robustness_v1"


def main() -> None:
    lock_path = OUTPUT / "experiment_lock.json"
    if lock_path.exists():
        raise FileExistsError("Experiment lock exists; refusing overwrite.")
    files = {
        "classifier_config_sha256": CONFIG_DIR / "steganalysis_classifier_robustness_v1.json",
        "dataset_manifest_sha256": OUTPUT / "dataset_manifest.csv",
        "fold_manifest_sha256": OUTPUT / "fold_manifest.csv",
        "audit_sha256": OUTPUT / "audit.md",
        "rf_reproduction_sha256": OUTPUT / "rf_reproduction.csv",
        "feature_schema_sha256": RESULTS_DIR / "strong_steganalysis_v1" / "feature_schema.json",
        "runner_sha256": Path("scripts/experiments/run_classifier_family_robustness.py"),
        "classifier_module_sha256": Path("osu_stego/analysis/classifier_families.py"),
        "sanity_sha256": Path("scripts/tools/check_classifier_family_robustness.py"),
        "preparation_sha256": Path("scripts/preprocessing/prepare_classifier_family_robustness.py"),
        "historical_features_sha256": RESULTS_DIR / "adaptive_layout_strong_v1" / "features.csv",
        "development_features_sha256": RESULTS_DIR / "pn_key_robustness_v1" / "timing_features.csv",
        "unseen_features_sha256": RESULTS_DIR / "one_codeword_floor_validation_v1" / "timing_features.csv",
        "pn_subset_lock_sha256": RESULTS_DIR / "pn_key_robustness_phase2_v1" / "detector_subset_lock.json",
    }
    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(path)
    lock = {
        "experiment_version": "classifier-family-robustness-v1",
        "lock_created_before_alternative_classifier_outcomes": True,
        "physical_replay_generation_authorized": False,
        "primary_rows": 2030,
        "primary_replays": 203,
        "development_replays": 68,
        "unseen_new_map_replays": 135,
        "classifiers": ["random_forest", "logistic_regression", "rbf_svm", "hist_gradient_boosting"],
        "feature_sets": ["baseline", "position_only", "full"],
        **{field: file_sha256(path) for field, path in files.items()},
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"experiment_lock_sha256={file_sha256(lock_path)}")


if __name__ == "__main__":
    main()
