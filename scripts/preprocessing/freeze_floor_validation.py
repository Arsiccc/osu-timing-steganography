"""Write the immutable pre-outcome lock for floor-policy validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from osu_stego.paths import CONFIG_DIR, RESULTS_DIR
from scripts.experiments.adaptive_layout_strong_common import LAYOUT_KEYS, MESSAGE_SEED, N_VALUE, PN_KEY
from scripts.experiments.run_adaptive_layout_strong import LEGACY_STRONG_EXPERIMENT
from scripts.experiments.run_ecc_equal_payload import key_id
from scripts.experiments.run_pilot_ber_sweep import stable_seed


OUTPUT_DIR = RESULTS_DIR / "one_codeword_floor_validation_v1"
POLICY = CONFIG_DIR / "payload_policy_one_codeword_floor_v1.json"
ADAPTIVE = CONFIG_DIR / "adaptive_alpha_sender_local_v2.json"
INTEGRITY = CONFIG_DIR / "ecc_integrity_rejection_v1.json"
OFFSETS = CONFIG_DIR / "map_time_offsets_floor_validation_v1.json"
RUNNER = Path("scripts/experiments/run_one_codeword_floor_validation.py")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"Frozen fajl već postoji sa drugim sadržajem: {path}")
    path.write_text(content, encoding="utf-8")


def main() -> None:
    manifest_path = OUTPUT_DIR / "corpus_manifest.csv"
    exclusions_path = OUTPUT_DIR / "exclusions.csv"
    selection_path = OUTPUT_DIR / "new_corpus_selection.json"
    calibration_path = OUTPUT_DIR / "calibration_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    exclusions = pd.read_csv(exclusions_path)
    old_maps = sorted(exclusions.loc[exclusions.entity_type == "beatmap_hash", "identifier"].astype(str).unique())
    new_maps = sorted(manifest.beatmap_hash.astype(str).unique())
    if len(manifest) != 300 or len(new_maps) != 20 or len(old_maps) < 34:
        raise RuntimeError("Corpus/exclusion cardinality nije očekivana.")
    if set(old_maps) & set(new_maps):
        raise RuntimeError("Historical/new beatmap overlap.")
    policy_copy = POLICY.read_text(encoding="utf-8")
    write_once(OUTPUT_DIR / "payload_policy.json", policy_copy)
    source_paths = [
        RUNNER, Path("osu_stego/stego/payload_policy.py"), Path("osu_stego/stego/payload_layout.py"),
        Path("osu_stego/stego/ecc.py"), Path("osu_stego/stego/integrity.py"),
        Path("osu_stego/stego/adaptive_alpha.py"), Path("osu_stego/analysis/timing_features.py"),
        Path("osu_stego/parsing/osr_writer.py"), Path("osu_stego/matching/matcher.py"),
        Path("scripts/experiments/run_layout_comparison.py"), Path("scripts/experiments/run_pilot_ber_sweep.py"),
    ]
    source_hashes = {str(path): sha256_file(path) for path in source_paths}
    lock = {
        "experiment_version": "one-codeword-floor-validation-v1",
        "lock_status": "frozen_before_primary_capacity_classification_and_all_stego_outcomes",
        "payload_policy_sha256": sha256_file(POLICY),
        "adaptive_alpha_policy_sha256": sha256_file(ADAPTIVE),
        "integrity_rule_sha256": sha256_file(INTEGRITY),
        "map_offsets_sha256": sha256_file(OFFSETS),
        "map_offsets_calibration_source": "240 disjoint calibration replays; 12 per new map",
        "corpus_manifest_sha256": sha256_file(manifest_path),
        "calibration_manifest_sha256": sha256_file(calibration_path),
        "corpus_selection_sha256": sha256_file(selection_path),
        "exclusions_sha256": sha256_file(exclusions_path),
        "historical_excluded_beatmaps": old_maps,
        "historical_excluded_beatmaps_count": len(old_maps),
        "new_beatmap_hashes": new_maps,
        "new_beatmap_count": len(new_maps),
        "primary_replay_files": sorted(manifest.replay_file.astype(str).tolist()),
        "primary_replay_count": len(manifest),
        "historical_beatmap_overlap": 0,
        "historical_replay_overlap": 0,
        "historical_username_overlap": 0,
        "N": N_VALUE, "secded": "extended-hamming-[8,4,4]",
        "message_seed": MESSAGE_SEED, "pn_key_id": key_id(PN_KEY),
        "layout_seeds": [{"seed": index, "key_id": key_id(key)} for index, key in enumerate(LAYOUT_KEYS)],
        "feature_schema": "strong-steganalysis-features-v1",
        "feature_schema_sha256": sha256_file(Path("osu_stego/analysis/timing_features.py")),
        "detector_training_features_sha256": sha256_file(RESULTS_DIR / "adaptive_layout_strong_v1" / "features.csv"),
        "detector_training_config_sha256": sha256_file(RESULTS_DIR / "adaptive_layout_strong_v1" / "config.json"),
        "rf": {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 2,
               "n_jobs": 1, "training_scope": "original_development_sender_local_adaptive_distributed_per_seed",
               "random_state": stable_seed(MESSAGE_SEED, LEGACY_STRONG_EXPERIMENT, "generalization-rf")},
        "bootstrap_iterations": 2000, "bootstrap_primary_unit": "beatmap_hash",
        "runner_sha256": sha256_file(RUNNER), "source_sha256": source_hashes,
        "git_commit": None, "git_note": "workspace has no Git worktree metadata",
        "post_lock_prohibitions": ["cap change", "integrity retuning", "alpha change", "feature change",
                                    "RF change", "seed selection", "outcome-based exclusion"],
    }
    write_once(OUTPUT_DIR / "validation_lock.json", json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(f"validation lock sha256={sha256_file(OUTPUT_DIR/'validation_lock.json')}")
    print(f"policy sha256={sha256_file(POLICY)} maps={len(new_maps)} replays={len(manifest)} overlap=0")


if __name__ == "__main__":
    main()
