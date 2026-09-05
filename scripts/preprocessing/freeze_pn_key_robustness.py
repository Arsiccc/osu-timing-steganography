"""Create the immutable pre-outcome lock for PN-key robustness v1."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from osu_stego.paths import CONFIG_DIR, RESULTS_DIR
from scripts.experiments.adaptive_layout_strong_common import LAYOUT_KEYS, MESSAGE_SEED, N_VALUE, PAYLOAD_FRACTION
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_layout_comparison import key_id


OUTPUT = RESULTS_DIR / "pn_key_robustness_v1"


def main() -> None:
    target = OUTPUT / "experiment_lock.json"
    if target.exists():
        raise FileExistsError("Experiment lock već postoji.")
    pn_path = CONFIG_DIR / "pn_key_robustness_v1.json"
    selection_path = OUTPUT / "pilot_selection.csv"
    config = json.loads(pn_path.read_text())
    selection = pd.read_csv(selection_path)
    source_paths = {
        "runner": Path("scripts/experiments/run_pn_key_robustness.py"),
        "analysis": Path("scripts/analysis/analyze_pn_key_robustness.py"),
        "selection": Path("scripts/preprocessing/prepare_pn_key_robustness.py"),
        "sanity": Path("scripts/tools/check_pn_key_robustness.py"),
        "pn_sequence": Path("osu_stego/stego/pn_sequence.py"),
        "payload_layout": Path("osu_stego/stego/payload_layout.py"),
        "ecc": Path("osu_stego/stego/ecc.py"),
        "integrity": Path("osu_stego/stego/integrity.py"),
        "adaptive_alpha": Path("osu_stego/stego/adaptive_alpha.py"),
        "matcher": Path("osu_stego/matching/matcher.py"),
        "writer": Path("osu_stego/parsing/osr_writer.py"),
        "replay_loader": Path("osu_stego/parsing/replay_loader.py"),
        "timing_features": Path("osu_stego/analysis/timing_features.py"),
    }
    lock = {
        "experiment_version": "pn-key-robustness-v1",
        "lock_created_before_any_physical_pn_outcome": True,
        "scientific_status": "existing-corpus-preregistered-parameter-robustness-not-independent-validation",
        "pn_config_sha256": file_sha256(pn_path),
        "pilot_selection_sha256": file_sha256(selection_path),
        "historical_pn_key_id": config["historical_reference"]["key_id"],
        "new_pn_key_ids": [row["key_id"] for row in config["new_keys"]],
        "incoming_replays": len(selection),
        "incoming_maps": int(selection.beatmap_hash.nunique()),
        "eligible_replays": int(selection.eligible_7_5.sum()),
        "eligible_maps": int(selection.loc[selection.eligible_7_5 == 1, "beatmap_hash"].nunique()),
        "eligibility_rule": "normal fixed 7.5% allocation contains at least one complete SECDED(8,4) codeword",
        "layout_seeds": [{"seed": i, "key_id": key_id(value)} for i, value in enumerate(LAYOUT_KEYS)],
        "N": N_VALUE, "payload_fraction": PAYLOAD_FRACTION, "message_seed": MESSAGE_SEED,
        "message_convention": "same stable replay-specific useful bits for every PN key and layout",
        "adaptive_policy_sha256": file_sha256(CONFIG_DIR / "adaptive_alpha_sender_local_v2.json"),
        "integrity_rule_sha256": file_sha256(CONFIG_DIR / "ecc_integrity_rejection_v1.json"),
        "map_offsets_sha256": file_sha256(CONFIG_DIR / "map_time_offsets.json"),
        "feature_schema_sha256": file_sha256(RESULTS_DIR / "strong_steganalysis_v1" / "feature_schema.json"),
        "rf": {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 2, "n_jobs": 1,
               "random_state_rule": "stable_seed(42,sender-local-heldout-validation-v1,generalization-rf)", "folds": 5},
        "source_sha256": {name: file_sha256(path) for name, path in source_paths.items()},
        "git_commit": "unavailable_not_a_git_worktree",
    }
    target.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"experiment_lock_sha256={file_sha256(target)}")


if __name__ == "__main__":
    main()
