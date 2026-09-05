"""Create the immutable pre-outcome lock for host-aware PN selection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from osu_stego.paths import CONFIG_DIR, RESULTS_DIR
from scripts.experiments.adaptive_layout_strong_common import LAYOUT_KEYS
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_layout_comparison import key_id


OUTPUT = RESULTS_DIR / "host_aware_pn_selection_v1"


def main() -> None:
    lock_path = OUTPUT / "experiment_lock.json"
    if lock_path.exists():
        raise FileExistsError("Experiment lock already exists; refusing overwrite.")
    paths = {
        "pn_candidates_sha256": CONFIG_DIR / "host_aware_pn_candidates_v1.json",
        "selector_sha256": OUTPUT / "frozen_selector.json",
        "message_families_sha256": OUTPUT / "message_families.json",
        "holdout_selection_sha256": OUTPUT / "holdout_selection.csv",
        "adaptive_policy_sha256": CONFIG_DIR / "adaptive_alpha_sender_local_v2.json",
        "integrity_rule_sha256": CONFIG_DIR / "ecc_integrity_rejection_v1.json",
        "map_offsets_sha256": CONFIG_DIR / "map_time_offsets.json",
        "feature_schema_sha256": RESULTS_DIR / "strong_steganalysis_v1" / "feature_schema.json",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    holdout = pd.read_csv(paths["holdout_selection_sha256"])
    source_paths = {
        "runner": Path("scripts/experiments/run_host_aware_pn_selection.py"),
        "host_aware_module": Path("osu_stego/stego/host_aware_pn.py"),
        "sanity": Path("scripts/tools/check_host_aware_pn_selection.py"),
        "preparation": Path("scripts/preprocessing/prepare_host_aware_pn_selection.py"),
        "payload_layout": Path("osu_stego/stego/payload_layout.py"),
        "pn_generator": Path("osu_stego/stego/pn_sequence.py"),
        "adaptive_alpha": Path("osu_stego/stego/adaptive_alpha.py"),
        "ecc": Path("osu_stego/stego/ecc.py"),
        "integrity": Path("osu_stego/stego/integrity.py"),
        "writer": Path("osu_stego/parsing/osr_writer.py"),
        "matcher": Path("osu_stego/matching/matcher.py"),
        "timing_features": Path("osu_stego/analysis/timing_features.py"),
    }
    lock = {
        "experiment_version": "host-aware-pn-selection-v1",
        "lock_created_before_holdout_physical_outcomes": True,
        "scientific_status": "internal exploratory replay-disjoint holdout",
        "N": 8,
        "payload_fraction": 0.075,
        "layout_seeds": [
            {"seed": index, "key_id": key_id(LAYOUT_KEYS[index])}
            for index in (0, 1, 2)
        ],
        "K_values": [1, 2, 4, 8],
        "message_families": ["all_zero", "all_one", "alternating", "random_a", "random_b"],
        "holdout_replays": len(holdout),
        "holdout_maps": int(holdout["beatmap_hash"].nunique()),
        "expected_units": len(holdout) * 5 * 3,
        "expected_index_known_conditions": len(holdout) * 5 * 3 * 4,
        "pn_index_side_information_bits": {"1": 0, "2": 1, "4": 2, "8": 3},
        "blind_rule": "zero accepted hypotheses reject; one accept; multiple same-message accept; multiple different-message reject",
        "git_commit": "unavailable_not_a_git_worktree",
        **{field: file_sha256(path) for field, path in paths.items()},
        "source_sha256": {name: file_sha256(path) for name, path in source_paths.items()},
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"experiment_lock_sha256={file_sha256(lock_path)}")


if __name__ == "__main__":
    main()
