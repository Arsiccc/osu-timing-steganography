"""Create the immutable cohort/provenance lock before validation outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from osu_stego.paths import CONFIG_DIR, METADATA_DIR, RESULTS_DIR
from scripts.experiments.adaptive_layout_strong_common import LAYOUT_KEYS, N_VALUE, PAYLOAD_FRACTION, PN_KEY, write_or_validate_config
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_ecc_equal_payload import key_id
from scripts.experiments.run_ecc_integrity_validation import EXPECTED_RULE_SHA256
from scripts.experiments.run_ecc_pilot import DEFAULT_PARTITION
from scripts.experiments.run_payload_sweep import message_length_for_fraction, nominal_capacity_bits
from scripts.experiments.sender_local_common import load_partitioned_cohort


OUTPUT_DIR = RESULTS_DIR / "ecc_integrity_validation_v1"
FROZEN_RULE = CONFIG_DIR / "ecc_integrity_rejection_v1.json"
POLICY = CONFIG_DIR / "adaptive_alpha_sender_local_v2.json"


def main() -> None:
    output_dir = OUTPUT_DIR
    lock_path = output_dir / "validation_lock.json"
    selection_path = output_dir / "validation_selection.csv"
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if file_sha256(FROZEN_RULE) != EXPECTED_RULE_SHA256:
            raise ValueError("Frozen rule se promenio nakon validation lock-a.")
        if file_sha256(selection_path) != lock["validation_selection_sha256"]:
            raise ValueError("Validation selection se promenio nakon lock-a.")
        runner = Path("scripts/experiments/run_ecc_integrity_validation.py")
        locked_runner_hash = lock["source_sha256"][str(runner)]
        current_runner_hash = file_sha256(runner)
        if current_runner_hash != locked_runner_hash:
            amendment = {
                "experiment_version": "ecc-integrity-validation-v1",
                "original_validation_lock_sha256": file_sha256(lock_path),
                "reason": (
                    "pre-main-run provenance fix: config_id now uses validation experiment "
                    "namespace instead of imported development-holdout namespace"
                ),
                "scientific_behavior_changed": False,
                "frozen_rule_changed": False,
                "selection_changed": False,
                "main_validation_outcome_rows_before_amendment": 0,
                "changed_source_file": str(runner),
                "locked_source_sha256": locked_runner_hash,
                "amended_source_sha256": current_runner_hash,
                "frozen_rule_sha256": file_sha256(FROZEN_RULE),
                "validation_selection_sha256": file_sha256(selection_path),
            }
            amendment_path = output_dir / "validation_lock_amendment.json"
            if not amendment_path.is_file() and (
                output_dir / "physical_provenance.csv"
            ).is_file():
                raise ValueError("Glavni validation outcomes postoje; novi source amendment nije dozvoljen.")
            write_or_validate_config(amendment_path, amendment)
            print(
                "validation lock amendment created | "
                f"sha256={file_sha256(amendment_path)}"
            )
        print(f"validation lock unchanged | sha256={file_sha256(lock_path)}")
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Validation output već sadrži fajlove pre lock-a.")
    if file_sha256(FROZEN_RULE) != EXPECTED_RULE_SHA256:
        raise ValueError("Frozen integrity rule nema očekivani SHA-256.")

    validation = load_partitioned_cohort(
        METADATA_DIR / "results_v3_clean.csv",
        METADATA_DIR / "performance_groups.csv",
        DEFAULT_PARTITION,
        "validation",
    ).copy()
    development = load_partitioned_cohort(
        METADATA_DIR / "results_v3_clean.csv",
        METADATA_DIR / "performance_groups.csv",
        DEFAULT_PARTITION,
        "development",
    )
    validation["nominal_payload_bits"] = validation.num_notes.map(
        lambda count: message_length_for_fraction(
            nominal_capacity_bits(int(count), N_VALUE), PAYLOAD_FRACTION
        )
    )
    validation["useful_bits"] = (validation.nominal_payload_bits // 8) * 4
    eligible = validation[validation.nominal_payload_bits >= 8].sort_values(
        ["beatmap_hash", "replay_file"]
    ).copy()
    ineligible = validation[validation.nominal_payload_bits < 8]
    design = pd.read_csv(RESULTS_DIR / "ecc_pilot_v1" / "pilot_selection.csv")
    holdout = pd.read_csv(RESULTS_DIR / "ecc_integrity_rejection_v1" / "holdout_selection.csv")
    if set(validation.beatmap_hash) & set(development.beatmap_hash):
        raise ValueError("Development i validation beatmape se preklapaju.")
    if set(eligible.replay_file) & (set(design.replay_file) | set(holdout.replay_file)):
        raise ValueError("Validation replay postoji u development design/holdout kohorti.")
    if len(validation) != 338 or validation.beatmap_hash.nunique() != 11:
        raise ValueError("Neočekivana frozen validation particija.")
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_columns = [
        "replay_file", "beatmap_hash", "performance_category", "num_notes",
        "nominal_payload_bits", "useful_bits",
    ]
    eligible[selection_columns].to_csv(selection_path, index=False)
    source_paths = [
        Path("osu_stego/stego/adaptive_alpha.py"),
        Path("osu_stego/stego/ecc.py"),
        Path("osu_stego/stego/integrity.py"),
        Path("osu_stego/stego/payload_layout.py"),
        Path("osu_stego/matching/matcher.py"),
        Path("osu_stego/parsing/osr_writer.py"),
        Path("osu_stego/parsing/replay_loader.py"),
        Path("scripts/experiments/run_layout_comparison.py"),
        Path("scripts/experiments/run_pilot_ber_sweep.py"),
        Path("scripts/experiments/run_ecc_integrity_holdout.py"),
        Path("scripts/experiments/run_ecc_integrity_validation.py"),
    ]
    lock = {
        "experiment_version": "ecc-integrity-validation-v1",
        "lock_created_before_any_validation_outcome": True,
        "prior_validation_result_rows": 0,
        "frozen_rule_sha256": file_sha256(FROZEN_RULE),
        "expected_frozen_rule_sha256": EXPECTED_RULE_SHA256,
        "adaptive_policy_sha256": file_sha256(POLICY),
        "split_sha256": file_sha256(DEFAULT_PARTITION),
        "validation_selection_sha256": file_sha256(selection_path),
        "cohort_sha256": file_sha256(METADATA_DIR / "results_v3_clean.csv"),
        "performance_groups_sha256": file_sha256(METADATA_DIR / "performance_groups.csv"),
        "map_offsets_sha256": file_sha256(CONFIG_DIR / "map_time_offsets.json"),
        "full_validation_replays": len(validation),
        "full_validation_beatmaps": int(validation.beatmap_hash.nunique()),
        "eligible_validation_replays": len(eligible),
        "eligible_validation_beatmaps": int(eligible.beatmap_hash.nunique()),
        "ineligible_replays_below_one_complete_codeword": len(ineligible),
        "ineligible_beatmaps": int(ineligible.beatmap_hash.nunique()),
        "eligibility_rule": "nominal payload at frozen 7.5% must be >= 8 physical bits",
        "development_validation_beatmap_overlap": 0,
        "validation_design_replay_overlap": 0,
        "validation_development_holdout_replay_overlap": 0,
        "N": N_VALUE,
        "payload_fraction": PAYLOAD_FRACTION,
        "pn_key_id": key_id(PN_KEY),
        "layout_key_ids": [key_id(value) for value in LAYOUT_KEYS],
        "layout_seed_count": len(LAYOUT_KEYS),
        "git_commit": "unavailable_not_a_git_worktree",
        "source_sha256": {str(path): file_sha256(path) for path in source_paths},
        "accept_reject_inputs": [
            "hard SECDED status", "syndrome", "overall parity", "received |C| values"
        ],
        "ground_truth_used_for_operational_decision": False,
        "rule_cli_override_available": False,
        "threshold_search": False,
    }
    write_or_validate_config(lock_path, lock)
    print(f"validation lock created | sha256={file_sha256(lock_path)}")
    print(
        f"eligible={len(eligible)}/338 | maps={eligible.beatmap_hash.nunique()}/11 | "
        f"development_map_overlap=0"
    )


if __name__ == "__main__":
    main()
