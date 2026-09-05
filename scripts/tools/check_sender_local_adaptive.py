"""Focused pre-run checks for sender-local adaptive alpha."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import osrparse
import pandas as pd

from osu_stego.paths import DATASET_DIR, METADATA_DIR, RESULTS_DIR
from osu_stego.stego.adaptive_alpha import (
    ACCURACY_FORMULA,
    alpha_from_sender_local_features,
    replay_local_quality,
    replay_local_quality_from_counts,
    load_sender_local_policy,
    alpha_from_replay,
)
from scripts.experiments.run_sender_local_development import (
    ALPHA_GRID,
    candidate_definitions,
)
from scripts.experiments.run_sender_local_validation import (
    assignments_from_policy,
    generalization_auc,
    paired_physical_rows,
    select_method,
    shuffled_assignments,
)
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.sender_local_common import (
    FEATURE_FIELDS,
    RESULT_FIELDS,
    load_partitioned_cohort,
)
from scripts.preprocessing.build_performance_groups import osu_standard_accuracy


PARTITION_PATH = RESULTS_DIR / "adaptive_alpha_validation_v1" / "heldout_partition.csv"
POLICY_PATH = Path("data/config/adaptive_alpha_sender_local_v1.json")


def main() -> None:
    if len(RESULT_FIELDS) != len(set(RESULT_FIELDS)):
        raise AssertionError("Development grid result schema ima duplirane kolone.")
    if len(FEATURE_FIELDS) != len(set(FEATURE_FIELDS)):
        raise AssertionError("Development grid feature schema ima duplirane kolone.")
    if tuple(inspect.signature(alpha_from_sender_local_features).parameters) != (
        "policy", "accuracy"
    ):
        raise AssertionError("Policy inference prima nedozvoljen input.")

    for counts in ((300, 10, 2, 1), (0, 1, 2, 3), (920, 704, 294, 427)):
        quality = replay_local_quality_from_counts(*counts)
        expected = osu_standard_accuracy(*counts)
        if not np.isclose(quality["accuracy"], expected):
            raise AssertionError("Sender-local accuracy se razlikuje od projekta.")

    cohort = load_partitioned_cohort(
        METADATA_DIR / "results_v3_clean.csv",
        METADATA_DIR / "performance_groups.csv",
        PARTITION_PATH,
        "development",
    )
    if len(cohort) != 611 or cohort["beatmap_hash"].nunique() != 23:
        raise AssertionError("Frozen development cohort nije 611/23.")
    partition = pd.read_csv(PARTITION_PATH)
    development = set(partition.loc[partition.partition.eq("development"), "beatmap_hash"])
    validation = set(partition.loc[partition.partition.eq("validation"), "beatmap_hash"])
    if development & validation or len(validation) != 11:
        raise AssertionError("Development/validation beatmap partition nije disjunktna.")

    row = cohort.iloc[0]
    paths = list(DATASET_DIR.glob(f"*/osr/{row.replay_file}"))
    if len(paths) != 1:
        raise AssertionError("Nije pronađen jedinstven single-.osr sanity primer.")
    replay = osrparse.Replay.from_path(paths[0])
    single = replay_local_quality(replay)
    if not np.isclose(single["accuracy"], row.accuracy):
        raise AssertionError("Single-.osr feature ne reprodukuje metadata accuracy.")
    policy = load_sender_local_policy(POLICY_PATH)
    alpha_single = alpha_from_replay(policy, replay)
    alpha_features = alpha_from_sender_local_features(
        policy, accuracy=float(single["accuracy"])
    )
    if alpha_single != alpha_features:
        raise AssertionError("Single-.osr policy se razlikuje od accuracy-only inference-a.")
    if policy.get("validation_evaluated_at_freeze") is not False:
        raise AssertionError("Policy nije zamrznuta pre held-out evaluacije.")
    if set(policy["development_beatmap_hashes"]) != development:
        raise AssertionError("Policy config development mape nisu zamrznuta particija.")
    if set(policy["validation_beatmap_hashes"]) != validation:
        raise AssertionError("Policy config validation mape nisu zamrznuta particija.")

    definitions = candidate_definitions(cohort)
    candidate_ids = [value["candidate_id"] for value in definitions]
    if candidate_ids != [
        "fixed_14", "fixed_15", "oracle_map_relative",
        "accuracy_q4_strong", "accuracy_q4_moderate", "accuracy_binary",
    ]:
        raise AssertionError("Pre-registered candidate skup je promenjen.")
    used_alphas = {
        float(alpha)
        for definition in definitions
        for alpha in definition["alpha_values"]
    }
    if not used_alphas.issubset(set(ALPHA_GRID)):
        raise AssertionError("Candidate koristi alpha van fizičkog grid-a.")

    pilot_dir = RESULTS_DIR / "sender_local_adaptive_v1_pilot_v2"
    if pilot_dir.is_dir():
        grid_results = pd.read_csv(pilot_dir / "development_alpha_grid_results.csv")
        grid_features = pd.read_csv(pilot_dir / "development_alpha_grid_features.csv")
        replay_rows = grid_results[
            ["replay_file", "beatmap_hash", "accuracy"]
        ].drop_duplicates("replay_file")
        sender_assignment = assignments_from_policy(replay_rows, policy)
        shuffled = shuffled_assignments(sender_assignment, 101, "development-pilot")
        if sender_assignment["selected_alpha"].value_counts().to_dict() != shuffled[
            "selected_alpha"
        ].value_counts().to_dict():
            raise AssertionError("Shuffled pilot nije sačuvao alpha frekvencije.")
        policy_hash = file_sha256(POLICY_PATH)
        sender_result, sender_features = select_method(
            grid_results,
            grid_features,
            sender_assignment,
            "sender_local_adaptive",
            policy_hash,
        )
        fixed_result, _ = select_method(
            grid_results,
            grid_features,
            sender_assignment.assign(selected_alpha=15.0),
            "fixed_15",
            policy_hash,
        )
        maps = sorted(sender_features["beatmap_hash"].astype(str).unique())
        train_maps = set(maps[:8])
        train = sender_features[
            sender_features["beatmap_hash"].astype(str).isin(train_maps)
        ]
        test = sender_features[
            ~sender_features["beatmap_hash"].astype(str).isin(train_maps)
        ]
        summary, prediction = generalization_auc(
            train, test, trees=20, seed=42, bootstrap_iterations=20,
            method="development_pilot_sanity",
        )
        if not np.isfinite(summary["roc_auc"]) or len(prediction) != len(test):
            raise AssertionError("Development-only generalization pilot nije validan.")
        physical = paired_physical_rows(
            fixed_result,
            sender_result,
            "development_pilot_sender_minus_fixed",
            seed=42,
            iterations=20,
        )
        if {row["metric"] for row in physical} != {
            "weighted_ber", "mean_energy_per_replay", "pooled_rms_shift",
            "pooled_mean_absolute_shift", "active_carrier_fraction",
            "dropped_for_hit_window", "dropped_for_chronology",
            "new_unmatched_events", "new_matched_events",
            "changed_match_status_total",
        }:
            raise AssertionError("Validation physical pilot nema očekivane metrike.")

    print("SENDER-LOCAL ADAPTIVE CHECK OK")
    print(f"formula={ACCURACY_FORMULA}")
    print("development=611/23 validation=338/11 overlap=0")
    print(f"candidates={candidate_ids}")
    print(f"single_osr={paths[0]}")
    print(f"single_osr_alpha={alpha_single}")
    if pilot_dir.is_dir():
        print("validation_pipeline_pilot=development_only_ok")


if __name__ == "__main__":
    main()
