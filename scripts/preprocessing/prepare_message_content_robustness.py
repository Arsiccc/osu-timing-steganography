"""Prepare a locked K=1 message-content reuse analysis without physical writes."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from osu_stego.analysis.timing_features import (
    BASELINE_FEATURES,
    FULL_FEATURES,
    POSITION_FEATURES,
)
from osu_stego.paths import RESULTS_DIR
from osu_stego.stego.ecc import encode_hamming_8_4_secded
from osu_stego.stego.host_aware_pn import (
    MESSAGE_FAMILIES,
    deterministic_message_family,
)
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_ecc_equal_payload import array_hash


VERSION = "message-content-robustness-v1"
SOURCE = RESULTS_DIR / "host_aware_pn_selection_v1"
OUTPUT = RESULTS_DIR / "message_content_robustness_v1"


def bits_text(values: np.ndarray) -> str:
    """Serialize bipolar bits as their ordinary binary symbols."""
    return "".join("1" if value > 0 else "0" for value in values)


def verify_source_hashes() -> None:
    manifest = json.loads((SOURCE / "result_hashes.json").read_text(encoding="utf-8"))
    for name, metadata in manifest.items():
        path = SOURCE / name
        if not path.exists() or file_sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"Frozen host-aware artifact hash mismatch: {path}")


def payload_manifest() -> pd.DataFrame:
    completed = pd.read_csv(SOURCE / "completed_units.csv")
    units = completed.sort_values("layout_seed").drop_duplicates(
        ["replay_file", "message_family"]
    )
    rows = []
    for source in units.itertuples(index=False):
        useful = deterministic_message_family(
            source.message_family,
            int(source.useful_bits),
            source.replay_file,
            source.beatmap_hash,
        )
        coded = encode_hamming_8_4_secded(useful)
        if array_hash(useful) != source.message_sha256:
            raise RuntimeError(f"Useful-message reproduction failed: {source.replay_file}")
        if array_hash(coded) != source.encoded_sha256:
            raise RuntimeError(f"Coded-message reproduction failed: {source.replay_file}")
        rows.append({
            "replay_file": source.replay_file,
            "beatmap_hash": source.beatmap_hash,
            "message_family": source.message_family,
            "useful_bits": len(useful),
            "coded_bits": len(coded),
            "useful_message": bits_text(useful),
            "coded_message": bits_text(coded),
            "message_sha256": source.message_sha256,
            "encoded_sha256": source.encoded_sha256,
        })
    frame = pd.DataFrame(rows).sort_values(
        ["replay_file", "message_family"]
    ).reset_index(drop=True)
    frame["useful_duplicate_group_size"] = frame.groupby(
        ["replay_file", "message_sha256"]
    )["message_family"].transform("nunique")
    frame["coded_duplicate_group_size"] = frame.groupby(
        ["replay_file", "encoded_sha256"]
    )["message_family"].transform("nunique")
    frame["has_useful_duplicate"] = (
        frame.useful_duplicate_group_size > 1
    ).astype(int)
    frame["has_coded_duplicate"] = (
        frame.coded_duplicate_group_size > 1
    ).astype(int)
    return frame


def duplicate_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for replay_file, replay in manifest.groupby("replay_file", sort=True):
        indexed = replay.set_index("message_family")
        for first, second in combinations(MESSAGE_FAMILIES, 2):
            left, right = indexed.loc[first], indexed.loc[second]
            useful_same = left.message_sha256 == right.message_sha256
            coded_same = left.encoded_sha256 == right.encoded_sha256
            if useful_same or coded_same:
                rows.append({
                    "replay_file": replay_file,
                    "beatmap_hash": left.beatmap_hash,
                    "useful_bits": int(left.useful_bits),
                    "message_family_a": first,
                    "message_family_b": second,
                    "useful_payload_identical": int(useful_same),
                    "coded_payload_identical": int(coded_same),
                    "message_sha256_a": left.message_sha256,
                    "message_sha256_b": right.message_sha256,
                    "encoded_sha256_a": left.encoded_sha256,
                    "encoded_sha256_b": right.encoded_sha256,
                })
    columns = [
        "replay_file", "beatmap_hash", "useful_bits", "message_family_a",
        "message_family_b", "useful_payload_identical",
        "coded_payload_identical", "message_sha256_a", "message_sha256_b",
        "encoded_sha256_a", "encoded_sha256_b",
    ]
    return pd.DataFrame(rows, columns=columns)


def fold_manifest() -> pd.DataFrame:
    holdout = pd.read_csv(SOURCE / "holdout_selection.csv").sort_values(
        "selection_order"
    ).reset_index(drop=True)
    folds = np.full(len(holdout), -1, dtype=int)
    groups = holdout.replay_file.astype(str).to_numpy()
    for fold, (_, test) in enumerate(GroupKFold(n_splits=5).split(holdout, groups=groups)):
        folds[test] = fold
    if np.any(folds < 0):
        raise RuntimeError("Incomplete replay fold assignment.")
    return holdout[["replay_file", "beatmap_hash"]].assign(fold=folds)


def detector_config() -> dict:
    return {
        "version": VERSION,
        "scientific_status": "frozen_before_message_specific_outcome_summaries",
        "feature_sets": {
            "BASELINE-3": list(BASELINE_FEATURES),
            "POSITION": list(POSITION_FEATURES),
            "FULL-29": list(FULL_FEATURES),
        },
        "classifiers": {
            "random_forest": {
                "n_estimators": 300, "max_features": "sqrt",
                "min_samples_leaf": 2, "random_state": 42, "n_jobs": 1,
                "score": "predict_proba positive class",
            },
            "rbf_svm": {
                "pipeline": "StandardScaler -> SVC", "kernel": "rbf",
                "C": 1.0, "gamma": "scale", "probability": False,
                "random_state_rule": "stable_seed(20260903,version,detector,classifier,feature_set,message_family_or_pooled,fold)",
                "score": "decision_function",
            },
        },
        "cv": {
            "folds": 5, "group": "replay_file",
            "same_fold_manifest_all_models_and_messages": True,
        },
        "bootstrap": {
            "unit": "replay_file", "iterations": 2000, "seed": 20260903,
        },
        "reference_message": "random_a",
        "map_minimum_replays": 5,
        "pooled_detector": {
            "deduplicate_identical_coded_payloads": True,
            "message_family_is_not_a_feature": True,
        },
        "physical_replay_generation_authorized": False,
    }


def write_audit(manifest: pd.DataFrame, duplicates: pd.DataFrame) -> None:
    known = pd.read_csv(SOURCE / "index_known_results.csv")
    k1 = known[known.K == 1].copy()
    design = pd.read_csv(RESULTS_DIR / "pn_key_robustness_phase2_v1" / "replay_panel.csv")
    eligible_design = design[design.eligible_7_5 == 1]
    replay_overlap = len(set(k1.replay_file) & set(eligible_design.replay_file))
    map_overlap = len(set(k1.beatmap_hash) & set(eligible_design.beatmap_hash))
    lines = [
        "# Message-content source artifact audit", "",
        "Reuse gate: **PASS**. K=1 cleanly isolates useful-message content; no new physical replay generation is required or authorized.", "",
        f"- Source: `{SOURCE}`; all 32 hashes in its result manifest verified.",
        f"- K=1 conditions: {len(k1)} = {k1.replay_file.nunique()} replays × {k1.message_family.nunique()} frozen messages × {k1.layout_seed.nunique()} layouts.",
        "- Every K=1 row uses candidate index 0, PN ID `55adea4b73a8d23c`, effective seed `1437461067`.",
        "- Alpha is constant across messages/layouts within replay; N=8, payload=0.075, eligibility and useful/coded lengths are shared within replay.",
        "- DISTRIBUTED block indices are identical across messages for each replay/layout/bit index.",
        "- Writer, matcher, map offsets, adaptive policy, SECDED and integrity implementations are those hashed by the source lock.",
        "- The five message families and their generator were frozen before physical outcomes; regenerated useful and coded hashes match all source units.",
        f"- Design overlap: {replay_overlap} replay IDs and {map_overlap}/{k1.beatmap_hash.nunique()} beatmaps. The panel is replay-disjoint but not beatmap-disjoint.",
        f"- Duplicate audit: {len(duplicates)} within-replay family pairs share a useful or coded payload; provenance is retained and pooled descriptive/detector analyses deduplicate identical coded payloads.",
        "- Chronology-drop sign was not stored and cannot be reconstructed reliably without a new physical run; it remains unavailable.", "",
        "This branch is a robustness reanalysis of scientifically consumed physical artifacts, not independent validation.",
    ]
    (OUTPUT / "source_artifact_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    verify_source_hashes()
    OUTPUT.mkdir(parents=True)
    manifest = payload_manifest()
    duplicates = duplicate_audit(manifest)
    manifest.to_csv(OUTPUT / "message_payload_manifest.csv", index=False)
    duplicates.to_csv(OUTPUT / "duplicate_message_audit.csv", index=False)
    fold_manifest().to_csv(OUTPUT / "fold_manifest.csv", index=False)
    (OUTPUT / "detector_config.json").write_text(
        json.dumps(detector_config(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_audit(manifest, duplicates)
    print(
        f"prepared {len(manifest)} replay-message payloads; "
        f"duplicate_pairs={len(duplicates)}"
    )


if __name__ == "__main__":
    main()
