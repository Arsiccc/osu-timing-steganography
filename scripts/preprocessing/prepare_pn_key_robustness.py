"""Freeze deterministic PN keys and a development pilot cohort before outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR, RESULTS_DIR
from scripts.experiments.adaptive_layout_strong_common import (
    LAYOUT_KEYS,
    MESSAGE_SEED,
    N_VALUE,
    PAYLOAD_FRACTION,
    PN_KEY,
)
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_layout_comparison import key_id
from scripts.experiments.run_payload_sweep import (
    message_length_for_fraction,
    nominal_capacity_bits,
)
from scripts.experiments.sender_local_common import load_partitioned_cohort


VERSION = "pn-key-robustness-v1"
MASTER_MATERIAL = "osu-stego|pn-key-robustness-v1|preregistered-2026-09-02"
OUTPUT_DIR = RESULTS_DIR / "pn_key_robustness_v1"
CONFIG_PATH = CONFIG_DIR / "pn_key_robustness_v1.json"
PARTITION_PATH = RESULTS_DIR / "adaptive_alpha_validation_v1" / "heldout_partition.csv"
RESULTS_PATH = METADATA_DIR / "results_v3_clean.csv"
PERFORMANCE_PATH = METADATA_DIR / "performance_groups.csv"


def derive_key(index: int) -> str:
    label = f"pn_robustness_{index:02d}"
    digest = hashlib.sha256(f"{MASTER_MATERIAL}|{label}".encode("utf-8")).hexdigest()
    return f"{VERSION}:{label}:{digest}"


def stable_order_value(beatmap_hash: str, replay_file: str) -> str:
    material = f"{VERSION}|pilot-selection|{beatmap_hash}|{replay_file}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def diverse_selection(cohort: pd.DataFrame, size: int = 100) -> pd.DataFrame:
    if size <= 0 or size > len(cohort):
        raise ValueError("Pilot size mora biti u opsegu dostupnog development cohort-a.")
    ordered = cohort.copy()
    ordered["selection_hash"] = [
        stable_order_value(str(row.beatmap_hash), str(row.replay_file))
        for row in ordered.itertuples(index=False)
    ]
    groups = {
        name: group.sort_values(["selection_hash", "replay_file"]).reset_index(drop=True)
        for name, group in ordered.groupby("beatmap_hash", sort=True)
    }
    chosen: list[pd.Series] = []
    depth = 0
    while len(chosen) < size:
        added = False
        for beatmap_hash in sorted(groups):
            group = groups[beatmap_hash]
            if depth < len(group):
                chosen.append(group.iloc[depth])
                added = True
                if len(chosen) == size:
                    break
        if not added:
            raise RuntimeError("Nije moguće popuniti pilot selection.")
        depth += 1
    selected = pd.DataFrame(chosen).reset_index(drop=True)
    selected.insert(0, "selection_order", np.arange(len(selected), dtype=int))
    return selected


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists() or (OUTPUT_DIR / "pilot_selection.csv").exists():
        raise FileExistsError("PN config/selection već postoji; odbijam overwrite.")

    new_keys = [derive_key(index) for index in range(10)]
    all_keys = [PN_KEY, *new_keys]
    ids = [key_id(value) for value in all_keys]
    if len(set(all_keys)) != len(all_keys) or len(set(ids)) != len(ids):
        raise RuntimeError("PN ključevi ili njihovi identifikatori nisu jedinstveni.")

    config = {
        "version": VERSION,
        "scientific_status": "preregistered_before_any_pn_outcome",
        "master_derivation_rule": "key = version:label:SHA256(master_material|label)",
        "master_material": MASTER_MATERIAL,
        "historical_reference": {"key": PN_KEY, "key_id": key_id(PN_KEY)},
        "new_keys": [
            {"index": index, "label": f"pn_robustness_{index:02d}", "key": value, "key_id": key_id(value)}
            for index, value in enumerate(new_keys)
        ],
        "pn_implementation": "SHA256(str(key)) first 4 bytes big-endian -> NumPy default_rng(PCG64)",
        "pn_sequence_sha256": file_sha256(Path("osu_stego/stego/pn_sequence.py")),
        "N": N_VALUE,
        "payload_fraction": PAYLOAD_FRACTION,
        "message_seed": MESSAGE_SEED,
        "layout_key_ids": [key_id(value) for value in LAYOUT_KEYS],
        "adaptive_policy_sha256": file_sha256(CONFIG_DIR / "adaptive_alpha_sender_local_v2.json"),
        "integrity_rule_sha256": file_sha256(CONFIG_DIR / "ecc_integrity_rejection_v1.json"),
        "feature_schema_sha256": file_sha256(RESULTS_DIR / "strong_steganalysis_v1" / "feature_schema.json"),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "pn_keys.json").write_bytes(CONFIG_PATH.read_bytes())

    cohort = load_partitioned_cohort(
        RESULTS_PATH, PERFORMANCE_PATH, PARTITION_PATH, "development"
    )
    selected = diverse_selection(cohort, 100)
    capacity = selected["num_notes"].astype(int).map(
        lambda count: nominal_capacity_bits(count, N_VALUE)
    )
    allocated = capacity.map(
        lambda value: message_length_for_fraction(value, PAYLOAD_FRACTION)
    )
    selected["nominal_capacity_bits"] = capacity
    selected["normal_allocated_bits"] = allocated
    selected["complete_codewords"] = allocated // 8
    selected["coded_bits"] = selected["complete_codewords"] * 8
    selected["useful_bits"] = selected["complete_codewords"] * 4
    selected["eligible_7_5"] = (selected["complete_codewords"] >= 1).astype(int)
    keep = [
        "selection_order", "replay_file", "beatmap_hash", "skill_category",
        "performance_category", "num_notes", "nominal_capacity_bits",
        "normal_allocated_bits", "complete_codewords", "coded_bits",
        "useful_bits", "eligible_7_5", "selection_hash",
    ]
    selected[keep].to_csv(OUTPUT_DIR / "pilot_selection.csv", index=False)
    print(f"incoming={len(selected)} maps={selected.beatmap_hash.nunique()}")
    eligible = selected[selected.eligible_7_5 == 1]
    print(f"eligible={len(eligible)} eligible_maps={eligible.beatmap_hash.nunique()}")
    print(f"pn_config_sha256={file_sha256(CONFIG_PATH)}")
    print(f"selection_sha256={file_sha256(OUTPUT_DIR / 'pilot_selection.csv')}")


if __name__ == "__main__":
    main()
