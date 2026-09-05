"""Freeze design evidence, PN candidates, messages, selector, and holdout panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from osu_stego.paths import CONFIG_DIR, METADATA_DIR, RESULTS_DIR
from osu_stego.stego.host_aware_pn import MESSAGE_FAMILIES
from osu_stego.stego.pn_sequence import _derive_seed
from scripts.experiments.adaptive_layout_strong_common import LAYOUT_KEYS
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_layout_comparison import key_id
from scripts.experiments.run_payload_sweep import (
    message_length_for_fraction,
    nominal_capacity_bits,
)
from scripts.experiments.sender_local_common import load_partitioned_cohort


VERSION = "host-aware-pn-selection-v1"
MASTER_MATERIAL = "osu-stego|host-aware-pn-selection-v1|preregistered-2026-09-02"
OUTPUT = RESULTS_DIR / "host_aware_pn_selection_v1"
PN_CONFIG = CONFIG_DIR / "host_aware_pn_candidates_v1.json"
DESIGN_PANEL = RESULTS_DIR / "pn_key_robustness_phase2_v1" / "replay_panel.csv"
PARTITION = RESULTS_DIR / "adaptive_alpha_validation_v1" / "heldout_partition.csv"
N_VALUE = 8
PAYLOAD_FRACTION = 0.075
K_VALUES = (1, 2, 4, 8)


def derive_candidate(index: int) -> dict[str, object]:
    label = f"host_aware_candidate_{index}"
    digest = hashlib.sha256(f"{MASTER_MATERIAL}|{label}".encode("utf-8")).hexdigest()
    key = f"{VERSION}:{label}:{digest}"
    return {
        "index": index,
        "label": label,
        "key": key,
        "key_id": key_id(key),
        "effective_seed_32": _derive_seed(key),
    }


def old_effective_seeds() -> set[int]:
    seeds: set[int] = set()
    for path in (
        CONFIG_DIR / "pn_key_robustness_v1.json",
        CONFIG_DIR / "pn_key_robustness_phase2_v1.json",
    ):
        config = json.loads(path.read_text(encoding="utf-8"))
        if "historical_reference" in config:
            seeds.add(_derive_seed(config["historical_reference"]["key"]))
        for candidate in config["new_keys"]:
            seeds.add(int(candidate.get("effective_seed_32", _derive_seed(candidate["key"]))))
    return seeds


def design_bits() -> pd.DataFrame:
    pieces = []
    for phase, path in (
        ("phase1", RESULTS_DIR / "pn_key_robustness_v1" / "bit_results.csv"),
        ("phase2", RESULTS_DIR / "pn_key_robustness_phase2_v1" / "bit_results.csv"),
    ):
        frame = pd.read_csv(path)
        if phase == "phase1":
            frame = frame[frame["layout_seed"].astype(int) < 3].copy()
            frame["candidate_order"] = frame["pn_key_index"].astype(int) + 1
        else:
            frame["candidate_order"] = frame["pn_key_index"].astype(int)
        frame["design_phase"] = phase
        frame["predicted_margin"] = (
            frame["true_bit"] * frame["clean_pn_correlation"]
            + frame["alpha"] * frame["clean_valid_residuals"]
        )
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def write_design_analysis(bits: pd.DataFrame) -> None:
    grouping = [
        "design_phase", "replay_file", "layout_seed", "candidate_order",
        "pn_key_id", "config_id",
    ]
    candidates = bits.groupby(grouping, as_index=False).agg(
        actual_bit_errors=("is_error", "sum"),
        coded_bits=("is_error", "size"),
        nonpositive_count=("predicted_margin", lambda values: int((values <= 0).sum())),
        minimum_margin=("predicted_margin", "min"),
        mean_margin=("predicted_margin", "mean"),
        lower_quartile_margin=("predicted_margin", lambda values: values.quantile(0.25)),
    )
    candidates["actual_ber"] = candidates["actual_bit_errors"] / candidates["coded_bits"]
    candidates.to_csv(OUTPUT / "selector_candidates.csv", index=False)

    rows: list[dict[str, object]] = []
    rows.append({
        "section": "bit_prediction", "design_phase": "combined", "K": "",
        "metric": "negative_margin_error_auc", "value": roc_auc_score(
            bits["is_error"], -bits["predicted_margin"]
        ),
    })
    for error_value, group in bits.groupby("is_error"):
        rows.append({
            "section": "bit_prediction", "design_phase": "combined", "K": "",
            "metric": f"median_margin_error_{int(error_value)}",
            "value": float(group["predicted_margin"].median()),
        })
    for phase, phase_group in candidates.groupby("design_phase"):
        for metric, risk in (
            ("nonpositive_count", phase_group["nonpositive_count"]),
            ("minimum_margin", -phase_group["minimum_margin"]),
            ("mean_margin", -phase_group["mean_margin"]),
        ):
            rows.append({
                "section": "config_prediction", "design_phase": phase, "K": "",
                "metric": f"spearman_risk_{metric}_vs_ber",
                "value": float(spearmanr(risk, phase_group["actual_ber"]).statistic),
            })
        for k_value in K_VALUES:
            chosen = []
            for _, group in phase_group[phase_group["candidate_order"] < k_value].groupby(
                ["replay_file", "layout_seed"], sort=False
            ):
                chosen.append(group.sort_values(
                    ["nonpositive_count", "minimum_margin", "mean_margin", "pn_key_id"],
                    ascending=[True, False, False, True],
                ).iloc[0])
            selected = pd.DataFrame(chosen)
            rows.append({
                "section": "nested_selector", "design_phase": phase, "K": k_value,
                "metric": "weighted_raw_ber",
                "value": selected["actual_bit_errors"].sum() / selected["coded_bits"].sum(),
            })
    pd.DataFrame(rows).to_csv(OUTPUT / "design_mechanism_analysis.csv", index=False)


def stable_holdout_hash(beatmap_hash: str, replay_file: str) -> str:
    material = f"{VERSION}|holdout|{beatmap_hash}|{replay_file}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def select_diverse_holdout(frame: pd.DataFrame, size: int) -> pd.DataFrame:
    work = frame.copy()
    work["selection_hash"] = [
        stable_holdout_hash(str(row.beatmap_hash), str(row.replay_file))
        for row in work.itertuples(index=False)
    ]
    groups = {
        name: group.sort_values(["selection_hash", "replay_file"]).reset_index(drop=True)
        for name, group in work.groupby("beatmap_hash", sort=True)
    }
    chosen = []
    depth = 0
    while len(chosen) < size:
        added = False
        for beatmap_hash in sorted(groups):
            if depth < len(groups[beatmap_hash]):
                chosen.append(groups[beatmap_hash].iloc[depth])
                added = True
                if len(chosen) == size:
                    break
        if not added:
            raise RuntimeError("Insufficient eligible holdout replays.")
        depth += 1
    selected = pd.DataFrame(chosen).reset_index(drop=True)
    selected.insert(0, "selection_order", np.arange(len(selected), dtype=int))
    return selected


def build_holdout() -> pd.DataFrame:
    cohort = load_partitioned_cohort(
        METADATA_DIR / "results_v3_clean.csv",
        METADATA_DIR / "performance_groups.csv",
        PARTITION,
        "development",
    )
    capacity = cohort["num_notes"].astype(int).map(
        lambda count: nominal_capacity_bits(count, N_VALUE)
    )
    allocated = capacity.map(
        lambda value: message_length_for_fraction(value, PAYLOAD_FRACTION)
    )
    cohort["nominal_capacity_bits"] = capacity
    cohort["normal_allocated_bits"] = allocated
    cohort["complete_codewords"] = allocated // 8
    cohort["coded_bits"] = cohort["complete_codewords"] * 8
    cohort["useful_bits"] = cohort["complete_codewords"] * 4
    cohort = cohort[cohort["complete_codewords"] >= 1].copy()
    design = pd.read_csv(DESIGN_PANEL)
    design_ids = set(design.loc[design["eligible_7_5"] == 1, "replay_file"].astype(str))
    cohort = cohort[~cohort["replay_file"].astype(str).isin(design_ids)]
    selected = select_diverse_holdout(cohort, 100)
    selected["design_replay_overlap"] = selected["replay_file"].isin(design_ids).astype(int)
    keep = [
        "selection_order", "replay_file", "beatmap_hash", "skill_category",
        "performance_category", "num_notes", "nominal_capacity_bits",
        "normal_allocated_bits", "complete_codewords", "coded_bits", "useful_bits",
        "selection_hash", "design_replay_overlap",
    ]
    return selected[keep]


def main() -> None:
    if OUTPUT.exists() or PN_CONFIG.exists():
        raise FileExistsError("Host-aware output/config already exists; refusing overwrite.")
    OUTPUT.mkdir(parents=True)
    bits = design_bits()
    write_design_analysis(bits)

    candidates = [derive_candidate(index) for index in range(8)]
    new_seeds = [int(value["effective_seed_32"]) for value in candidates]
    collisions = set(new_seeds) & old_effective_seeds()
    if len(new_seeds) != len(set(new_seeds)) or collisions:
        raise RuntimeError(f"Effective PN seed collision; stop without replacement: {collisions}")
    pn_config = {
        "version": VERSION,
        "scientific_status": "frozen_before_holdout_physical_outcomes",
        "master_material": MASTER_MATERIAL,
        "derivation_rule": "key=version:label:SHA256(master|label)",
        "pn_generator_semantics": "SHA256(str(key)) first 4 bytes big-endian -> NumPy default_rng",
        "nested_candidate_subsets": {str(k): list(range(k)) for k in K_VALUES},
        "candidates": candidates,
        "effective_seed_collision_count": 0,
    }
    PN_CONFIG.write_text(json.dumps(pn_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "pn_candidates.json").write_bytes(PN_CONFIG.read_bytes())

    selector = {
        "version": VERSION,
        "uses_post_write_information": False,
        "margin_equation": "m_j=b_j*sum_{i in clean-valid block j}(r_i*p_i)+alpha*n_clean_valid_j",
        "ranking": [
            "minimize count(m_j <= 0)", "maximize min(m_j)",
            "maximize mean(m_j)", "lexicographically minimize pn_key_id",
        ],
        "zero_boundary": "m_j == 0 counts as non-positive",
        "design_sources": ["pn_key_robustness_v1", "pn_key_robustness_phase2_v1"],
        "design_warning": "consumed exploratory data; not final selector evidence",
    }
    (OUTPUT / "frozen_selector.json").write_text(
        json.dumps(selector, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    messages = {
        "version": VERSION,
        "families": list(MESSAGE_FAMILIES),
        "definitions": {
            "all_zero": "all bipolar -1",
            "all_one": "all bipolar +1",
            "alternating": "-1,+1 repeating from information-bit index 0",
            "random_a": "deterministic SHA256-seeded stream scoped by replay and beatmap",
            "random_b": "independent deterministic SHA256-seeded stream scoped by replay and beatmap",
        },
        "length_rule": "normal 7.5% allocation, complete SECDED words only",
    }
    (OUTPUT / "message_families.json").write_text(
        json.dumps(messages, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    holdout = build_holdout()
    holdout.to_csv(OUTPUT / "holdout_selection.csv", index=False)
    design_maps = set(pd.read_csv(DESIGN_PANEL).query("eligible_7_5 == 1")["beatmap_hash"])
    holdout_maps = set(holdout["beatmap_hash"])
    print(f"holdout={len(holdout)} maps={len(holdout_maps)} replay_overlap={holdout.design_replay_overlap.sum()}")
    print(f"beatmap_overlap={len(design_maps & holdout_maps)}")
    for path in (PN_CONFIG, OUTPUT / "frozen_selector.json", OUTPUT / "message_families.json", OUTPUT / "holdout_selection.csv"):
        print(f"{path}: {file_sha256(path)}")


if __name__ == "__main__":
    main()
