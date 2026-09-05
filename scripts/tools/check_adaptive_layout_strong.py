"""Focused pre-run checks for adaptive PREFIX vs DISTRIBUTED strong analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from osu_stego.analysis.layout_diagnostics import quarter_shift_diagnostics
from osu_stego.analysis.timing_features import residual_timing_features
from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR
from osu_stego.stego.adaptive_alpha import (
    alpha_from_sender_local_features,
    load_sender_local_policy,
)
from osu_stego.stego.payload_layout import (
    embed_message_with_layout,
    extract_bipolar_message_with_layout,
    select_payload_blocks,
)
from osu_stego.stego.pn_sequence import generate_pn_sequence
from scripts.experiments.adaptive_layout_strong_common import (
    FEATURE_FIELDS,
    LAYOUT_KEYS,
    MESSAGE_SEED,
    N_VALUE,
    PAYLOAD_FRACTION,
    PN_KEY,
    PREFIX_LAYOUT_KEY,
    RESULT_FIELDS,
    completed_configs,
    message_sha256,
    method_configs,
    replace_config_rows,
)
from scripts.experiments.run_layout_comparison import (
    build_physical_stego,
    experiment_config_id,
)
from scripts.experiments.run_payload_sweep import (
    deterministic_message,
    message_length_for_fraction,
    nominal_capacity_bits,
)
from scripts.experiments.run_pilot_ber_sweep import (
    build_beatmap_index,
    load_map_offsets,
    prepare_replay,
)
from scripts.experiments.sender_local_common import load_partitioned_cohort


PARTITION_PATH = Path("results/adaptive_alpha_validation_v1/heldout_partition.csv")
POLICY_PATH = Path("data/config/adaptive_alpha_sender_local_v1.json")


def synthetic_checks() -> None:
    residuals = np.zeros(800, dtype=float)
    message = np.asarray([1, -1, 1, 1, -1, -1, 1, -1], dtype=np.int8)
    prefix = embed_message_with_layout(
        residuals, message, PN_KEY, PREFIX_LAYOUT_KEY, 14.0, N_VALUE, "prefix"
    )
    legacy_equivalent = embed_message_with_layout(
        residuals, message, PN_KEY, "any-other-layout-key", 14.0, N_VALUE, "prefix"
    )
    if not np.array_equal(prefix, legacy_equivalent):
        raise AssertionError("PREFIX zavisi od layout ključa.")
    for layout_key in LAYOUT_KEYS:
        blocks = select_payload_blocks(
            len(residuals), N_VALUE, len(message), layout_key, "distributed"
        )
        if len(np.unique(blocks)) != len(blocks):
            raise AssertionError("DISTRIBUTED blokovi nisu jedinstveni.")
        if np.any(blocks < 0) or np.any(blocks >= len(residuals) // N_VALUE):
            raise AssertionError("DISTRIBUTED blok je van kapaciteta.")
        stego = embed_message_with_layout(
            residuals, message, PN_KEY, layout_key, 14.0, N_VALUE, "distributed"
        )
        decoded = extract_bipolar_message_with_layout(
            stego, PN_KEY, layout_key, N_VALUE, len(message), "distributed"
        )
        if not np.array_equal(decoded, message):
            raise AssertionError("Noiseless DISTRIBUTED BER nije 0.")
    first = select_payload_blocks(
        len(residuals), N_VALUE, len(message), LAYOUT_KEYS[0], "distributed"
    )
    second = select_payload_blocks(
        len(residuals), N_VALUE, len(message), LAYOUT_KEYS[1], "distributed"
    )
    if np.array_equal(first, second):
        raise AssertionError("Promena layout seed-a nije promenila blokove.")
    chips_a = generate_pn_sequence(PN_KEY, len(message) * N_VALUE)
    chips_b = generate_pn_sequence(PN_KEY, len(message) * N_VALUE)
    if not np.array_equal(chips_a, chips_b):
        raise AssertionError("Layout test je promenio PN sekvencu.")


def note_index_checks() -> None:
    values = np.asarray([1.0, np.nan, np.nan, np.nan, 100.0, 100.0, 100.0, 100.0])
    features = residual_timing_features(values)
    observed = tuple(
        features[f"quarter_{quarter}_mean_absolute_residual"]
        for quarter in range(1, 5)
    )
    if observed != (1.0, 0.0, 100.0, 100.0):
        raise AssertionError("Quarter features ne koriste originalne note indekse.")
    diagnostics = quarter_shift_diagnostics(
        num_notes=8,
        requested_positions=np.asarray([0, 2, 4, 7]),
        active_positions=np.asarray([0, 4, 7]),
        active_shifts_ms=np.asarray([10, -10, 20]),
    )
    if diagnostics["sum_squared_shift_ms2_q1"] != 100.0:
        raise AssertionError("Quarter energy nije note-indeksirana.")
    if sum(diagnostics[f"active_carriers_q{q}"] for q in range(1, 5)) != 3:
        raise AssertionError("Quarter active carrier zbir nije očuvan.")


def resume_identity_checks() -> None:
    base = {
        "method": "sender_local_adaptive",
        "layout": "distributed",
        "layout_seed": 0,
        "policy_sha256": "policy-a",
        "N": 8,
        "payload_fraction": 0.075,
        "input_checksum": "input-a",
    }
    identity = experiment_config_id(**base)
    for field, value in (
        ("method", "fixed_15"),
        ("layout", "prefix"),
        ("layout_seed", 1),
        ("policy_sha256", "policy-b"),
        ("N", 16),
        ("payload_fraction", 0.1),
        ("input_checksum", "input-b"),
    ):
        changed = dict(base)
        changed[field] = value
        if experiment_config_id(**changed) == identity:
            raise AssertionError(f"Resume identity ignoriše {field}.")
    with tempfile.TemporaryDirectory(prefix="adaptive_layout_resume_") as name:
        directory = Path(name)
        result_path = directory / "physical.csv"
        feature_path = directory / "features.csv"
        replace_config_rows(
            result_path,
            RESULT_FIELDS,
            [{"status": "OK", "config_id": identity}],
        )
        replace_config_rows(
            feature_path,
            FEATURE_FIELDS,
            [
                {"config_id": identity, "label": 0},
                {"config_id": identity, "label": 1},
            ],
        )
        if completed_configs(result_path, feature_path) != {identity}:
            raise AssertionError("Kompletan layout config nije prepoznat za resume.")


def physical_prefix_check() -> None:
    cohort = load_partitioned_cohort(
        METADATA_DIR / "results_v3_clean.csv",
        METADATA_DIR / "performance_groups.csv",
        PARTITION_PATH,
        "development",
        1,
    )
    beatmaps = build_beatmap_index(DATASET_DIR)
    offsets = load_map_offsets(CONFIG_DIR / "map_time_offsets.json")
    context = prepare_replay(cohort.iloc[0], DATASET_DIR, beatmaps, offsets)
    policy = load_sender_local_policy(POLICY_PATH)
    alpha = alpha_from_sender_local_features(
        policy,
        accuracy=float(cohort.iloc[0].accuracy),
    )
    capacity = nominal_capacity_bits(len(context.original_residuals), N_VALUE)
    num_bits = message_length_for_fraction(capacity, PAYLOAD_FRACTION)
    message_a = deterministic_message(
        context.replay_file,
        context.beatmap_hash,
        0.0,
        N_VALUE,
        PAYLOAD_FRACTION,
        num_bits,
        MESSAGE_SEED,
    )
    message_b = deterministic_message(
        context.replay_file,
        context.beatmap_hash,
        999.0,
        N_VALUE,
        0.5,
        num_bits,
        MESSAGE_SEED,
    )
    if not np.array_equal(message_a, message_b):
        raise AssertionError("Message zavisi od layout/config pomoćnih argumenata.")
    with tempfile.TemporaryDirectory(prefix="adaptive_layout_prefix_") as name:
        first_path = Path(name) / "first.osr"
        second_path = Path(name) / "second.osr"
        first = build_physical_stego(
            context,
            message_a,
            alpha,
            N_VALUE,
            "prefix",
            PN_KEY,
            PREFIX_LAYOUT_KEY,
            float(policy["hit_margin_ms"]),
            first_path,
        )
        second = build_physical_stego(
            context,
            message_a,
            alpha,
            N_VALUE,
            "prefix",
            PN_KEY,
            LAYOUT_KEYS[0],
            float(policy["hit_margin_ms"]),
            second_path,
        )
        if first_path.read_bytes() != second_path.read_bytes():
            raise AssertionError("PREFIX physical .osr zavisi od layout ključa.")
        if not np.array_equal(first[1], second[1], equal_nan=True):
            raise AssertionError("PREFIX round-trip timing nije ekvivalentan.")
        if first[2] != second[2]:
            raise AssertionError("PREFIX physical diagnostics nisu ekvivalentne.")
    if alpha_from_sender_local_features(
        policy,
        accuracy=float(cohort.iloc[0].accuracy),
    ) != alpha:
        raise AssertionError("Layout seed je uticao na sender-local alpha.")
    if message_sha256(message_a) != message_sha256(message_b):
        raise AssertionError("Message hash nije stabilan.")


def main() -> None:
    if len(method_configs()) != 2 * (1 + len(LAYOUT_KEYS)):
        raise AssertionError("Zamrznuti method/layout config skup nije kompletan.")
    synthetic_checks()
    note_index_checks()
    resume_identity_checks()
    physical_prefix_check()
    print("ADAPTIVE LAYOUT STRONG CHECK OK")
    print(f"layout_seeds={len(LAYOUT_KEYS)} configs_per_replay={len(method_configs())}")


if __name__ == "__main__":
    main()
