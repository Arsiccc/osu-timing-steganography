"""Focused checks for layout experiment identity and atomic resume storage."""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts.experiments.run_layout_comparison import (
    FEATURE_FIELDS,
    RESULT_FIELDS,
    completed_feature_keys,
    completed_result_keys,
    experiment_config_id,
    replace_config_rows,
)


def _base_row(config_id: str) -> dict:
    return {
        "experiment_version": "test",
        "config_id": config_id,
    }


def main() -> None:
    base = {
        "replay_file": "replay.osr",
        "layout": "distributed",
        "alpha": 14.0,
        "n_frames_per_bit": 8,
        "payload_fraction": 0.1,
        "message_seed": 42,
        "pn_key_id": "pn-a",
        "layout_key_id": "layout-a",
        "hit_margin_ms": 5.0,
    }
    identity = experiment_config_id(**base)

    for field, changed in (
        ("message_seed", 43),
        ("pn_key_id", "pn-b"),
        ("layout_key_id", "layout-b"),
        ("hit_margin_ms", 6.0),
    ):
        variant = dict(base)
        variant[field] = changed
        if experiment_config_id(**variant) == identity:
            raise AssertionError(f"Config identity ignoriše polje {field}.")

    with tempfile.TemporaryDirectory(prefix="layout_resume_check_") as name:
        directory = Path(name)
        results = directory / "results.csv"
        features = directory / "features.csv"

        ok_row = {
            **_base_row(identity),
            "status": "OK",
        }
        clean = {
            **_base_row(identity),
            "label": 0,
        }
        stego = {
            **_base_row(identity),
            "label": 1,
        }

        replace_config_rows(results, RESULT_FIELDS, identity, [ok_row])
        replace_config_rows(features, FEATURE_FIELDS, identity, [clean, stego])
        if completed_result_keys(results) != {identity}:
            raise AssertionError("Validan result nije prepoznat za resume.")
        if completed_feature_keys(features) != {identity}:
            raise AssertionError("Tačan clean/stego par nije prepoznat.")

        # Replacing the same config must not append duplicate rows.
        replace_config_rows(features, FEATURE_FIELDS, identity, [clean, stego])
        if completed_feature_keys(features) != {identity}:
            raise AssertionError("Atomic upsert je proizveo duplikate.")

        # An incomplete pair must never be considered complete.
        replace_config_rows(features, FEATURE_FIELDS, identity, [clean])
        if completed_feature_keys(features):
            raise AssertionError("Nepotpun feature par je prihvaćen za resume.")

    print("LAYOUT EXPERIMENT RESUME CHECK OK")


if __name__ == "__main__":
    main()
