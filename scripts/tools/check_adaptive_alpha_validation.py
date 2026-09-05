"""Focused sanity checks before adaptive-alpha validation runs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import METADATA_DIR
from scripts.experiments.run_adaptive_alpha_comparison import ADAPTIVE_ALPHA
from scripts.experiments.run_adaptive_alpha_validation import (
    CATEGORY_AVAILABILITY,
    ENERGY_FIELDS,
    EXPERIMENT_VERSION,
    FEATURE_FIELDS,
    RESULT_FIELDS,
    completed_config_ids,
    deterministic_beatmap_partition,
    energy_summary,
)


def check_partition() -> None:
    metadata = pd.read_csv(METADATA_DIR / "performance_groups.csv")
    first = deterministic_beatmap_partition(metadata["beatmap_hash"], 0.30, 42)
    second = deterministic_beatmap_partition(metadata["beatmap_hash"], 0.30, 42)
    if not first.equals(second):
        raise AssertionError("Held-out beatmap partition nije determinističan.")
    development = set(first.loc[first["partition"] == "development", "beatmap_hash"])
    validation = set(first.loc[first["partition"] == "validation", "beatmap_hash"])
    if development & validation or development | validation != set(metadata["beatmap_hash"]):
        raise AssertionError("Held-out beatmap partition nije disjunktna/potpuna.")
    if len(validation) != 11 or len(development) != 23:
        raise AssertionError("Neočekivana 30% podela za 34 aktivne beatmape.")


def check_energy_math() -> None:
    frame = pd.DataFrame(
        {
            "replay_file": ["a", "b"],
            "requested_carriers": [3, 2],
            "active_carriers": [2, 1],
            "sum_squared_shift_ms2": [392.0, 225.0],
            "total_absolute_shift_ms": [28.0, 15.0],
            "dropped_for_hit_window": [1, 0],
            "dropped_for_chronology": [0, 1],
            "new_unmatched_events": [2, 1],
            "new_matched_events": [1, 0],
            "changed_match_status_total": [3, 1],
        }
    )
    summary = energy_summary(frame)
    if summary["sum_squared_shift_ms2"] != 617.0:
        raise AssertionError("Netačan zbir fizičke energije.")
    if not np.isclose(summary["rms_applied_shift_ms"], np.sqrt(617.0 / 3.0)):
        raise AssertionError("Netačan pooled RMS upisanih shiftova.")
    if summary["new_unmatched_events"] != 3:
        raise AssertionError("Novi unmatched događaji nisu sabrani kao događaji.")


def check_schema_and_resume() -> None:
    if len(RESULT_FIELDS) != len(set(RESULT_FIELDS)):
        raise AssertionError("Duplirana result kolona.")
    if len(FEATURE_FIELDS) != len(set(FEATURE_FIELDS)):
        raise AssertionError("Duplirana feature kolona.")
    if not set(ENERGY_FIELDS).issubset(RESULT_FIELDS):
        raise AssertionError("Result schema ne sadrži sva energy polja.")
    with tempfile.TemporaryDirectory(prefix="adaptive_validation_check_") as name:
        result_path = Path(name) / "results.csv"
        row = {field: "" for field in RESULT_FIELDS}
        row.update({"status": "OK", "config_id": "config-a"})
        pd.DataFrame([row], columns=RESULT_FIELDS).to_csv(result_path, index=False)
        if completed_config_ids(result_path, RESULT_FIELDS, False) != {"config-a"}:
            raise AssertionError("Resume nije prepoznao kompletan result red.")


def main() -> None:
    if ADAPTIVE_ALPHA != {
        "Very Poor": 20.0,
        "Poor": 16.0,
        "Good": 12.0,
        "Very Good": 10.0,
    }:
        raise AssertionError("Frozen adaptive policy je promenjena.")
    if "not-sender-local" not in CATEGORY_AVAILABILITY:
        raise AssertionError("Oracle category ograničenje nije eksplicitno zapisano.")
    check_partition()
    check_energy_math()
    check_schema_and_resume()
    print("ADAPTIVE ALPHA VALIDATION CHECK OK")
    print(f"experiment={EXPERIMENT_VERSION}")
    print("maps: development=23 validation=11")
    print(f"category={CATEGORY_AVAILABILITY}")


if __name__ == "__main__":
    main()
