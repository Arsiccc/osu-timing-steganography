"""Prepare the exact 80-replay BER pilot cohort for the fine payload sweep.

Run from project root:

    python3 -m scripts.tools.prepare_fine_payload_pilot

The script joins the previously saved pilot replay selection with the current
clean metadata and writes a filtered metadata CSV for run_payload_sweep.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from osu_stego.paths import METADATA_DIR, PILOT_BER_RESULTS_DIR, RESULTS_DIR


EXPECTED_TOTAL = 80
EXPECTED_PER_CATEGORY = 20


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the exact 80-replay pilot cohort."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=METADATA_DIR / "results_v3_clean.csv",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=PILOT_BER_RESULTS_DIR / "pilot_ber_selected_replays.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "payload" / "fine_pilot_selected_replays.csv",
    )
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    selection = pd.read_csv(args.selection)

    if "replay_file" not in metadata.columns:
        raise ValueError("Metadata nema kolonu 'replay_file'.")
    if "replay_file" not in selection.columns:
        raise ValueError("Selection nema kolonu 'replay_file'.")

    replay_files = selection["replay_file"].astype(str)

    if replay_files.duplicated().any():
        duplicates = replay_files[replay_files.duplicated()].tolist()
        raise ValueError(f"Selection sadrži duplikate: {duplicates[:5]}")

    selected = metadata[
        metadata["replay_file"].astype(str).isin(set(replay_files))
    ].copy()

    missing = sorted(
        set(replay_files)
        - set(selected["replay_file"].astype(str))
    )
    if missing:
        raise ValueError(
            f"{len(missing)} replay-eva iz pilot selection nije u clean metadata. "
            f"Primer: {missing[:5]}"
        )

    if len(selected) != EXPECTED_TOTAL:
        raise ValueError(
            f"Očekivano {EXPECTED_TOTAL} replay-eva, dobijeno {len(selected)}."
        )

    category_column = (
        "category"
        if "category" in selected.columns
        else "skill_category"
        if "skill_category" in selected.columns
        else None
    )

    if category_column is not None:
        counts = selected[category_column].value_counts()
        bad = counts[counts != EXPECTED_PER_CATEGORY]
        if not bad.empty:
            raise ValueError(
                "Pilot više nije balansiran 20 po kategoriji: "
                f"{counts.to_dict()}"
            )
        print("Po kategoriji:", counts.to_dict())

    # Preserve the exact order from the saved pilot selection.
    order = {
        replay_file: index
        for index, replay_file in enumerate(replay_files)
    }
    selected["_pilot_order"] = (
        selected["replay_file"].astype(str).map(order)
    )
    selected = (
        selected.sort_values("_pilot_order")
        .drop(columns="_pilot_order")
        .reset_index(drop=True)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)

    print(f"Replay-eva: {len(selected)}")
    print(f"Sačuvano:   {args.output}")


if __name__ == "__main__":
    main()
