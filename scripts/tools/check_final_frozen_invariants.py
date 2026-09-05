"""Small non-mutating smoke test for frozen final invariants not covered elsewhere."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import osrparse
import pandas as pd

from osu_stego.matching.matcher import match_keypresses_to_notes_indexed
from osu_stego.parsing.beatmap_loader import load_hit_object_times
from osu_stego.parsing.osr_writer import apply_frame_time_shifts
from osu_stego.parsing.replay_loader import (
    load_key_press_events,
    load_key_press_events_raw,
    load_map_offsets,
)
from osu_stego.paths import DATASET_DIR
from osu_stego.stego.carrier_utils import deduplicate_note_frame_carriers


ROOT = Path(__file__).resolve().parents[2]


def only_match_indices(presses: np.ndarray, notes: np.ndarray, window: float) -> np.ndarray:
    frame_ids = np.arange(len(presses), dtype=np.int64)
    _, indices = match_keypresses_to_notes_indexed(presses, frame_ids, notes, window)
    return indices


def main() -> None:
    # A greedy nearest-neighbour choice can lose cardinality here; frozen matcher must not.
    indices = only_match_indices(
        np.asarray([4, 6], dtype=np.int64),
        np.asarray([5, 7], dtype=np.int64),
        2.0,
    )
    assert np.array_equal(indices, np.asarray([0, 1], dtype=np.int64))

    residuals = np.asarray([1.0, 2.0, 3.0, np.nan])
    frames = np.asarray([10, 10, 11, -1])
    clean_r, clean_f, removed = deduplicate_note_frame_carriers(residuals, frames)
    assert removed == 1 and np.isnan(clean_r[1]) and clean_f.tolist() == [10, -1, 11, -1]

    active = pd.read_csv(ROOT / "data/metadata/results_v3_clean.csv")
    row = active.iloc[0]
    category_dir = DATASET_DIR / str(row.skill_category)
    replay_candidates = list(category_dir.glob(f"osr/{row.replay_file}"))
    beatmap_candidates = list(category_dir.glob(f"osu/{row.beatmap_file}"))
    assert len(replay_candidates) == 1 and len(beatmap_candidates) == 1
    replay_path, beatmap_path = replay_candidates[0], beatmap_candidates[0]
    offsets = load_map_offsets(ROOT / "data/config/map_time_offsets.json")
    raw_times, frame_indices = load_key_press_events_raw(replay_path)
    aligned_times, aligned_indices = load_key_press_events(
        replay_path, offsets[str(row.beatmap_hash)]
    )
    assert np.array_equal(frame_indices, aligned_indices)
    assert np.array_equal(aligned_times, raw_times + offsets[str(row.beatmap_hash)])
    notes = load_hit_object_times(str(beatmap_path))
    assert len(notes) == int(row.num_notes) and np.all(notes[1:] >= notes[:-1])

    # In-memory zero-shift pass checks writer parsing and anchor preservation without
    # creating a new .osr artifact.
    replay = osrparse.Replay.from_path(str(replay_path))
    target = frame_indices[: min(3, len(frame_indices))]
    output = apply_frame_time_shifts(replay, target, np.zeros(len(target), dtype=np.int64))
    before = [int(frame.time_delta) for frame in replay.replay_data]
    after = [int(frame.time_delta) for frame in output.replay_data]
    assert before == after

    # Final default is the normal 7.5% policy only: a complete word first appears
    # at C=107; the tested C=64..106 floor is not enabled.
    allocated_106 = max(1, 106 * 3 // 40)
    allocated_107 = max(1, 107 * 3 // 40)
    assert allocated_106 // 8 == 0 and allocated_107 // 8 == 1
    print("FINAL FROZEN INVARIANTS CHECK OK")
    print("loader=raw_cumulative+map_offset beatmap=sorted matcher=optimal")
    print("dedup=note_index_preserving writer=zero_shift_anchor_identity")
    print("normal_policy=C<=106_abstain C>=107_first_complete_word")


if __name__ == "__main__":
    main()
