"""Bit-level diagnostics for physical spread-spectrum decoding."""

from __future__ import annotations

import numpy as np

from osu_stego.parsing.osr_writer import make_chronology_safe_shifts
from osu_stego.stego.payload_layout import (
    embed_message_with_layout,
    message_correlations_with_layout,
    select_payload_blocks,
)
from osu_stego.stego.pn_sequence import generate_pn_sequence


def bit_level_diagnostics(
    *,
    context,
    roundtrip_residuals: np.ndarray,
    message: np.ndarray,
    alpha: float,
    n_frames_per_bit: int,
    pn_key: str | int,
    layout_key: str | int,
    hit_margin_ms: float,
) -> tuple[list[dict[str, float | int]], dict[str, int]]:
    """Reconstruct decoder scores and writer/carrier state for every bit."""
    original = np.asarray(context.original_residuals, dtype=np.float64)
    roundtrip = np.asarray(roundtrip_residuals, dtype=np.float64)
    message = np.asarray(message, dtype=np.int8)
    note_frames = np.asarray(context.note_frame_indices, dtype=np.int64)
    blocks = select_payload_blocks(
        len(original), n_frames_per_bit, len(message), layout_key, "distributed"
    )
    target = embed_message_with_layout(
        original,
        message,
        pn_key,
        layout_key,
        alpha,
        n_frames_per_bit,
        "distributed",
        quantize=True,
    )
    base_usable = (note_frames != -1) & ~np.isnan(original) & ~np.isnan(target)
    requested_shifts = np.zeros(len(original), dtype=np.int64)
    requested_shifts[base_usable] = np.rint(
        target[base_usable] - original[base_usable]
    ).astype(np.int64)
    originally_requested = base_usable & (requested_shifts != 0)
    limit = max(0.0, float(context.hit_window_ms) - hit_margin_ms)
    hit_safe = np.zeros(len(original), dtype=bool)
    hit_safe[base_usable] = np.abs(
        original[base_usable] + requested_shifts[base_usable]
    ) <= limit
    hit_drop_mask = originally_requested & ~hit_safe
    requested_shifts[hit_drop_mask] = 0
    target_positions = np.flatnonzero(base_usable & (requested_shifts != 0))
    safe_shifts, _ = make_chronology_safe_shifts(
        context.replay,
        note_frames[target_positions],
        requested_shifts[target_positions],
    )
    active_positions = target_positions[safe_shifts != 0]
    chronology_drop_positions = target_positions[safe_shifts == 0]
    correlations = message_correlations_with_layout(
        roundtrip,
        pn_key,
        layout_key,
        n_frames_per_bit,
        len(message),
        "distributed",
    )
    pn = generate_pn_sequence(pn_key, len(message) * n_frames_per_bit)
    capacity = len(original) // n_frames_per_bit
    rows: list[dict[str, float | int]] = []
    for bit_index, block_index in enumerate(blocks.tolist()):
        start = block_index * n_frames_per_bit
        end = start + n_frames_per_bit
        positions = np.arange(start, end, dtype=np.int64)
        clean_block = original[start:end]
        roundtrip_block = roundtrip[start:end]
        clean_valid = ~np.isnan(clean_block)
        roundtrip_valid = ~np.isnan(roundtrip_block)
        block_pn = pn[
            bit_index * n_frames_per_bit : (bit_index + 1) * n_frames_per_bit
        ].astype(np.float64)
        clean_values = clean_block[clean_valid]
        decoded = 1 if correlations[bit_index] >= 0.0 else -1
        requested_count = int(np.sum(originally_requested[positions]))
        active_count = int(np.isin(positions, active_positions).sum())
        hit_drops = int(np.sum(hit_drop_mask[positions]))
        chronology_drops = int(np.isin(positions, chronology_drop_positions).sum())
        rows.append(
            {
                "bit_index": bit_index,
                "true_bit": int(message[bit_index]),
                "decoded_bit": decoded,
                "is_error": int(decoded != int(message[bit_index])),
                "correlation": float(correlations[bit_index]),
                "abs_correlation": float(abs(correlations[bit_index])),
                "normalized_abs_correlation": float(
                    abs(correlations[bit_index])
                    / (alpha * max(1, int(np.sum(roundtrip_valid))))
                ),
                "block_index": int(block_index),
                "block_start_note": start,
                "block_end_note_exclusive": end,
                "block_center_fraction": float((start + end) / (2.0 * len(original))),
                "block_quarter": int(min(3, (4 * block_index) // capacity) + 1),
                "roundtrip_valid_carriers": int(np.sum(roundtrip_valid)),
                "requested_carriers": requested_count,
                "active_carriers": active_count,
                "dropped_carriers": requested_count - active_count,
                "hit_window_drops": hit_drops,
                "chronology_drops": chronology_drops,
                "new_unmatched_events": int(np.sum(clean_valid & ~roundtrip_valid)),
                "new_matched_events": int(np.sum(~clean_valid & roundtrip_valid)),
                "rematching_changes": int(np.sum(clean_valid != roundtrip_valid)),
                "clean_valid_residuals": int(len(clean_values)),
                "clean_mean_residual": float(np.mean(clean_values)) if len(clean_values) else np.nan,
                "clean_variance": float(np.var(clean_values)) if len(clean_values) else np.nan,
                "clean_mean_absolute_residual": float(np.mean(np.abs(clean_values))) if len(clean_values) else np.nan,
                "clean_pn_correlation": float(
                    np.sum(clean_block[clean_valid] * block_pn[clean_valid])
                ),
            }
        )
    totals = {
        "requested_carriers": int(np.sum(originally_requested)),
        "active_carriers": int(len(active_positions)),
        "dropped_for_hit_window": int(np.sum(hit_drop_mask)),
        "dropped_for_chronology": int(len(chronology_drop_positions)),
        "new_unmatched_events": int(np.sum(~np.isnan(original) & np.isnan(roundtrip))),
        "new_matched_events": int(np.sum(np.isnan(original) & ~np.isnan(roundtrip))),
        "changed_match_status_total": int(
            np.sum(np.isnan(original) != np.isnan(roundtrip))
        ),
    }
    return rows, totals


def error_runs(errors: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, length)`` for each consecutive run of one-valued errors."""
    values = np.asarray(errors, dtype=np.int8)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values.tolist() + [0]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - start))
            start = None
    return runs
