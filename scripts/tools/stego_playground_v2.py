"""Interaktivni playground za ručni osu! Spread Spectrum stego.

Pokretanje iz PROJECT ROOT-a:

    python3 -m scripts.tools.stego_playground info PATH.osr --n 16

    python3 -m scripts.tools.stego_playground encode PATH.osr \
        --message "Pozdrav iz Petnice" --alpha 10 --n 16

    python3 -m scripts.tools.stego_playground decode \
        results/playground/FILE_stego_a10_n16.osr

    python3 -m scripts.tools.stego_playground analyze \
        results/playground/FILE_stego_a10_n16.osr --compare-original

Namerno koristi isti:
- note-indexed encoder/decoder
- duplicate-carrier handling
- hit-window zaštitu
- chronology-safe writer
- write -> reload -> rematch put

kao glavni BER eksperiment.

Sidecar:
Uz generisani .osr pravi se FILE.osr.json. On NE čuva plaintext niti ključ.
Čuva samo parametre, dužinu poruke i SHA-256 fingerprint radi provere decode-a.

Stegoanalizator:
Za jednu datu alpha/N konfiguraciju trenira RF nad postojećim
steganalysis_features.csv, ali iz treninga izbacuje source replay ciljanog
fajla kada ga može identifikovati. Dobijeni predict_proba je demo score za
jedan fajl, NE zamena za GroupKFold ROC-AUC rezultat iz rada.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import osrparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from osu_stego.matching.matcher import hit_window_50_ms
from osu_stego.parsing.osr_writer import (
    make_chronology_safe_shifts,
    write_stego_replay,
)
from osu_stego.parsing.replay_loader import load_map_offsets
from osu_stego.paths import (
    CONFIG_DIR,
    DATASET_DIR,
    RESULTS_DIR,
    STEGANALYSIS_RESULTS_DIR,
)
from osu_stego.stego.decoder import extract_bipolar_message_indexed
from osu_stego.stego.encoder import embed_message_indexed
from osu_stego.stego.message_utils import (
    bipolar_to_bits,
    bits_to_bipolar,
    bits_to_bytes,
    bytes_to_bits,
)
from scripts.experiments.run_pilot_ber_sweep import (
    BeatmapData,
    build_beatmap_index,
    effective_od,
    load_residuals,
)
from scripts.experiments.run_steganalysis import (
    FEATURE_COLUMNS,
    residual_features,
)


DEFAULT_ALPHA = 10.0
DEFAULT_N = 16
DEFAULT_HIT_MARGIN_MS = 5.0

PLAYGROUND_DIR = RESULTS_DIR / "playground"
DEFAULT_FEATURES_PATH = (
    STEGANALYSIS_RESULTS_DIR
    / "steganalysis_features.csv"
)

DEFAULT_PREDICTIONS_PATH = (
    STEGANALYSIS_RESULTS_DIR
    / "steganalysis_oof_predictions.csv"
)


@dataclass
class FileContext:
    path: Path
    replay: osrparse.Replay
    beatmap_hash: str
    beatmap: BeatmapData
    offset_ms: int
    hit_window_ms: float
    residuals: np.ndarray
    note_frame_indices: np.ndarray
    duplicate_carriers_removed: int


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def key_fingerprint(key: str) -> str:
    return sha256_hex(
        key.encode("utf-8")
    )[:16]


def sidecar_path(osr_path: Path) -> Path:
    return Path(
        str(osr_path) + ".json"
    )


def load_sidecar(
    osr_path: Path,
) -> Optional[dict]:
    path = sidecar_path(osr_path)

    if not path.is_file():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def project_relative(
    path: Path,
) -> str:
    resolved = path.resolve()

    try:
        return str(
            resolved.relative_to(
                Path.cwd().resolve()
            )
        )
    except ValueError:
        return str(resolved)


def resolve_source_path(
    metadata: Optional[dict],
) -> Optional[Path]:
    if not metadata:
        return None

    value = metadata.get(
        "source_path"
    )
    if not value:
        return None

    candidate = Path(value)

    if not candidate.is_absolute():
        candidate = (
            Path.cwd()
            / candidate
        )

    return candidate


def read_key(
    supplied: Optional[str],
) -> str:
    if supplied is not None:
        if supplied == "":
            raise ValueError(
                "Ključ ne sme biti prazan."
            )
        return supplied

    value = getpass.getpass(
        "Tajni ključ: "
    )

    if not value:
        raise ValueError(
            "Ključ ne sme biti prazan."
        )

    return value


def build_context(
    osr_path: Path,
    beatmaps: dict[str, BeatmapData],
    offsets: dict[str, int],
) -> FileContext:
    if not osr_path.is_file():
        raise FileNotFoundError(
            osr_path
        )

    replay = osrparse.Replay.from_path(
        osr_path
    )
    beatmap_hash = str(
        replay.beatmap_hash
    )

    if beatmap_hash not in beatmaps:
        raise KeyError(
            "Ne mogu da pronađem odgovarajući .osu "
            f"za beatmap hash {beatmap_hash} u "
            f"{DATASET_DIR}."
        )

    if beatmap_hash not in offsets:
        raise KeyError(
            "Ne postoji map-level offset za "
            f"{beatmap_hash} u map_time_offsets.json."
        )

    beatmap = beatmaps[
        beatmap_hash
    ]
    offset_ms = int(
        offsets[beatmap_hash]
    )

    od = effective_od(
        beatmap.base_od,
        replay,
    )
    hit_window_ms = float(
        hit_window_50_ms(od)
    )

    (
        residuals,
        note_frame_indices,
        duplicates_removed,
    ) = load_residuals(
        osr_path,
        beatmap,
        offset_ms,
        hit_window_ms,
    )

    return FileContext(
        path=osr_path,
        replay=replay,
        beatmap_hash=beatmap_hash,
        beatmap=beatmap,
        offset_ms=offset_ms,
        hit_window_ms=hit_window_ms,
        residuals=residuals,
        note_frame_indices=(
            note_frame_indices
        ),
        duplicate_carriers_removed=(
            duplicates_removed
        ),
    )


def load_project_resources():
    beatmaps = build_beatmap_index(
        DATASET_DIR
    )
    offsets = load_map_offsets(
        CONFIG_DIR
        / "map_time_offsets.json"
    )

    return beatmaps, offsets


def capacity_bits(
    context: FileContext,
    n_frames_per_bit: int,
) -> int:
    return (
        len(context.residuals)
        // n_frames_per_bit
    )


def capacity_bytes(
    context: FileContext,
    n_frames_per_bit: int,
) -> int:
    return (
        capacity_bits(
            context,
            n_frames_per_bit,
        )
        // 8
    )


def encode_to_file(
    context: FileContext,
    message_bytes: bytes,
    key: str,
    alpha: float,
    n_frames_per_bit: int,
    output_path: Path,
    hit_margin_ms: float,
) -> dict:
    if not message_bytes:
        raise ValueError(
            "Poruka je prazna."
        )

    message_bits = bytes_to_bits(
        message_bytes
    )
    message_bipolar = bits_to_bipolar(
        message_bits
    )

    required = (
        len(message_bits)
        * n_frames_per_bit
    )

    if required > len(
        context.residuals
    ):
        raise ValueError(
            f"Poruka ima {len(message_bytes)} bajtova "
            f"({len(message_bits)} bita), što zahteva "
            f"{required} note-pozicija pri N={n_frames_per_bit}. "
            f"Replay ima {len(context.residuals)} nota. "
            f"Maksimum je približno "
            f"{capacity_bytes(context, n_frames_per_bit)} "
            "celih UTF-8 bajtova."
        )

    original = (
        context.residuals
    )
    note_frames = (
        context.note_frame_indices
    )

    target_stego = (
        embed_message_indexed(
            original,
            message_bipolar,
            key,
            alpha,
            n_frames_per_bit,
            quantize=True,
        )
    )

    requested_shifts = np.zeros(
        required,
        dtype=np.int64,
    )

    base_usable = (
        (
            note_frames[:required]
            != -1
        )
        & ~np.isnan(
            original[:required]
        )
        & ~np.isnan(
            target_stego[:required]
        )
    )

    requested_shifts[
        base_usable
    ] = np.rint(
        target_stego[:required][
            base_usable
        ]
        - original[:required][
            base_usable
        ]
    ).astype(np.int64)

    originally_requested = (
        base_usable
        & (requested_shifts != 0)
    )

    limit = max(
        0.0,
        context.hit_window_ms
        - hit_margin_ms,
    )

    hit_safe = np.zeros(
        required,
        dtype=bool,
    )

    hit_safe[
        base_usable
    ] = (
        np.abs(
            original[:required][
                base_usable
            ]
            + requested_shifts[
                base_usable
            ]
        )
        <= limit
    )

    drop_hit = int(
        np.sum(
            originally_requested
            & ~hit_safe
        )
    )

    requested_shifts[
        originally_requested
        & ~hit_safe
    ] = 0

    target_positions = (
        np.flatnonzero(
            base_usable
            & (
                requested_shifts
                != 0
            )
        )
    )

    target_frames = (
        note_frames[
            target_positions
        ]
    )
    requested_target_shifts = (
        requested_shifts[
            target_positions
        ]
    )

    (
        safe_target_shifts,
        drop_chronology,
    ) = make_chronology_safe_shifts(
        context.replay,
        target_frames,
        requested_target_shifts,
    )

    active_mask = (
        safe_target_shifts != 0
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_stego_replay(
        str(context.path),
        str(output_path),
        target_frames[
            active_mask
        ],
        safe_target_shifts[
            active_mask
        ],
    )

    requested_count = int(
        np.sum(
            originally_requested
        )
    )
    active_count = int(
        np.sum(active_mask)
    )

    metadata = {
        "format": (
            "osu-stego-playground-v1"
        ),
        "source_path": (
            project_relative(
                context.path
            )
        ),
        "source_replay_file": (
            context.path.name
        ),
        "beatmap_hash": (
            context.beatmap_hash
        ),
        "alpha": float(alpha),
        "n_frames_per_bit": int(
            n_frames_per_bit
        ),
        "message_num_bytes": int(
            len(message_bytes)
        ),
        "message_num_bits": int(
            len(message_bits)
        ),
        "message_sha256": (
            sha256_hex(
                message_bytes
            )
        ),
        "key_fingerprint": (
            key_fingerprint(key)
        ),
        "hit_margin_ms": float(
            hit_margin_ms
        ),
        "requested_carriers": (
            requested_count
        ),
        "active_carriers": (
            active_count
        ),
        "active_carrier_fraction": (
            active_count
            / requested_count
            if requested_count
            else 0.0
        ),
        "dropped_for_hit_window": (
            drop_hit
        ),
        "dropped_for_chronology": (
            int(drop_chronology)
        ),
    }

    with sidecar_path(
        output_path
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return metadata


def decode_file(
    context: FileContext,
    key: str,
    n_frames_per_bit: int,
    num_bytes: int,
) -> tuple[
    bytes,
    np.ndarray,
]:
    if num_bytes <= 0:
        raise ValueError(
            "Broj bajtova mora biti > 0."
        )

    num_bits = num_bytes * 8

    decoded_bipolar = (
        extract_bipolar_message_indexed(
            context.residuals,
            key,
            n_frames_per_bit,
            num_bits,
        )
    )

    decoded_bits = bipolar_to_bits(
        decoded_bipolar
    )
    decoded_bytes = bits_to_bytes(
        decoded_bits
    )

    return (
        decoded_bytes,
        decoded_bits,
    )


def print_context_info(
    context: FileContext,
    n_value: int,
) -> None:
    valid = int(
        np.sum(
            ~np.isnan(
                context.residuals
            )
        )
    )
    total = len(
        context.residuals
    )
    match_ratio = (
        valid / total
        if total
        else 0.0
    )

    print(
        f"Fajl:              "
        f"{context.path}"
    )
    print(
        f"Igrač:             "
        f"{context.replay.username}"
    )
    print(
        f"Beatmap hash:       "
        f"{context.beatmap_hash}"
    )
    print(
        f"Nota:               "
        f"{total}"
    )
    print(
        f"Matched:            "
        f"{valid}/{total} "
        f"({match_ratio:.2%})"
    )
    print(
        f"Duplicate carriers: "
        f"{context.duplicate_carriers_removed}"
    )
    print(
        f"Hit window:         "
        f"±{context.hit_window_ms:.1f} ms"
    )
    print(
        f"N:                  "
        f"{n_value}"
    )
    print(
        f"Nominalni kapacitet:"
        f" {capacity_bits(context, n_value)} bita"
        f" / {capacity_bytes(context, n_value)} celih bajtova"
    )


def train_single_file_analyzer(
    features_path: Path,
    alpha: float,
    n_value: int,
    exclude_replay_file: Optional[str],
) -> tuple[
    RandomForestClassifier,
    pd.DataFrame,
]:
    if not features_path.is_file():
        raise FileNotFoundError(
            f"Ne postoji stegoanalysis feature dataset: "
            f"{features_path}"
        )

    frame = pd.read_csv(
        features_path
    )

    selected = frame[
        (
            frame["alpha"]
            == float(alpha)
        )
        & (
            frame[
                "n_frames_per_bit"
            ]
            == int(n_value)
        )
    ].copy()

    if selected.empty:
        available = (
            frame[
                [
                    "alpha",
                    "n_frames_per_bit",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "alpha",
                    "n_frames_per_bit",
                ]
            )
        )

        raise ValueError(
            "Nema stegoanalysis trening podataka "
            f"za alpha={alpha:g}, N={n_value}.\n"
            "Dostupne konfiguracije:\n"
            + available.to_string(
                index=False
            )
        )

    if exclude_replay_file:
        selected = selected[
            selected["replay_file"]
            != exclude_replay_file
        ].copy()

    if (
        selected["label"].nunique()
        != 2
    ):
        raise ValueError(
            "Trening skup nema obe klase "
            "clean/stego."
        )

    X = selected[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float64
    )
    y = selected[
        "label"
    ].to_numpy(
        dtype=np.int8
    )

    if np.any(
        ~np.isfinite(X)
    ):
        raise ValueError(
            "Training feature matrica sadrži NaN/inf."
        )

    model = (
        RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    )

    model.fit(
        X,
        y,
    )

    return model, selected


def analyzer_score(
    model: RandomForestClassifier,
    residuals: np.ndarray,
) -> tuple[
    float,
    dict,
]:
    stats = residual_features(
        residuals
    )

    vector = np.array(
        [
            [
                float(
                    stats[
                        column
                    ]
                )
                for column
                in FEATURE_COLUMNS
            ]
        ],
        dtype=np.float64,
    )

    score = float(
        model.predict_proba(
            vector
        )[0, 1]
    )

    return score, stats


def command_info(
    args,
) -> None:
    beatmaps, offsets = (
        load_project_resources()
    )
    context = build_context(
        args.osr,
        beatmaps,
        offsets,
    )

    print_context_info(
        context,
        args.n,
    )


def command_encode(
    args,
) -> None:
    beatmaps, offsets = (
        load_project_resources()
    )
    context = build_context(
        args.osr,
        beatmaps,
        offsets,
    )

    message_text = (
        args.message
        if args.message is not None
        else input(
            "Poruka za skrivanje: "
        )
    )
    message_bytes = (
        message_text.encode(
            "utf-8"
        )
    )
    key = read_key(
        args.key
    )

    if args.output is None:
        PLAYGROUND_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = (
            PLAYGROUND_DIR
            / (
                f"{args.osr.stem}"
                f"_stego_a{args.alpha:g}"
                f"_n{args.n}.osr"
            )
        )
    else:
        output = args.output

    print_context_info(
        context,
        args.n,
    )
    print()

    metadata = encode_to_file(
        context=context,
        message_bytes=message_bytes,
        key=key,
        alpha=args.alpha,
        n_frames_per_bit=args.n,
        output_path=output,
        hit_margin_ms=(
            args.hit_margin_ms
        ),
    )

    # End-to-end sanity odmah nakon upisa.
    stego_context = (
        build_context(
            output,
            beatmaps,
            offsets,
        )
    )
    (
        decoded_bytes,
        decoded_bits,
    ) = decode_file(
        stego_context,
        key,
        args.n,
        len(message_bytes),
    )

    original_bits = bytes_to_bits(
        message_bytes
    )
    bit_errors = int(
        np.sum(
            original_bits
            != decoded_bits
        )
    )
    ber = (
        bit_errors
        / len(original_bits)
    )

    print(
        "\n=== ENCODE GOTOV ==="
    )
    print(
        f"Stego .osr:        {output}"
    )
    print(
        f"Sidecar:           "
        f"{sidecar_path(output)}"
    )
    print(
        f"Poruka:            "
        f"{message_text!r}"
    )
    print(
        f"UTF-8 bajtova:     "
        f"{len(message_bytes)}"
    )
    print(
        f"Bitova:            "
        f"{len(original_bits)}"
    )
    print(
        f"alpha / N:         "
        f"{args.alpha:g} / {args.n}"
    )
    print(
        f"Active carriers:   "
        f"{metadata['active_carriers']}/"
        f"{metadata['requested_carriers']} "
        f"({metadata['active_carrier_fraction']:.2%})"
    )
    print(
        f"Drop hit-window:   "
        f"{metadata['dropped_for_hit_window']}"
    )
    print(
        f"Drop chronology:   "
        f"{metadata['dropped_for_chronology']}"
    )
    print(
        f"Round-trip BER:    "
        f"{ber:.4f} "
        f"({bit_errors}/{len(original_bits)})"
    )
    print(
        "Round-trip tekst:  "
        + decoded_bytes.decode(
            "utf-8",
            errors="replace",
        )
    )


def command_decode(
    args,
) -> None:
    beatmaps, offsets = (
        load_project_resources()
    )
    context = build_context(
        args.osr,
        beatmaps,
        offsets,
    )

    metadata = load_sidecar(
        args.osr
    )

    n_value = args.n
    num_bytes = args.bytes

    if metadata:
        if n_value is None:
            n_value = int(
                metadata[
                    "n_frames_per_bit"
                ]
            )

        if num_bytes is None:
            num_bytes = int(
                metadata[
                    "message_num_bytes"
                ]
            )

    if n_value is None:
        raise ValueError(
            "Nije pronađen sidecar. "
            "Navedi --n."
        )

    if num_bytes is None:
        raise ValueError(
            "Nije pronađen sidecar. "
            "Navedi --bytes."
        )

    key = read_key(
        args.key
    )

    if metadata:
        expected_fingerprint = (
            metadata.get(
                "key_fingerprint"
            )
        )

        if (
            expected_fingerprint
            and key_fingerprint(key)
            != expected_fingerprint
        ):
            print(
                "UPOZORENJE: fingerprint ključa "
                "se ne poklapa sa sidecar-om."
            )

    (
        decoded_bytes,
        decoded_bits,
    ) = decode_file(
        context,
        key,
        int(n_value),
        int(num_bytes),
    )

    print(
        "=== DECODE ==="
    )
    print(
        f"Fajl:        {args.osr}"
    )
    print(
        f"N:           {n_value}"
    )
    print(
        f"Bajtova:     {num_bytes}"
    )
    print(
        f"Bitovi:      "
        f"{''.join(str(int(x)) for x in decoded_bits.tolist())}"
    )
    print(
        f"HEX:         "
        f"{decoded_bytes.hex(' ')}"
    )
    print(
        "UTF-8:       "
        + decoded_bytes.decode(
            "utf-8",
            errors="replace",
        )
    )

    if metadata:
        expected_hash = (
            metadata.get(
                "message_sha256"
            )
        )

        if expected_hash:
            exact = (
                sha256_hex(
                    decoded_bytes
                )
                == expected_hash
            )

            print(
                "Fingerprint: "
                + (
                    "TAČAN — poruka je dekodovana identično."
                    if exact
                    else
                    "NE ODGOVARA — najmanje jedan bit/bajt je pogrešan."
                )
            )



def load_reference_scores(
    predictions_path: Path,
    alpha: float,
    n_value: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load leakage-safe OOF scores for a configuration."""
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Ne postoji OOF predictions CSV: {predictions_path}"
        )

    frame = pd.read_csv(predictions_path)

    required = {
        "alpha",
        "n_frames_per_bit",
        "label",
        "oof_probability_stego",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"OOF predictions CSV nema kolone: {sorted(missing)}"
        )

    selected = frame[
        (frame["alpha"] == float(alpha))
        & (frame["n_frames_per_bit"] == int(n_value))
    ].copy()

    if "scope" in selected.columns:
        all_scope = selected[selected["scope"] == "ALL"]
        if not all_scope.empty:
            selected = all_scope

    clean_scores = selected.loc[
        selected["label"] == 0,
        "oof_probability_stego",
    ].to_numpy(dtype=np.float64)

    stego_scores = selected.loc[
        selected["label"] == 1,
        "oof_probability_stego",
    ].to_numpy(dtype=np.float64)

    clean_scores = clean_scores[np.isfinite(clean_scores)]
    stego_scores = stego_scores[np.isfinite(stego_scores)]

    if len(clean_scores) == 0 or len(stego_scores) == 0:
        raise ValueError(
            f"Nema OOF referentnih score-ova za alpha={alpha:g}, N={n_value}."
        )

    return clean_scores, stego_scores


def percentile_rank(
    values: np.ndarray,
    score: float,
) -> float:
    return float(np.mean(values <= score) * 100.0)


def score_reference_summary(
    clean_scores: np.ndarray,
    stego_scores: np.ndarray,
    score: float,
) -> dict:
    clean_q25, clean_median, clean_q75 = np.percentile(
        clean_scores,
        [25, 50, 75],
    )
    stego_q25, stego_median, stego_q75 = np.percentile(
        stego_scores,
        [25, 50, 75],
    )

    in_clean_iqr = clean_q25 <= score <= clean_q75
    in_stego_iqr = stego_q25 <= score <= stego_q75

    if in_clean_iqr and in_stego_iqr:
        interpretation = (
            "AMBIGUOUS — score je u centralnom rasponu obe klase."
        )
    elif in_clean_iqr:
        interpretation = (
            "CLEAN-LIKE — score je u centralnom rasponu clean klase, "
            "ali ovo nije sigurna klasifikacija."
        )
    elif in_stego_iqr:
        interpretation = (
            "STEGO-LIKE — score je u centralnom rasponu stego klase, "
            "ali ovo nije sigurna klasifikacija."
        )
    else:
        clean_distance = abs(score - clean_median)
        stego_distance = abs(score - stego_median)

        if abs(clean_distance - stego_distance) < 0.03:
            interpretation = (
                "AMBIGUOUS / ATYPIČAN — score nije u centralnom "
                "IQR-u nijedne klase."
            )
        elif clean_distance < stego_distance:
            interpretation = (
                "LEAN CLEAN — van centralnih IQR-ova, "
                "ali bliže clean medijani."
            )
        else:
            interpretation = (
                "LEAN STEGO — van centralnih IQR-ova, "
                "ali bliže stego medijani."
            )

    return {
        "clean_q25": float(clean_q25),
        "clean_median": float(clean_median),
        "clean_q75": float(clean_q75),
        "stego_q25": float(stego_q25),
        "stego_median": float(stego_median),
        "stego_q75": float(stego_q75),
        "clean_percentile": percentile_rank(clean_scores, score),
        "stego_percentile": percentile_rank(stego_scores, score),
        "interpretation": interpretation,
    }


def command_analyze(
    args,
) -> None:
    beatmaps, offsets = load_project_resources()
    context = build_context(
        args.osr,
        beatmaps,
        offsets,
    )

    metadata = load_sidecar(args.osr)

    alpha = args.alpha
    n_value = args.n

    if metadata:
        if alpha is None:
            alpha = float(metadata["alpha"])
        if n_value is None:
            n_value = int(metadata["n_frames_per_bit"])

    if alpha is None:
        alpha = DEFAULT_ALPHA
    if n_value is None:
        n_value = DEFAULT_N

    source_replay_file = None
    if metadata:
        source_replay_file = metadata.get("source_replay_file")

    if not source_replay_file:
        source_replay_file = args.osr.name

    model, training = train_single_file_analyzer(
        features_path=args.features,
        alpha=float(alpha),
        n_value=int(n_value),
        exclude_replay_file=source_replay_file,
    )

    score, stats = analyzer_score(
        model,
        context.residuals,
    )

    clean_reference, stego_reference = load_reference_scores(
        predictions_path=args.predictions,
        alpha=float(alpha),
        n_value=int(n_value),
    )

    reference = score_reference_summary(
        clean_scores=clean_reference,
        stego_scores=stego_reference,
        score=score,
    )

    print("=== SINGLE-FILE STEGOANALYSIS ===")
    print(f"Fajl:          {args.osr}")
    print(f"Config modela: alpha={alpha:g}, N={n_value}")
    print(f"Training rows: {len(training)}")
    print(f"Excluded:      {source_replay_file}")
    print()
    print(f"variance:      {stats['variance']:.4f}")
    print(f"kurtosis(ex):  {stats['excess_kurtosis']:.4f}")
    print(f"autocorr lag1: {stats['autocorrelation_lag1']:.4f}")
    print()
    print(f"RF stego score: {score:.4f}")
    print(f"Interpretacija: {reference['interpretation']}")

    print("\n=== OOF REFERENCE DISTRIBUTIONS ===")
    print(
        "Clean IQR:     "
        f"{reference['clean_q25']:.4f} — {reference['clean_q75']:.4f} "
        f"(median {reference['clean_median']:.4f})"
    )
    print(
        "Stego IQR:     "
        f"{reference['stego_q25']:.4f} — {reference['stego_q75']:.4f} "
        f"(median {reference['stego_median']:.4f})"
    )
    print(
        "Target percentile among clean: "
        f"{reference['clean_percentile']:.1f}%"
    )
    print(
        "Target percentile among stego: "
        f"{reference['stego_percentile']:.1f}%"
    )

    print(
        "\nNapomena: RF score nije kalibrisana verovatnoća da je fajl stego. "
        "Referentne distribucije su leakage-safe OOF score-ovi "
        "iz GroupKFold eksperimenta."
    )

    if args.compare_original:
        if not metadata:
            print(
                "\n--compare-original je preskočen: ovaj fajl nema "
                "playground sidecar, pa nema zabeleženog source replay-a."
            )
            return

        original_path = resolve_source_path(metadata)

        if original_path is None or not original_path.is_file():
            print(
                "\nNe mogu da pronađem originalni fajl iz sidecar-a "
                "za poređenje."
            )
            return

        original_context = build_context(
            original_path,
            beatmaps,
            offsets,
        )
        clean_score, clean_stats = analyzer_score(
            model,
            original_context.residuals,
        )

        original_reference = score_reference_summary(
            clean_scores=clean_reference,
            stego_scores=stego_reference,
            score=clean_score,
        )

        print("\n=== ORIGINAL vs TARGET ===")
        print(f"Original: {original_path}")
        print(f"RF score original: {clean_score:.4f}")
        print(f"  {original_reference['interpretation']}")
        print(f"RF score target:   {score:.4f}")
        print(f"  {reference['interpretation']}")
        print(f"Δ score:           {score - clean_score:+.4f}")
        print(
            f"Δ variance:        "
            f"{stats['variance'] - clean_stats['variance']:+.4f}"
        )
        print(
            f"Δ kurtosis:        "
            f"{stats['excess_kurtosis'] - clean_stats['excess_kurtosis']:+.4f}"
        )
        print(
            f"Δ autocorr lag1:   "
            f"{stats['autocorrelation_lag1'] - clean_stats['autocorrelation_lag1']:+.4f}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Ručni osu! Spread Spectrum "
            "encode/decode/steganalysis playground."
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    info = subparsers.add_parser(
        "info",
        help=(
            "Prikaži matching i nominalni kapacitet replay-a."
        ),
    )
    info.add_argument(
        "osr",
        type=Path,
    )
    info.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
    )
    info.set_defaults(
        handler=command_info
    )

    encode = (
        subparsers.add_parser(
            "encode",
            help=(
                "Ručno sakrij UTF-8 tekst u .osr."
            ),
        )
    )
    encode.add_argument(
        "osr",
        type=Path,
        help="Originalni clean .osr.",
    )
    encode.add_argument(
        "--message",
        type=str,
        default=None,
        help=(
            "Tekst poruke. Ako se izostavi, "
            "skripta će te pitati."
        ),
    )
    encode.add_argument(
        "--key",
        type=str,
        default=None,
        help=(
            "Tajni ključ. Ako se izostavi, "
            "biće interaktivno zatražen."
        ),
    )
    encode.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
    )
    encode.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
    )
    encode.add_argument(
        "--hit-margin-ms",
        type=float,
        default=DEFAULT_HIT_MARGIN_MS,
    )
    encode.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    encode.set_defaults(
        handler=command_encode
    )

    decode = (
        subparsers.add_parser(
            "decode",
            help=(
                "Dekoduj tekst iz stego .osr."
            ),
        )
    )
    decode.add_argument(
        "osr",
        type=Path,
    )
    decode.add_argument(
        "--key",
        type=str,
        default=None,
    )
    decode.add_argument(
        "--n",
        type=int,
        default=None,
        help=(
            "Ako postoji sidecar, nije potrebno."
        ),
    )
    decode.add_argument(
        "--bytes",
        type=int,
        default=None,
        help=(
            "Broj očekivanih bajtova. "
            "Ako postoji sidecar, nije potrebno."
        ),
    )
    decode.set_defaults(
        handler=command_decode
    )

    analyze = (
        subparsers.add_parser(
            "analyze",
            help=(
                "RF stego score za jedan .osr."
            ),
        )
    )
    analyze.add_argument(
        "osr",
        type=Path,
    )
    analyze.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Config trening skupa. "
            "Sidecar ga automatski popunjava."
        ),
    )
    analyze.add_argument(
        "--n",
        type=int,
        default=None,
    )
    analyze.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
    )
    analyze.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help=(
            "OOF reference score-ovi iz GroupKFold "
            "stegoanalysis eksperimenta."
        ),
    )
    analyze.add_argument(
        "--compare-original",
        action="store_true",
        help=(
            "Ako postoji playground sidecar, "
            "prikaži score originala i stego fajla."
        ),
    )
    analyze.set_defaults(
        handler=command_analyze
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "n"):
        n_value = getattr(
            args,
            "n",
            None,
        )
        if (
            n_value is not None
            and n_value <= 0
        ):
            parser.error(
                "N mora biti > 0."
            )

    if hasattr(
        args,
        "alpha",
    ):
        alpha = getattr(
            args,
            "alpha",
            None,
        )
        if (
            alpha is not None
            and alpha <= 0
        ):
            parser.error(
                "alpha mora biti > 0."
            )

    try:
        args.handler(args)
    except Exception as exc:
        parser.exit(
            1,
            f"GREŠKA: {exc}\n",
        )


if __name__ == "__main__":
    main()
