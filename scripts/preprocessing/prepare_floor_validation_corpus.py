"""Deterministically prepare a new-map floor-policy validation corpus.

Selection uses only source identity/header metadata and pre-existing technical
quality rules. It never inspects residuals, capacity, stego outcomes, or
detector features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import osrparse
import pandas as pd

from archive.dataset_build.build_balanced_dataset import read_osr_header_fast
from osu_stego.parsing.beatmap_loader import load_hit_object_times
from osu_stego.paths import DATA_DIR, RESULTS_DIR


EXPERIMENT_VERSION = "one-codeword-floor-validation-v1"
SELECTION_SEED = "floor-validation-new-corpus-v1"
MAX_REPLAY_VERSION = 20220930
TARGET_MAPS = 20
CALIBRATION_REPLAYS_PER_MAP = 12
PRIMARY_REPLAYS_PER_MAP = 15
DEFAULT_SOURCE_ARCHIVE = Path.home() / "Downloads" / "archive"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("|".join((SELECTION_SEED, *parts)).encode()).hexdigest()


def historical_exclusions(results_root: Path, data_root: Path) -> tuple[pd.DataFrame, set[str], set[str], set[str]]:
    map_sources: dict[str, set[str]] = defaultdict(set)
    replay_sources: dict[str, set[str]] = defaultdict(set)
    user_sources: dict[str, set[str]] = defaultdict(set)
    for path in sorted(results_root.rglob("*.csv")):
        try:
            columns = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        wanted = [name for name in ("beatmap_hash", "replay_file", "player", "username") if name in columns]
        if not wanted:
            continue
        try:
            frame = pd.read_csv(path, usecols=wanted)
        except Exception:
            continue
        source = str(path)
        if "beatmap_hash" in frame:
            for value in frame.beatmap_hash.dropna().astype(str).unique():
                map_sources[value].add(source)
        if "replay_file" in frame:
            for value in frame.replay_file.dropna().astype(str).unique():
                replay_sources[value].add(source)
        for column in ("player", "username"):
            if column in frame:
                for value in frame[column].dropna().astype(str).unique():
                    user_sources[value.casefold()].add(source)
    for path in sorted((data_root / "dataset").glob("*/osr/*.osr")) + sorted((data_root / "excluded_bad_maps").glob("*/osr/*.osr")):
        try:
            mode, _, beatmap_hash, username = read_osr_header_fast(path)
        except Exception:
            continue
        if mode == 0:
            map_sources[beatmap_hash].add(str(path.parent.parent.parent))
            replay_sources[path.name].add(str(path.parent.parent.parent))
            user_sources[username.casefold()].add(str(path.parent.parent.parent))
    rows = []
    for entity_type, values in (("beatmap_hash", map_sources), ("replay_file", replay_sources), ("username_casefold", user_sources)):
        for identifier, sources in sorted(values.items()):
            rows.append({"entity_type": entity_type, "identifier": identifier,
                         "source_count": len(sources), "sources": json.dumps(sorted(sources))})
    return pd.DataFrame(rows), set(map_sources), set(replay_sources), set(user_sources)


def load_source_index(path: Path, excluded_maps: set[str], excluded_users: set[str]) -> pd.DataFrame:
    columns = ["replayHash", "beatmapHash", "playerName", "performance-IsFail"]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame = frame.dropna(subset=["replayHash", "beatmapHash", "playerName"]).copy()
    frame["replayHash"] = frame.replayHash.astype(str)
    frame["beatmapHash"] = frame.beatmapHash.astype(str)
    frame["playerName"] = frame.playerName.astype(str)
    frame["username_casefold"] = frame.playerName.str.casefold()
    valid_hex = frame.replayHash.str.fullmatch(r"[0-9a-f]{32}") & frame.beatmapHash.str.fullmatch(r"[0-9a-f]{32}")
    not_fail = ~frame["performance-IsFail"].astype(str).str.casefold().isin(("true", "1"))
    frame = frame[valid_hex & not_fail & ~frame.beatmapHash.isin(excluded_maps)
                  & ~frame.username_casefold.isin(excluded_users)]
    frame = frame.drop_duplicates("replayHash")
    return frame.sort_values(["beatmapHash", "replayHash"]).reset_index(drop=True)


def validate_candidate(source_replays: Path, source_beatmaps: Path, row: object) -> dict[str, object] | None:
    replay_path = source_replays / f"{row.replayHash}.osr"
    beatmap_path = source_beatmaps / f"{row.beatmapHash}.osu"
    if not replay_path.is_file() or not beatmap_path.is_file():
        return None
    try:
        mode, version, replay_map, username = read_osr_header_fast(replay_path)
        if mode != 0 or version > MAX_REPLAY_VERSION or replay_map != row.beatmapHash:
            return None
        if username.casefold() != row.username_casefold:
            return None
        if hashlib.md5(beatmap_path.read_bytes()).hexdigest() != row.beatmapHash:
            return None
        replay = osrparse.Replay.from_path(replay_path)
        if not replay.replay_data:
            return None
        notes = load_hit_object_times(str(beatmap_path))
        if len(notes) == 0:
            return None
    except Exception:
        return None
    return {
        "replay_file": replay_path.name, "replay_hash": row.replayHash,
        "beatmap_hash": row.beatmapHash, "username": username,
        "username_casefold": username.casefold(), "replay_version": version,
        "source_replay": str(replay_path), "source_beatmap": str(beatmap_path),
        "source_replay_sha256": sha256_file(replay_path),
        "source_beatmap_sha256": sha256_file(beatmap_path),
    }


def select_corpus(index: pd.DataFrame, source_replays: Path, source_beatmaps: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    needed = CALIBRATION_REPLAYS_PER_MAP + PRIMARY_REPLAYS_PER_MAP
    counts = index.groupby("beatmapHash").username_casefold.nunique()
    map_order = sorted(counts[counts >= needed].index, key=lambda value: stable_hash("map", value))
    used_users: set[str] = set()
    calibration = []
    primary = []
    stats = defaultdict(int)
    for beatmap_hash in map_order:
        candidates = index[index.beatmapHash == beatmap_hash].drop_duplicates("username_casefold").copy()
        candidates["order"] = candidates.replayHash.map(lambda value: stable_hash("replay", beatmap_hash, value))
        accepted = []
        for row in candidates.sort_values("order").itertuples(index=False):
            if row.username_casefold in used_users:
                stats["global_duplicate_user"] += 1
                continue
            item = validate_candidate(source_replays, source_beatmaps, row)
            if item is None:
                stats["technical_invalid"] += 1
                continue
            accepted.append(item)
            if len(accepted) == needed:
                break
        if len(accepted) < needed:
            stats["maps_insufficient_after_validation"] += 1
            continue
        for item in accepted:
            used_users.add(str(item["username_casefold"]))
        calibration.extend(accepted[:CALIBRATION_REPLAYS_PER_MAP])
        primary.extend(accepted[CALIBRATION_REPLAYS_PER_MAP:])
        if len(primary) == TARGET_MAPS * PRIMARY_REPLAYS_PER_MAP:
            break
    if len(primary) != TARGET_MAPS * PRIMARY_REPLAYS_PER_MAP:
        raise RuntimeError(f"Dostupno je samo {len(primary)} primary replay-a na {len(set(row['beatmap_hash'] for row in primary))} mapa.")
    return pd.DataFrame(calibration), pd.DataFrame(primary), dict(stats)


def copy_manifest(frame: pd.DataFrame, destination: Path) -> pd.DataFrame:
    (destination / "New" / "osr").mkdir(parents=True, exist_ok=True)
    (destination / "New" / "osu").mkdir(parents=True, exist_ok=True)
    rows = []
    for row in frame.itertuples(index=False):
        replay_target = destination / "New" / "osr" / row.replay_file
        beatmap_target = destination / "New" / "osu" / f"{row.beatmap_hash}.osu"
        if not replay_target.exists():
            shutil.copy2(row.source_replay, replay_target)
        if not beatmap_target.exists():
            shutil.copy2(row.source_beatmap, beatmap_target)
        item = row._asdict()
        item.update({"dataset_category": "New", "copied_replay": str(replay_target),
                     "copied_beatmap": str(beatmap_target),
                     "copied_replay_sha256": sha256_file(replay_target),
                     "copied_beatmap_sha256": sha256_file(beatmap_target)})
        if item["source_replay_sha256"] != item["copied_replay_sha256"] or item["source_beatmap_sha256"] != item["copied_beatmap_sha256"]:
            raise RuntimeError("Kopirani corpus fajl nema isti SHA-256 kao izvor.")
        rows.append(item)
    return pd.DataFrame(rows)


def write_once(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"Pre-registration fajl već postoji sa drugim sadržajem: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_ARCHIVE / "index.csv")
    parser.add_argument("--source-replays", type=Path, default=DEFAULT_SOURCE_ARCHIVE / "replays" / "osr")
    parser.add_argument("--source-beatmaps", type=Path, default=DEFAULT_SOURCE_ARCHIVE / "beatmaps")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "one_codeword_floor_validation_v1")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR / "one_codeword_floor_validation_v1")
    args = parser.parse_args()
    exclusions, old_maps, old_replays, old_users = historical_exclusions(RESULTS_DIR, DATA_DIR)
    if len(old_maps) < 34:
        raise RuntimeError("Historical exclusion audit nije pronašao svih najmanje 34 mape.")
    index = load_source_index(args.source_index, old_maps, old_users)
    calibration, primary, stats = select_corpus(index, args.source_replays, args.source_beatmaps)
    if set(primary.beatmap_hash) & old_maps or set(calibration.beatmap_hash) & old_maps:
        raise RuntimeError("Nova corpus selekcija preklapa historical beatmap hash.")
    if set(primary.username_casefold) & old_users or set(calibration.username_casefold) & old_users:
        raise RuntimeError("Nova corpus selekcija preklapa historical username.")
    if set(primary.replay_file) & old_replays or set(calibration.replay_file) & old_replays:
        raise RuntimeError("Nova corpus selekcija preklapa historical replay filename.")
    if set(primary.replay_file) & set(calibration.replay_file):
        raise RuntimeError("Calibration i primary replay skupovi se preklapaju.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exclusions.to_csv(args.output_dir / "exclusions.csv", index=False)
    calibration_copied = copy_manifest(calibration, args.data_dir / "calibration")
    primary_copied = copy_manifest(primary, args.data_dir / "primary")
    calibration_copied.to_csv(args.output_dir / "calibration_manifest.csv", index=False)
    primary_copied.to_csv(args.output_dir / "corpus_manifest.csv", index=False)
    selection = {
        "experiment_version": EXPERIMENT_VERSION,
        "scientific_status": "preregistered_before_capacity_and_stego_outcomes",
        "source": "local Kaggle o!rdr archive and beatmap mirror",
        "source_index": str(args.source_index), "source_index_sha256": sha256_file(args.source_index),
        "selection_seed": SELECTION_SEED, "selection_basis": "deterministic hash ordering of maps and replay hashes",
        "allowed_fields": ["replayHash", "beatmapHash", "playerName", "performance-IsFail", "replay header mode/version"],
        "forbidden_fields_not_inspected": ["capacity", "BER", "AUC", "confidence", "rematching", "integrity outcome", "timing features"],
        "filters": {"osu_standard": True, "performance_is_fail": False,
                    "max_replay_version": MAX_REPLAY_VERSION, "valid_full_parse": True,
                    "matching_beatmap_md5": True, "nonempty_replay_data": True,
                    "nonempty_hit_objects": True, "historical_map_overlap": 0,
                    "historical_username_overlap": 0, "global_duplicate_users": 0},
        "map_sampling": "hash-ranked among maps with at least 27 distinct eligible source usernames; accept first 20 passing technical validation",
        "within_map_sampling": "hash-ranked replay IDs; first 12 calibration, next 15 primary",
        "target_maps": TARGET_MAPS, "primary_replays_per_map": PRIMARY_REPLAYS_PER_MAP,
        "calibration_replays_per_map": CALIBRATION_REPLAYS_PER_MAP,
        "primary_replays": len(primary), "calibration_replays": len(calibration),
        "new_maps": sorted(primary.beatmap_hash.unique().tolist()),
        "historical_excluded_maps": len(old_maps), "historical_excluded_replay_ids": len(old_replays),
        "historical_excluded_usernames": len(old_users), "selection_diagnostics": stats,
    }
    write_once(args.output_dir / "new_corpus_selection.json", json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(f"new primary corpus: {len(primary)} replays, {primary.beatmap_hash.nunique()} maps")
    print(f"calibration corpus: {len(calibration)} replays, {calibration.beatmap_hash.nunique()} maps")
    print("historical map/user/replay overlap: 0/0/0")


if __name__ == "__main__":
    main()
