"""Batch obrada .osr/.osu parova -> CSV sa statistikom.

Dva moda:
  1. FLAT (podrazumevani, run_batch): folder sadrži proizvoljnu mešavinu
     .osr i .osu fajlova u JEDNOM nivou. skill_category ostaje None -- za
     ovaj mod ne znamo kategoriju iz same strukture foldera.
  2. KATEGORIZOVAN (--categorized, run_batch_categorized): folder ima
     podstrukturu data_dir/<Kategorija>/{osr,osu}/ (kao što je prave
     apiv2*.py i import_kaggle_dataset.py skripte). skill_category se
     popunjava DIREKTNO iz imena foldera -- nije potreban API poziv, jer
     je kategorizacija već urađena kad su fajlovi raspoređeni na disk.

Svaki .osr fajl se uparuje sa odgovarajućim .osu fajlom PREKO MD5 HASH-A
(replay.beatmap_hash), nikad preko imena fajla -- imena fajlova nisu
pouzdana za uparivanje (videli smo slučaj gde je pogrešna difikultija bila
priložena uz replay).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR

import osrparse
import slider

from osu_stego.analysis.residual_stats import compute_residual_statistics
from osu_stego.parsing.beatmap_loader import load_hit_object_times
from osu_stego.matching.matcher import hit_window_50_ms, match_keypresses_to_notes
from osu_stego.parsing.replay_loader import load_key_press_times, load_map_offsets

DEFAULT_CATEGORIES = ["Elita", "Napredni", "Prosecni", "Pocetni"]


@dataclass
class ReplayResult:
    """Jedan red u izlaznom CSV-u -- rezultat uspešne obrade jednog replay-a."""

    replay_file: str
    beatmap_file: str
    player: str
    beatmap_hash: str
    overall_difficulty: float
    num_notes: int
    num_presses: int
    num_matched: int
    match_ratio: float
    mean: float
    std: float
    excess_kurtosis: float
    autocorrelation_lag1: float
    skill_category: str | None = None


def build_beatmap_hash_index(directory: Path) -> dict[str, Path]:
    """Računa MD5 hash svakog .osu fajla u folderu -> mapa hash -> putanja."""
    index: dict[str, Path] = {}
    for osu_path in directory.glob("*.osu"):
        md5 = hashlib.md5(osu_path.read_bytes()).hexdigest()
        index[md5] = osu_path
    return index


def process_single_replay(
    osr_path: Path,
    hash_index: dict[str, Path],
    min_matched: int,
    map_offsets: dict[str, int],
) -> ReplayResult:
    """Obrađuje jedan .osr fajl kroz ceo lanac parsiranje -> statistika.

    Raises
    ------
    ValueError
        Sa jasnim opisom razloga preskakanja (nema odgovarajuće mape,
        premalo uparenih nota, nula varijansa) -- poziva funkcija hvata
        ovo i loguje umesto da prekine ceo batch.
    """
    replay = osrparse.Replay.from_path(osr_path)

    beatmap_path = hash_index.get(replay.beatmap_hash)
    if beatmap_path is None:
        raise ValueError(
            f"Nema odgovarajućeg .osu fajla za beatmap_hash={replay.beatmap_hash}"
        )

    beatmap = slider.Beatmap.from_path(beatmap_path)
    od = beatmap.od()

    if replay.beatmap_hash not in map_offsets:
        raise ValueError(
            f"Nema kalibrisanog time offseta za beatmap_hash={replay.beatmap_hash}"
        )

    presses = load_key_press_times(
        str(osr_path),
        map_offsets[replay.beatmap_hash],
    )
    notes = load_hit_object_times(str(beatmap_path))

    window = hit_window_50_ms(od)
    matched_press, matched_note = match_keypresses_to_notes(presses, notes, window)

    if len(matched_press) < min_matched:
        raise ValueError(
            f"Samo {len(matched_press)} uparenih nota (< min_matched={min_matched}), "
            "statistika nije pouzdana."
        )

    residuals = matched_press - matched_note
    stats = compute_residual_statistics(residuals)

    return ReplayResult(
        replay_file=osr_path.name,
        beatmap_file=beatmap_path.name,
        player=replay.username,
        beatmap_hash=replay.beatmap_hash,
        overall_difficulty=od,
        num_notes=len(notes),
        num_presses=len(presses),
        num_matched=stats.n,
        match_ratio=stats.n / len(notes) if len(notes) > 0 else 0.0,
        mean=stats.mean,
        std=stats.std,
        excess_kurtosis=stats.excess_kurtosis,
        autocorrelation_lag1=stats.autocorrelation_lag1,
        skill_category=None,
    )


def _write_results(results: list[ReplayResult], output_csv: Path) -> None:
    if not results:
        print("Nema uspešno obrađenih replay-a -- CSV nije kreiran.")
        return
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"Upisano {len(results)} redova u {output_csv}")


def _print_skipped(skipped: list[tuple[str, str]]) -> None:
    if skipped:
        print(f"\nPreskočeno {len(skipped)} fajlova:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


def run_batch(
    data_dir: Path,
    output_csv: Path,
    min_matched: int,
    map_offsets: dict[str, int],
) -> None:
    """Obrađuje sve .osr fajlove u data_dir (FLAT -- .osr i .osu pomešani
    u istom folderu) i piše uspešne rezultate u CSV."""
    hash_index = build_beatmap_hash_index(data_dir)
    print(f"Indeksirano {len(hash_index)} .osu fajlova po MD5 hash-u.")

    osr_paths = sorted(data_dir.glob("*.osr"))
    print(f"Pronađeno {len(osr_paths)} .osr fajlova.")

    results: list[ReplayResult] = []
    skipped: list[tuple[str, str]] = []

    for osr_path in osr_paths:
        try:
            result = process_single_replay(osr_path, hash_index, min_matched, map_offsets)
            results.append(result)
        except Exception as exc:  # noqa: BLE001 -- namerno hvatamo sve da batch ne padne
            skipped.append((osr_path.name, str(exc)))

    _write_results(results, output_csv)
    _print_skipped(skipped)


def run_batch_categorized(
    data_dir: Path,
    output_csv: Path,
    min_matched: int,
    categories: list[str] | None = None,
    map_offsets: dict[str, int] | None = None,
) -> None:
    """Obrađuje strukturu data_dir/<Kategorija>/osr/*.osr + data_dir/<Kategorija>/osu/*.osu.

    Za razliku od run_batch (flat mod), ovde se hash-indeks .osu fajlova
    gradi ODVOJENO po svakoj kategoriji (jer apiv2.py/import_kaggle_dataset.py
    čuvaju zaseban osu/ podfolder po kategoriji, ponekad sa istim beatmap-ovima
    ponovljenim u više kategorija). skill_category kolona se popunjava
    DIREKTNO iz imena foldera -- nema potrebe za API pozivom, kategorizacija
    je već urađena kad su fajlovi raspoređeni na disk.
    """
    categories = categories or DEFAULT_CATEGORIES
    if map_offsets is None:
        raise ValueError("map_offsets je obavezan u categorized modu.")
    all_results: list[ReplayResult] = []
    all_skipped: list[tuple[str, str]] = []

    for category in categories:
        category_dir = data_dir / category
        osr_dir = category_dir / "osr"
        osu_dir = category_dir / "osu"

        if not osr_dir.is_dir() or not osu_dir.is_dir():
            print(f"[{category}] Preskačem -- nema osr/ i osu/ podfoldera u {category_dir}")
            continue

        hash_index = build_beatmap_hash_index(osu_dir)
        osr_paths = sorted(osr_dir.glob("*.osr"))
        print(f"[{category}] {len(osr_paths)} .osr fajlova, {len(hash_index)} .osu u indeksu.")

        for osr_path in osr_paths:
            try:
                result = process_single_replay(osr_path, hash_index, min_matched, map_offsets)
                result.skill_category = category
                all_results.append(result)
            except Exception as exc:  # noqa: BLE001
                all_skipped.append((f"[{category}] {osr_path.name}", str(exc)))

    _write_results(all_results, output_csv)
    _print_skipped(all_skipped)

    print("\nPregled po kategoriji:")
    for category in categories:
        count = sum(1 for r in all_results if r.skill_category == category)
        print(f"  {category}: {count} uspešno obrađenih")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch obrada .osr/.osu parova -> CSV sa timing-residual statistikom."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Folder sa .osr i .osu fajlovima."
    )
    parser.add_argument(
        "--output-csv", type=Path, required=True, help="Putanja izlaznog CSV fajla."
    )
    parser.add_argument(
        "--map-offsets",
        type=Path,
        default=CONFIG_DIR / "map_time_offsets.json",
        help="JSON sa fiksnim beatmap_hash -> time offset mapiranjem.",
    )
    parser.add_argument(
        "--min-matched",
        type=int,
        default=10,
        help="Minimalan broj uparenih nota da bi se replay smatrao validnim (default: 10).",
    )
    parser.add_argument(
        "--categorized",
        action="store_true",
        help=(
            "Koristi strukturu data-dir/<Kategorija>/osr/ + osu/ "
            "(Elita/Napredni/Prosecni/Pocetni) umesto flat foldera."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Koje kategorije obraditi u --categorized modu (default: sve četiri).",
    )
    args = parser.parse_args()

    map_offsets = load_map_offsets(args.map_offsets)
    print(f"Učitano {len(map_offsets)} kalibrisanih map time offseta.")

    if args.categorized:
        run_batch_categorized(
            args.data_dir,
            args.output_csv,
            args.min_matched,
            args.categories,
            map_offsets,
        )
    else:
        run_batch(
            args.data_dir,
            args.output_csv,
            args.min_matched,
            map_offsets,
        )


if __name__ == "__main__":
    main()