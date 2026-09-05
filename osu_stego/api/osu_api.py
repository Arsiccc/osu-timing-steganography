"""Download osu!standard replays by sampling ranked players.

The rankings endpoint is used because beatmap leaderboards undersample lower
rank groups. Endpoint pagination at very deep ranks remains an empirical API
limitation; the collection summary reports counters needed to diagnose it.
"""

import os
import time
import requests

CLIENT_ID = os.getenv("OSU_CLIENT_ID")
CLIENT_SECRET = os.getenv("OSU_CLIENT_SECRET")

CATEGORIES = {
    "Elita": (1, 500),
    "Napredni": (501, 10000),
    "Prosecni": (10001, 100000),
    "Pocetni": (100001, 10000000),
}

# Empirical page-size assumption for the osu! API v2 rankings endpoint.
PAGE_SIZE = 50

# Sample pages across each rank interval instead of clustering near its start.
N_PAGES_PER_CATEGORY = 15

TARGET_PER_CATEGORY = 250
MODE = "osu"

def get_access_token():
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError(
            "Nedostaju OSU_CLIENT_ID i/ili OSU_CLIENT_SECRET environment promenljive."
        )
    url = "https://osu.ppy.sh/oauth/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
        "scope": "public",
    }
    res = requests.post(url, data=data, timeout=10)
    res.raise_for_status()
    return res.json()["access_token"]


def compute_sample_pages(low_rank: int, high_rank: int, n_pages: int) -> list[int]:
    """Bira n_pages ravnomerno raspoređenih stranica unutar [low_rank, high_rank]."""
    low_page = max(1, low_rank // PAGE_SIZE)
    high_page = max(low_page, high_rank // PAGE_SIZE)
    if high_page == low_page:
        return [low_page]
    step = max(1, (high_page - low_page) // n_pages)
    return list(range(low_page, high_page + 1, step))[:n_pages]


def fetch_rankings_page(page: int, debug_label: str = "") -> list[dict]:
    """Preuzima jednu stranicu globalnih rankings-a. Vraća listu 'entry' objekata
    (svaki bi trebalo da sadrži usera i njegov global_rank).
    """
    url = f"https://osu.ppy.sh/api/v2/rankings/{MODE}/performance"
    res = requests.get(url, headers=HEADERS, params={"page": page}, timeout=10)
    time.sleep(0.3)

    if res.status_code != 200:
        print(f"  [x] Rankings stranica {page} nije dostupna (status {res.status_code}).")
        return []

    data = res.json()

    return data.get("ranking", [])


def extract_user_id_and_rank(entry: dict):
    """Pokušava više uobičajenih šema za izvlačenje (user_id, global_rank) iz
    jednog rankings 'entry' objekta -- šema nije 100% potvrđena bez uživo testa.
    """
    user_id = None
    if "user" in entry and isinstance(entry["user"], dict):
        user_id = entry["user"].get("id")
    if user_id is None:
        user_id = entry.get("user_id")

    rank = entry.get("global_rank")
    if rank is None:
        rank = entry.get("rank")

    return user_id, rank


def fetch_user_best_scores(user_id: int, debug_label: str = "") -> list[dict]:
    """Preuzima najbolje score-ove datog korisnika."""
    url = f"https://osu.ppy.sh/api/v2/users/{user_id}/scores/best"
    res = requests.get(
        url, headers=HEADERS, params={"mode": MODE, "limit": 50}, timeout=10
    )
    time.sleep(0.3)

    if res.status_code != 200:
        return []

    data = res.json()

    return data if isinstance(data, list) else []


def score_has_replay(score: dict) -> bool:
    """Proverava da li score ima dostupan replay -- pokušava obe uobičajene
    varijante imena polja (has_replay / replay)."""
    if "has_replay" in score:
        return bool(score["has_replay"])
    if "replay" in score:
        return bool(score["replay"])
    return False


def download_file(url, file_path, custom_headers=None):
    if os.path.exists(file_path):
        return True
    try:
        res = requests.get(url, headers=custom_headers, timeout=10)
        if res.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(res.content)
            return True
        else:
            print(f"    [x] Download neuspešan (status {res.status_code}): {url}")
    except Exception as e:
        print(f"  [x] Greška pri preuzimanju: {e}")
    return False


def determine_category(rank):
    """Vraća kategoriju za dati rang, ili None ako je rang nepoznat/van opsega
    -- videti napomenu u apiv2_fixed.py o tome zašto NE fallback-ujemo na
    'Pocetni' kad je rang nepoznat."""
    if not rank:
        return None
    for cat, (low, high) in CATEGORIES.items():
        if low <= rank <= high:
            return cat
    return None


def count_existing_replays():
    existing = {}
    for cat in CATEGORIES:
        osr_dir = f"dataset/{cat}/osr"
        existing[cat] = (
            len([f for f in os.listdir(osr_dir) if f.endswith(".osr")])
            if os.path.isdir(osr_dir)
            else 0
        )
    return existing


def process_category(category: str, low_rank: int, high_rank: int, counts: dict) -> None:
    if counts[category] >= TARGET_PER_CATEGORY:
        print(f"[{category}] Već popunjeno ({counts[category]}/{TARGET_PER_CATEGORY}), preskačem.")
        return

    pages = compute_sample_pages(low_rank, high_rank, N_PAGES_PER_CATEGORY)
    print(f"[{category}] Uzorkujem stranice: {pages}")

    # Dijagnostički brojači -- razlikuju nekoliko mogućih uzroka "0 sačuvanih
    # replay-a": (a) /rankings ne vraća unose za tražene stranice, (b) unosi
    # postoje ali njihov rang ne odgovara traženoj kategoriji (pa se odbacuju
    # PRE poziva ka /scores/best), (c) /scores/best vraća prazno za date
    # korisnike, (d) score-ovi postoje ali nemaju replay=True.
    stats = {
        "entries_seen": 0,
        "entries_user_id_none": 0,
        "entries_category_mismatch": 0,
        "rank_min_seen": None,
        "rank_max_seen": None,
        "users_checked": 0,
        "users_with_zero_scores": 0,
        "total_scores_seen": 0,
        "scores_with_replay": 0,
    }

    for page in pages:
        if counts[category] >= TARGET_PER_CATEGORY:
            break

        entries = fetch_rankings_page(page, debug_label="rankings")
        if not entries:
            print(f"  [{category}] Stranica {page}: nema podataka.")
            continue

        print(f"  [{category}] Stranica {page}: {len(entries)} igrača.")

        for entry in entries:
            if counts[category] >= TARGET_PER_CATEGORY:
                break

            user_id, entry_rank = extract_user_id_and_rank(entry)

            stats["entries_seen"] += 1
            if entry_rank is not None:
                stats["rank_min_seen"] = (
                    entry_rank if stats["rank_min_seen"] is None else min(stats["rank_min_seen"], entry_rank)
                )
                stats["rank_max_seen"] = (
                    entry_rank if stats["rank_max_seen"] is None else max(stats["rank_max_seen"], entry_rank)
                )

            if user_id is None:
                stats["entries_user_id_none"] += 1
                continue

            # Koristimo rang direktno sa rankings stranice (već ga imamo,
            # nema potrebe za dodatnim pozivom kao u apiv2_fixed.py).
            actual_category = determine_category(entry_rank)
            if actual_category != category:
                # Rankings stranica ponekad ne odgovara tačno očekivanom
                # opsegu (npr. granica stranice ne poklapa se savršeno sa
                # granicom kategorije) -- ako igrač ipak pripada NEKOJ drugoj
                # kategoriji koja nije popunjena, iskoristimo ga tamo umesto
                # da ga odbacimo.
                if actual_category is None or counts.get(actual_category, TARGET_PER_CATEGORY) >= TARGET_PER_CATEGORY:
                    stats["entries_category_mismatch"] += 1
                    continue
                category_to_use = actual_category
            else:
                category_to_use = category

            scores = fetch_user_best_scores(user_id, debug_label="scores")

            stats["users_checked"] += 1
            if not scores:
                stats["users_with_zero_scores"] += 1
            stats["total_scores_seen"] += len(scores)
            stats["scores_with_replay"] += sum(1 for s in scores if score_has_replay(s))

            for score in scores:
                if not score_has_replay(score):
                    continue

                score_id = score.get("id")
                beatmap = score.get("beatmap") or {}
                beatmap_id = beatmap.get("id")
                if not score_id or not beatmap_id:
                    continue

                osr_url = f"https://osu.ppy.sh/api/v2/scores/osu/{score_id}/download"
                osu_url = f"https://osu.ppy.sh/osu/{beatmap_id}"
                osr_path = f"dataset/{category_to_use}/osr/{score_id}.osr"
                osu_path = f"dataset/{category_to_use}/osu/{beatmap_id}.osu"

                if download_file(osr_url, osr_path, custom_headers=HEADERS):
                    download_file(osu_url, osu_path)
                    counts[category_to_use] += 1
                    print(
                        f"    [+] [{category_to_use}] Sačuvan {score_id}.osr "
                        f"(user={user_id}, rang=#{entry_rank}) | "
                        f"{counts[category_to_use]}/{TARGET_PER_CATEGORY}"
                    )
                    break  # jedan replay po igracu je dovoljno, idemo na sledeceg

    print(
        f"  [DIJAGNOSTIKA {category}] entries_seen={stats['entries_seen']} | "
        f"user_id_none={stats['entries_user_id_none']} | "
        f"category_mismatch={stats['entries_category_mismatch']} | "
        f"rank_opseg_vidjen=[{stats['rank_min_seen']}, {stats['rank_max_seen']}] | "
        f"users_checked={stats['users_checked']} | "
        f"users_with_zero_scores={stats['users_with_zero_scores']} | "
        f"total_scores_seen={stats['total_scores_seen']} | "
        f"scores_with_replay={stats['scores_with_replay']}"
    )


def main():
    global HEADERS

    for cat in CATEGORIES:
        os.makedirs(f"dataset/{cat}/osu", exist_ok=True)
        os.makedirs(f"dataset/{cat}/osr", exist_ok=True)

    counts = count_existing_replays()
    print("[i] Zatečeno na disku:", counts)

    print("[*] Autentifikacija na osu! API v2...")
    token = get_access_token()
    HEADERS = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    print("[+] Token uspešno kreiran!")
    for category, (low, high) in CATEGORIES.items():
        process_category(category, low, high, counts)

    print("\n[+] Završeno! Stanje dataset-a:", counts)


if __name__ == "__main__":
    main()
