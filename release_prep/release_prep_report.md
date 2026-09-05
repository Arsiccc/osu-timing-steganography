# Public release preparation report

## Outcome

- **SCIENTIFIC STATE: A — FROZEN AND VERIFIED**
- **PUBLICATION STATE: C — DATA/PRIVACY RIGHTS DECISION REQUIRED**

The versioned tooling defect is fixed and the proposed data-free public tree passes
its staged checks. Publication still requires a README, a user-selected code license,
and explicit decisions about third-party data, player-linked artifacts, and the
separate reproduction bundle.

## Finalizer correction

V1 used an unbounded `results/*/result_hashes.json` glob, so an existing canonical
final manifest became a ninth input whenever output was redirected. V2 retains the
v1 scientific generator but replaces that discovery with an exact eight-manifest
allowlist, bounds the historical language-audit inputs, defaults to a distinct v2
output, and refuses to overwrite `final_audit_v1`.

- Frozen v1 finalizer unchanged:
  `7c930100de258dc47bb03404df9053711462db9b626373adaba32b3d5516c337`.
- V2 finalizer:
  `9deed319aac4d4a2e4a4d5e478bc7bb77ceb44c0b5908763a825f1433f66e9fe`.
- Regression test: PASS.
- Eight approved manifests / 200 entries: PASS.
- Deterministic clean v2 rerun: PASS, byte-identical.
- V2 output manifest:
  `c3972411994019cca0d0f8e9dee3f231ed001cf93a7db7b43ea4a3912a2780a1`.

V1/v2 comparison found 47 identical and 12 expected tooling-different files. The
differences are seven generator metadata files, three reproduction/tooling documents,
the reporting-only language audit (18 internal-AGENTS rows removed), and the resulting
hash manifest. Unexpected scientific differences: 0. The generated system-spec hash
remains the frozen
`b99790fe0b39813e2c0a9777d7b98f05c69f2f3c5b43e3322a028bd78c0c7038`.

## Dependencies and paths

`pandas` and `requests` were added to `requirements.txt` because public code imports
them directly. Existing versions were not changed. `pandas` is required for analysis,
checks, and finalization; `requests` is required for the optional API acquisition
module and must not be supplied only transitively by `slider`.

Personal/machine-specific paths remaining in the exact proposed public set: **0**.
Eight frozen/non-canonical artifacts with such paths remain unchanged and local-only.

## Player data and redistribution

Direct player records in the proposed public set: **0**. The canonical dataset
registry exposes only aggregate counts and SHA-256 set digests.

The automated scan flagged 158 CSV files with player/replay/rank-like identifier
columns: 152 non-canonical result CSVs, 2 metadata CSVs, 3 archive CSVs, and the one
public-safe canonical registry containing only set hashes. Four raw replay-file groups
contain 3,347 `.osr` copies, and one rank cache has 3,472 keys. These counts include
duplicates/backups and heuristic column matches; they are an exposure inventory, not
a count of distinct people.

Seven third-party/raw-data groups have unresolved redistribution status and are marked
`UNKNOWN — MANUAL RIGHTS REVIEW`. No legal conclusion was inferred.

## Public and local sets

The proposed public set contains **208 files / 1,785,522 bytes (1.70 MiB)**. It keeps
source, checks, configs, the complete small canonical final audit, the release audit,
and release documentation. It excludes raw replay/beatmap corpora, caches, player-
linked metadata, backups, historical/intermediate result matrices, archive material,
cleanup working papers, and `AGENTS.md`.

Largest proposed public files at the time of this report:

| Bytes | Path |
|---:|---|
| 82,865 | `scripts/analysis/finalize_research_results.py` |
| 47,079 | `results/final_audit_v1/figures/figure_6_pn_robustness.png` |
| 44,867 | `results/final_audit_v1/figures/figure_2_reliability_detectability.png` |
| 41,746 | `scripts/experiments/run_layout_comparison.py` |
| 40,393 | `scripts/experiments/run_strong_steganalysis.py` |
| 39,427 | `scripts/experiments/run_adaptive_layout_strong.py` |
| 38,101 | `scripts/analysis/analyze_message_content_robustness.py` |
| 37,690 | `results/final_audit_v1/figures/figure_5_integrity_tradeoff.png` |
| 36,850 | `scripts/experiments/run_adaptive_alpha_validation.py` |
| 36,029 | `results/final_audit_v1/figures/figure_1_system_pipeline.png` |
| 34,709 | `scripts/tools/stego_playground_v2.py` |
| 34,075 | `results/final_audit_v1/figures/figure_3_position_signature.png` |
| 32,466 | `results/final_audit_v1/figures/figure_7_message_robustness.png` |
| 32,423 | `scripts/experiments/run_payload_sweep.py` |
| 32,150 | `scripts/experiments/run_pilot_ber_sweep.py` |
| 31,541 | `results/final_audit_v1/figures/figure_4_decoder_confidence.png` |
| 29,117 | `scripts/experiments/run_steganalysis.py` |
| 28,580 | `scripts/experiments/run_one_codeword_floor_validation.py` |
| 27,542 | `scripts/experiments/run_ecc_integrity_holdout.py` |
| 27,258 | `scripts/experiments/run_ecc_equal_payload.py` |

Largest local-only data/result groups:

| Bytes | Files | Group |
|---:|---:|---|
| 173,228,442 | 2,874 | `backups/` |
| 88,137,789 | 1,088 | `data/dataset/` |
| 72,025,622 | 33 | `results/host_aware_pn_selection_v1/` |
| 67,822,945 | 40 | `results/pn_key_robustness_phase2_v1/` |
| 48,557,171 | 13 | `results/adaptive_layout_strong_v1/` |
| 39,629,292 | 28 | `results/pn_key_robustness_v1/` |
| 38,061,727 | 580 | `data/one_codeword_floor_validation_v1/` |
| 30,552,534 | 28 | `results/payload/` |
| 27,714,128 | 7 | `results/decoder_error_analysis_v1/` |
| 21,886,800 | 16 | `results/sender_local_adaptive_v1/` |
| 14,088,907 | 23 | `results/classifier_family_robustness_v1/` |
| 10,990,860 | 33 | `results/one_codeword_floor_validation_v1/` |
| 7,804,137 | 8 | `results/ber/` |
| 7,364,980 | 10 | `results/adaptive_alpha_validation_v1/` |
| 4,820,261 | 13 | `results/strong_steganalysis_v1/` |
| 4,819,622 | 13 | `results/strong_steganalysis_v1_legacy_seed_bootstrap2000/` |
| 4,819,618 | 13 | `results/strong_steganalysis_v1_pre_legacy_seed_correction/` |
| 4,041,657 | 5 | `results/steganalysis/` |
| 3,892,797 | 4 | `results/adaptive_alpha/` |
| 3,586,263 | 51 | `data/excluded_bad_maps/` |

## Staged public test

The staged tree passed parse/compile (122 Python files), core imports, exhaustive
SECDED, payload/layout/PN determinism and noiseless decode, decoder zero-tie, floor
policy, synthetic integrity, v2 synthetic boundary, and all included manifest checks.
Static scans found no personal paths, email addresses, or probable secrets.

Full finalizer regeneration and corpus-backed checks are intentionally unavailable
without the excluded raw/frozen artifact bundle. This is documented rather than
treated as a silent success.

## AGENTS.md and remaining decisions

`AGENTS.md` is **LOCAL_ONLY** because it primarily contains agent workflow rules and
historical internal state. Its publishable frozen scientific details already exist
in the canonical audit. The final README must still migrate clear dataset membership
(949 active; 51 excluded without a corruption claim), frozen method identity, and
careful result/limitation wording from canonical sources—not prompt language.

Before publication the user must:

1. choose a software license;
2. approve the README, citation, and supported environment;
3. decide rights/privacy treatment for raw data and row-level artifacts;
4. decide whether/how to release the eight-input reproduction bundle;
5. document or replace the macOS-only `sips` figure-rendering dependency.
