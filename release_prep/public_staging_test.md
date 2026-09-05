# Public staging test

## Staging method

A fresh temporary directory was populated only from the 208 exact paths in
`public_release_manifest.txt`. The real repository layout and local-only data were
not modified.

## Passed checks

- Manifest completeness: 208 listed files copied, 0 missing.
- Parse/compile: all 122 staged Python files compiled in memory without writing
  bytecode.
- Imports: core matching, parsing, writer, PN, payload layout, ECC, integrity, and
  v2 finalizer modules imported successfully.
- Exhaustive SECDED check: passed all 16 messages, 128 single-error cases, 448
  double-error cases, and the existing erasure cases.
- DISTRIBUTED/PREFIX layout, nesting, determinism, PN/layout independence, and
  noiseless decode check: passed.
- Decoder zero-correlation tie rule: passed (`0 -> +1`).
- One-codeword policy boundaries and legacy normal-allocation equivalence: passed.
- Synthetic frozen-integrity decisions: passed.
- V2 finalizer input-boundary/determinism regression using temporary synthetic
  manifests: passed.
- Canonical final-audit manifest: 58/58 entries passed.
- Release-audit manifest: 2/2 entries passed.
- Release-preparation manifest at staging time: 11/11 entries passed.
- Static public-tree scan: 0 personal absolute paths, 0 personal email addresses,
  and 0 probable hard-coded credentials/private keys/tokens.

## Intentionally unavailable checks

- Full v2 final-audit regeneration requires the eight approved branch manifests and
  their 200 frozen artifacts. They are local-only pending artifact rights/privacy
  review, so the staged source-only tree correctly fails the prerequisite check
  rather than silently producing a partial audit.
- `check_final_frozen_invariants` requires active raw replays, beatmaps, and
  `data/metadata/results_v3_clean.csv`.
- Adaptive-alpha repeated-parse and physical round-trip checks require raw `.osr`
  data.
- Artifact-schema/result checks for historical experiment branches require the
  excluded non-canonical result matrices.
- No network/API or new physical replay experiment was attempted.

## Result

**PASS for the proposed data-free public tree.** The unavailable checks are explicit
data/artifact prerequisites, not hidden test failures. Public reproduction
documentation must explain how a separately reviewed artifact bundle can supply
them.
