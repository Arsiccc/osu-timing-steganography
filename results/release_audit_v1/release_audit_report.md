# Public-release finalizer audit v1

This is a tooling/reproducibility audit, not a new scientific experiment.

## Frozen identities

- Frozen v1 finalizer SHA-256:
  `7c930100de258dc47bb03404df9053711462db9b626373adaba32b3d5516c337`
- Frozen final system specification SHA-256:
  `b99790fe0b39813e2c0a9777d7b98f05c69f2f3c5b43e3322a028bd78c0c7038`
- Frozen canonical audit manifest SHA-256:
  `6eed818fb1c3e428a649d46bf801b7e30408ba37a969af743558edad495afd36`

The v1 source and `results/final_audit_v1/` were not modified.

## Versioned correction

V2 finalizer SHA-256:
`9deed319aac4d4a2e4a4d5e478bc7bb77ceb44c0b5908763a825f1433f66e9fe`.

V2 replaces the v1 `results/*/result_hashes.json` glob with the exact eight
approved branch manifests recorded in `input_manifest.json`. It also bounds the
historical language-audit documents, preventing later release outputs from becoming
recursive inputs. The internal `AGENTS.md` is not a v2 input. V2 refuses to overwrite
the canonical v1 directory.

## Regression and deterministic regeneration

- Focused synthetic input-boundary regression: **PASS**.
- Real approved-manifest validation: **PASS**, 8 manifests / 200 entries.
- V2 clean rerun comparison: **PASS**, byte-for-byte deterministic.
- Generated v2 output manifest SHA-256:
  `c3972411994019cca0d0f8e9dee3f231ed001cf93a7db7b43ea4a3912a2780a1`.
- API access, physical `.osr` generation, rematching, offset calibration, and model
  fitting: **not performed**.

## V1 versus v2 output classification

Both outputs contained the same 59 relative files. Of these, 47 were byte-identical
and 12 differed:

- **IDENTICAL:** system specification, canonical metrics, claim registry,
  denominators, uncertainty results, narrative/claims, figures, paper tables, and
  all other scientific outputs (47 files).
- **EXPECTED TOOLING DIFFERENCE:** seven figure metadata files record the v2
  generator version/hash; two documentation files name the v2 entry point; the
  reproducibility report records the bounded allowlist; the reporting-only language
  audit omits 18 rows sourced from local-only `AGENTS.md`; and `result_hashes.json`
  necessarily reflects those provenance changes (12 files).
- **UNEXPECTED SCIENTIFIC DIFFERENCE:** none.

The generated v2 system specification retained the exact frozen system-specification
hash. No scientific result or final-system semantic changed.
