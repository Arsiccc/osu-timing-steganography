# Machine-path audit

## Proposed public set

The package source, scripts, configuration, selected audit files, and release
documentation were scanned for Unix home paths, Windows user-profile paths, and the
historical desktop workspace name.

Remaining personal/machine-specific absolute paths: **0**.

The three executable defaults formerly tied to one user's Downloads directory are
now based on `Path.home()` and remain configurable command-line arguments. URL paths
such as the osu! API `/users/...` endpoint are not filesystem paths.

## Frozen and local-only material

Eight non-canonical historical result files contain personal absolute source/output
paths. They remain unchanged to preserve frozen provenance and hashes and are covered
by the local-only result scope:

- `results/one_codeword_floor_pilot_v1/config.json`
- `results/classifier_family_steganalysis_v1/dataset_manifest.csv`
- `results/message_content_robustness_v1/source_artifact_audit.md`
- `results/validation/dataset_manifest.csv`
- `results/one_codeword_floor_validation_v1/corpus_manifest.csv`
- `results/one_codeword_floor_validation_v1/new_corpus_selection.json`
- `results/one_codeword_floor_validation_v1/calibration_manifest.csv`
- `results/one_codeword_floor_validation_v1/exclusions.csv`

`AGENTS.md`, backups, raw data, archive material, non-canonical results, and cleanup
working documents are local-only. No hash-pinned scientific evidence was rewritten
for cosmetic path removal.

