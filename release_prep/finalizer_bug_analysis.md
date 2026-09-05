# Finalizer v1 bug analysis

## Classification

**Reproducibility-tooling defect.** This is not a scientific-method change and is
not evidence that a frozen metric, denominator, uncertainty result, figure datum,
or final-system rule is wrong.

## Exact cause

`scripts/analysis/finalize_research_results.py` defines `audit_manifests()` by
globally enumerating:

```text
results/*/result_hashes.json
```

The enumeration is unrelated to the selected output directory. The canonical run
was created when `results/final_audit_v1/` was absent or replaced as its output, so
the eight intended branch manifests were discovered. Once that canonical output
exists, running v1 with `--output-dir` pointed at a safe temporary directory also
discovers `results/final_audit_v1/result_hashes.json`. It therefore reports nine
input manifests and incorporates the generated final audit as an input to another
final audit.

The same unbounded discovery rule can also admit future adjacent result manifests.
It is thus an input-boundary defect, not merely a special case involving one name.

## Effect on the frozen audit

The already-frozen canonical directory remains internally valid:

- all 58 entries in its manifest match;
- its system-specification SHA-256 remains
  `b99790fe0b39813e2c0a9777d7b98f05c69f2f3c5b43e3322a028bd78c0c7038`;
- its manifest SHA-256 remains
  `6eed818fb1c3e428a649d46bf801b7e30408ba37a969af743558edad495afd36`;
- the scientific source tables and implementation hashes remain unchanged.

The defect affects attempts to reproduce the canonical bookkeeping after the
canonical output already exists. It does not retroactively change the eight inputs
used by the frozen audit.

## Safe minimal reproduction

From the repository root, choose a new temporary directory and run v1 without
touching the canonical output:

```bash
tmp_dir="$(mktemp -d)"
python3 -B -m scripts.analysis.finalize_research_results \
  --output-dir "$tmp_dir/final_audit_v1" \
  --check-reproducibility
```

The command reports nine frozen manifests because the canonical final manifest is
now adjacent to the eight branch manifests under `results/`. The temporary run may
be deleted after inspection. It performs analysis-only regeneration: no API call,
physical replay generation, rematching, or model training.

## Corrective design

The v1 file is hash-pinned and must remain unchanged. Version 2 delegates stable
scientific generation to v1 but replaces unbounded manifest discovery with an
explicit, version-controlled tuple of the eight approved branch manifests. It also
bounds the historical language-audit input list so newly generated release reports
cannot recursively alter finalizer output. Output defaults to a new v2 directory
and v2 refuses to overwrite the canonical v1 audit.

