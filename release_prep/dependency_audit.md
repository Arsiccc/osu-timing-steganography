# Dependency audit

The existing dependency file used direct pins only for `osrparse` and `slider` and
omitted two packages imported directly by public source.

## Added declarations

- `pandas`: required throughout experiment analysis, preprocessing, result checks,
  and both finalizer versions. It is required for normal public analysis and audit
  workflows, not merely an optional private-data path.
- `requests`: imported directly by `osu_stego/api/osu_api.py` for the optional osu!
  data-acquisition workflow. `slider` currently also depends on `requests`, but a
  direct import must not rely on that transitive installation.

Both names were added to `requirements.txt` without changing any existing version,
upgrading packages, or introducing a new packaging system. The locally observed
versions during this audit were pandas 3.0.5 and requests 2.34.2; these observations
are not new project pins.

## Remaining environment limitation

`numpy`, `pandas`, `requests`, `scipy`, and `scikit-learn` remain unpinned, so the
dependency file does not yet define a byte-reproducible environment. Choosing a
supported Python/version matrix is a separate manual release decision; this cleanup
does not silently raise the historical minimum Python version.

