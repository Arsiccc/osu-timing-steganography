# Final reproducibility audit

- Frozen branch manifests: PASS (8 manifests; 200 artifacts).
- No physical replay generation: PASS by entry-point code-path audit.
- No API access, offset calibration, corpus rematching, or model fitting: PASS.
- No frozen source rows/configs changed: PASS; finalizer writes only its selected output directory.
- Deterministic canonical metrics, claim registry, tables, metadata, and figures: PASS: clean temporary regeneration matched byte-for-byte.
- System specification SHA-256: `b99790fe0b39813e2c0a9777d7b98f05c69f2f3c5b43e3322a028bd78c0c7038`.
