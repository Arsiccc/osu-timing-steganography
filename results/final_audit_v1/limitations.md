# Limitations registry

System spec: `b99790fe0b39813e2c0a9777d7b98f05c69f2f3c5b43e3322a028bd78c0c7038`.

- Corpora are not fully independent external sources; the new-map corpus remains in the same broader Kaggle/o!rdr source family.
- PN, message, and classifier robustness analyses reuse scientifically consumed data.
- The PN generator has a 32-bit effective seed and is not a cryptographic KDF. Wrong-key BER is not a security proof.
- Steganalysis covers 29 hand-engineered timing features and four preregistered classifier families, not arbitrary attackers; per-map detectability is heterogeneous.
- PN realization and, more modestly, message content change reliability.
- SECDED(8,4) halves the useful coding rate. Integrity rejection lowers silent acceptance while rejecting some correct messages; the two-codeword subgroup was especially fragile.
- The normal 7.5% policy abstains on short replay files, so reported eligible-case reliability is not population-wide coverage.
- The method is osu!standard-specific. Three maps were quarantined because one fixed map-level offset could not represent them reliably; this does not imply file corruption.
- DT/HT behavior is limited to the empirically validated pipeline's stored timestamp handling and must not be generalized.
- No single historical experiment measured every claimed property on every corpus; final metrics remain scope-labelled.
