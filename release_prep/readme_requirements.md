# README requirements

The final README must be written only after the release manifest, license choice,
and data-availability wording are approved. It must contain the following.

## Project and method

- A concise project overview and the research question: detectability/reliability of
  spread-spectrum timing steganography in osu!standard replay files.
- The final frozen method from `results/final_audit_v1/final_system_spec.json`:
  sender-local adaptive alpha v2, keyed DISTRIBUTED placement, N=8, normal 7.5%
  coded-bit allocation, SECDED(8,4), frozen integrity rejection, and abstention when
  no complete codeword fits.
- A clear statement that PN is non-cryptographic, the method is not encryption or
  authentication, and the work does not establish undetectability or security.

## Repository use

- Public repository layout, distinguishing core package, experiment/analysis
  scripts, tests/checks, configs, canonical audit, and release audit.
- Supported Python version(s), installation with `python -m pip install -r
  requirements.txt`, and any platform requirement for figure rendering.
- Environment variables `OSU_CLIENT_ID` and `OSU_CLIENT_SECRET` only in the optional
  acquisition section; never include real values.
- Basic synthetic encode/decode usage that does not require private data.
- V2 finalizer invocation and the separately supplied artifact bundle required for
  full audit regeneration.

## Data and reproduction

- Data availability statement explaining that raw `.osr`, `.osu`, API caches, and
  player-linked tables are excluded pending rights/privacy review.
- Exact expected local paths or configurable arguments for user-supplied data,
  without personal absolute paths.
- A tiered reproduction path: data-free synthetic checks, artifact-based final audit,
  and full physical experiments requiring separately obtained data.
- Frozen identities:
  - system spec: `b99790fe0b39813e2c0a9777d7b98f05c69f2f3c5b43e3322a028bd78c0c7038`;
  - canonical audit manifest: `6eed818fb1c3e428a649d46bf801b7e30408ba37a969af743558edad495afd36`.

## Results and limitations

Any README number must cite one of these canonical sources rather than an
intermediate table:

- corpus/matching: `canonical_metrics.csv`, claims `DATA_001` and `MATCH_001`;
- reliability/confidence: `canonical_metrics.csv`, claims `DECODER_001` and
  `CONF_001`;
- DISTRIBUTED detectability: `final_claim_registry.csv`, claims `DIST_001` and
  `DIST_002`;
- ECC/integrity: `table_ecc_integrity.csv`, claims `ECC_001`, `INT_001`, `INT_002`;
- PN/message/classifier robustness: `table_robustness.csv`, claims `PN_001`,
  `MSG_001`, `CLF_001`.

The README must prominently summarize the limitations in `limitations.md`, including
above-chance detectability, corpus/adversary bounds, 50% ECC rate, abstention,
non-cryptographic 32-bit PN seed, and PN/message sensitivity. It must not combine
differently scoped metrics into a synthetic end-to-end headline.

## Project metadata

- Citation section placeholder until the author chooses the desired citation form.
- License status. Until the user selects a code license, state that no license has
  yet been granted; separately state that any future code license does not grant
  rights to third-party replay or beatmap data.

