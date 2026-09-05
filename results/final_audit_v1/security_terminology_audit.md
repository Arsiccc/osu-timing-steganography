# Security terminology audit

PASS with limitations. Final material treats PN as a spreading/detection key, not encryption, authentication, tamper resistance, or a cryptographically strong key. The implementation's 32-bit NumPy seed is explicitly non-cryptographic. Wrong-key decoding and AUC results address tested reliability/detectability only; they do not establish confidentiality or security against chosen-message or arbitrary steganalysis attacks.

One legacy source-code docstring (`osu_stego/stego/pn_sequence.py`) says an attacker cannot approximately guess the PN sequence. That wording is stronger than the demonstrated evidence; it is recorded as a minor documentation issue and is not repeated in final scientific claims. No scientific behavior was changed.

`archive/legacy_scripts/run_decode.py` contains a hardcoded historical PN research key. It is not an osu! API credential and is not part of the final protocol; it must not be presented as secret key management.
