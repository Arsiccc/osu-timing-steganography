"""Exhaustive single/double-error checks for Hamming SECDED (8,4)."""

from __future__ import annotations

import itertools

import numpy as np

from osu_stego.stego.ecc import decode_hamming_8_4_secded, encode_hamming_8_4_secded
from osu_stego.stego.ecc import (
    decode_hamming_8_4_one_erasure,
    decode_hamming_8_4_with_erasures,
)


def main() -> None:
    for incomplete in (
        np.ones(3, dtype=np.int8),
        np.ones(7, dtype=np.int8),
    ):
        try:
            if len(incomplete) == 3:
                encode_hamming_8_4_secded(incomplete)
            else:
                decode_hamming_8_4_secded(incomplete)
        except ValueError:
            pass
        else:
            raise AssertionError("Nepotpun SECDED codeword mora biti odbijen.")
    messages = np.asarray(list(itertools.product((-1, 1), repeat=4)), dtype=np.int8)
    one_erasure_two_error_wrong = 0
    four_erasure_correct = 0
    four_erasure_ambiguous = 0
    for message in messages:
        encoded = encode_hamming_8_4_secded(message)
        decoded, status = decode_hamming_8_4_secded(encoded)
        if not np.array_equal(decoded, message) or status.tolist() != ["clean"]:
            raise AssertionError("Clean SECDED round trip nije tačan.")
        for position in range(8):
            damaged = encoded.copy()
            damaged[position] *= -1
            decoded, status = decode_hamming_8_4_secded(damaged)
            if not np.array_equal(decoded, message) or status.tolist() != ["corrected_single"]:
                raise AssertionError("SECDED nije ispravio single-bit grešku.")
        for left, right in itertools.combinations(range(8), 2):
            damaged = encoded.copy()
            damaged[[left, right]] *= -1
            _, status = decode_hamming_8_4_secded(damaged)
            if status.tolist() != ["detected_double"]:
                raise AssertionError("SECDED nije detektovao double-bit grešku.")
        for erasure in range(8):
            decoded, status = decode_hamming_8_4_one_erasure(encoded, erasure)
            if not np.array_equal(decoded, message) or status != "erasures_recovered":
                raise AssertionError("One-erasure clean slučaj nije ispravno dekodovan.")
            for error in range(8):
                if error == erasure:
                    continue
                damaged = encoded.copy()
                damaged[error] *= -1
                decoded, status = decode_hamming_8_4_one_erasure(damaged, erasure)
                if not np.array_equal(decoded, message) or status != "erasures_plus_error_corrected":
                    raise AssertionError("One erasure + one error nije ispravljeno.")
        for erasure_count in (1, 2, 3):
            for erasures in itertools.combinations(range(8), erasure_count):
                decoded, _ = decode_hamming_8_4_with_erasures(
                    encoded, erasures, max_unknown_errors=0
                )
                if not np.array_equal(decoded, message):
                    raise AssertionError("Do tri čista erasure-a moraju biti ispravljiva.")
        for erasure in range(8):
            remaining = [index for index in range(8) if index != erasure]
            for errors in itertools.combinations(remaining, 2):
                damaged = encoded.copy()
                damaged[list(errors)] *= -1
                decoded, _ = decode_hamming_8_4_one_erasure(damaged, erasure)
                if decoded is not None and not np.array_equal(decoded, message):
                    one_erasure_two_error_wrong += 1
        for erasures in itertools.combinations(range(8), 4):
            decoded, _ = decode_hamming_8_4_with_erasures(
                encoded, erasures, max_unknown_errors=0
            )
            if decoded is None:
                four_erasure_ambiguous += 1
            elif np.array_equal(decoded, message):
                four_erasure_correct += 1
            else:
                raise AssertionError("Čist four-erasure slučaj ne sme tiho pogrešiti.")
    if one_erasure_two_error_wrong != 2688:
        raise AssertionError("Neočekivana one-erasure + two-error algebra.")
    if (four_erasure_correct, four_erasure_ambiguous) != (896, 224):
        raise AssertionError("Neočekivana four-erasure algebra.")
    print("ECC SECDED CHECK OK")
    print("messages=16 single_errors=128 double_errors=448")
    print("one_erasure_plus_one_error=896 exhaustive_ok")
    print("one_erasure_plus_two_errors=2688 unique_wrong")
    print("four_erasures=896_correct 224_ambiguous")


if __name__ == "__main__":
    main()
