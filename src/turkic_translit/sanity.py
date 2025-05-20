"""Helper functions for Levenshtein and byte checks."""

from rapidfuzz.distance import Levenshtein
import io
import os


def median_lev(file_lat: str, file_ipa: str, sample: int = 5000) -> float:
    from statistics import median

    m = []
    with (
        io.open(file_lat, encoding="utf8") as f1,
        io.open(file_ipa, encoding="utf8") as f2,
    ):
        for i, (line, i_) in enumerate(zip(f1, f2)):
            if i == sample:
                break
            m.append(Levenshtein.normalized_distance(line.strip(), i_.strip()))
    return median(m)


def bytes_per_char(filename: str) -> float:
    import io

    b = os.path.getsize(filename)
    with io.open(filename, encoding="utf8") as f:
        chars = sum(len(line) for line in f)
    return b / chars


def is_nfc(filename: str) -> bool:
    import unicodedata
    import io

    with io.open(filename, encoding="utf8") as f:
        return all(unicodedata.is_normalized("NFC", line) for line in f)
