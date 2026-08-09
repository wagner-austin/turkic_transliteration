"""Corpus sanity checks: edit distance, byte density and normalisation.

Each check reads its files directly and reports a number. They exist to
answer questions about a downloaded or transliterated corpus that its own
line count cannot: whether two parallel files really correspond, whether
the text is the script it claims to be, and whether it is normalised.
"""

import os
import unicodedata
from statistics import median

from rapidfuzz.distance import Levenshtein

__all__ = ["bytes_per_char", "is_nfc", "median_lev"]


def median_lev(file_lat: str, file_ipa: str, sample: int = 5000) -> float:
    """Report the median edit distance between two aligned files.

    Args:
        file_lat: Path of the first file, one record per line.
        file_ipa: Path of the second file, aligned line for line with
            the first. Pairing stops at whichever file ends sooner.
        sample: Number of lines to compare before stopping.

    Returns:
        The median normalised Levenshtein distance, from 0.0 for
        identical lines to 1.0 for lines sharing no characters.

    Raises:
        StatisticsError: If no pair was compared, which means at least
            one of the files was empty.
        OSError: If either file cannot be read.
    """
    distances = []
    with (
        open(file_lat, encoding="utf8") as first,
        open(file_ipa, encoding="utf8") as second,
    ):
        for index, (left, right) in enumerate(zip(first, second, strict=False)):
            if index == sample:
                break
            distances.append(Levenshtein.normalized_distance(left.strip(), right.strip()))
    return median(distances)


def bytes_per_char(filename: str) -> float:
    """Report a file's size divided by the characters it decodes to.

    The ratio distinguishes scripts: ASCII text measures 1.0, Cyrillic
    about 2.0, and most CJK about 3.0. A file that claims to be one
    script and measures like another is worth looking at.

    Args:
        filename: Path of the file to measure.

    Returns:
        Bytes on disk per decoded character. Line endings count, so a
        file stored with CRLF measures above the ratio its script alone
        would give.

    Raises:
        ZeroDivisionError: If the file decodes to no characters.
        OSError: If the file cannot be read.
    """
    size = os.path.getsize(filename)
    with open(filename, encoding="utf8") as handle:
        characters = sum(len(line) for line in handle)
    return size / characters


def is_nfc(filename: str) -> bool:
    """Report whether every line of a file is in composed normal form.

    Args:
        filename: Path of the file to check.

    Returns:
        True when the whole file is NFC. Decomposed text renders
        identically but compares unequal, so a corpus mixing the two
        silently splits counts for the same word.

    Raises:
        OSError: If the file cannot be read.
    """
    with open(filename, encoding="utf8") as handle:
        return all(unicodedata.is_normalized("NFC", line) for line in handle)
