"""Which rules wrote an IPA file, recorded beside it and checked before cleaning.

The cleaner keeps a token only when every letter in it is one the installed
rules can emit, because a letter they cannot emit can only have come through
untransliterated from foreign text. That reasoning holds only for IPA the
installed rules wrote. On 2026-11-07 a Kazakh file transliterated before
0.5.7, which spelled Cyrillic ‹я› as ``ja`` with an ASCII ``a`` and ‹щ› as
``ɕː``, went through the 0.5.10 cleaner: about 54,700 Kazakh tokens such as
``sɑjasɑt`` and ``sijaqtə`` were dropped as foreign, and nothing said why,
because nothing recorded which rules had written the file.

So ``turkic-translit translit`` writes this record beside every IPA file it
writes, and :func:`require_written_by_installed_rules` refuses a file whose
record is missing, names another language, names other rules, or describes
other bytes. The record sits at :func:`manifest_path_for` of the IPA file,
the same place a downloaded corpus keeps its manifest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

from turkic_translit.core import _RULE_DIR
from turkic_translit.corpus.errors import (
    TransliteratedByOtherRulesError,
    TransliterationOutputChangedError,
    TransliterationRecordLanguageError,
    TransliterationRecordMissingError,
)
from turkic_translit.corpus.manifest import manifest_path_for
from turkic_translit.digests import require_sha256_hex, rule_digest, sha256_of_file
from turkic_translit.validation import (
    ERR_FIELD_TYPE,
    FieldError,
    require_non_empty_str,
    require_present,
)


class TransliterationRecord(TypedDict):
    """The rules that wrote one IPA file, and the bytes they wrote.

    Attributes:
        language: Language code whose rules ran.
        rule_file: Name of the rule file, e.g. ``kk_ipa.rules``.
        rule_digest: :func:`turkic_translit.digests.rule_digest` of that
            file as installed when the transliteration ran.
        output_sha256: SHA-256 of the IPA file as written.
    """

    language: str
    rule_file: str
    rule_digest: str
    output_sha256: str


def ipa_rule_file(language: str) -> str:
    """Name the IPA rule file a language's transliteration compiles.

    Args:
        language: Language code.

    Returns:
        The rule file's name within the packaged rules directory.
    """
    return f"{language}_ipa.rules"


def encode_transliteration_record(record: TransliterationRecord) -> dict[str, str]:
    """Render a record as a plain mapping ready for JSON.

    Args:
        record: The record to encode.

    Returns:
        A mapping carrying every field.
    """
    return {
        "language": record["language"],
        "rule_file": record["rule_file"],
        "rule_digest": record["rule_digest"],
        "output_sha256": record["output_sha256"],
    }


def decode_transliteration_record(
    source: Mapping[str, str | int | float | bool | None],
) -> TransliterationRecord:
    """Validate a loosely-typed mapping into a :class:`TransliterationRecord`.

    Args:
        source: Mapping holding every record field, as read from JSON.

    Returns:
        A fully validated record.

    Raises:
        FieldError: If a field is missing, not a string, empty, or a digest
            that is not 64 lowercase hexadecimal characters.
    """
    return TransliterationRecord(
        language=require_non_empty_str("language", require_present("language", source)),
        rule_file=require_non_empty_str("rule_file", require_present("rule_file", source)),
        rule_digest=require_sha256_hex("rule_digest", require_present("rule_digest", source)),
        output_sha256=require_sha256_hex("output_sha256", require_present("output_sha256", source)),
    )


def write_transliteration_record(output_path: Path, language: str) -> TransliterationRecord:
    """Record the installed rules and the written bytes beside an IPA file.

    Args:
        output_path: The IPA file, already written and closed.
        language: Language code whose rules wrote it.

    Returns:
        The record written.

    Raises:
        FileNotFoundError: When the IPA file or the language's rule file
            does not exist.
    """
    rule_file = ipa_rule_file(language)
    record = TransliterationRecord(
        language=language,
        rule_file=rule_file,
        rule_digest=rule_digest(_RULE_DIR / rule_file),
        output_sha256=sha256_of_file(output_path),
    )
    manifest_path_for(output_path).write_text(
        json.dumps(encode_transliteration_record(record), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def read_transliteration_record(path: Path) -> TransliterationRecord:
    """Read a record back from disk and validate it.

    Args:
        path: Location of the record.

    Returns:
        The validated record.

    Raises:
        FieldError: If the document is not a mapping, or a field fails
            validation.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise FieldError(
            ERR_FIELD_TYPE,
            "<transliteration record>",
            f"expected a mapping of fields, got {type(document).__name__}",
        )
    return decode_transliteration_record(document)


def require_written_by_installed_rules(corpus: Path, language: str) -> TransliterationRecord:
    """Refuse an IPA file the installed rules did not write as it now stands.

    Args:
        corpus: The IPA file.
        language: The language its file name carries.

    Returns:
        The file's record, once every check has passed.

    Raises:
        TransliterationRecordMissingError: When no record sits beside it.
        TransliterationRecordLanguageError: When the record names another
            language than ``language``.
        TransliteratedByOtherRulesError: When the record's rule digest is
            not the installed rule file's.
        TransliterationOutputChangedError: When the file's SHA-256 is not
            the one recorded.
        FieldError: When the record cannot be read as one.
    """
    record_path = manifest_path_for(corpus)
    if not record_path.is_file():
        raise TransliterationRecordMissingError(str(corpus), str(record_path))
    record = read_transliteration_record(record_path)
    if record["language"] != language:
        raise TransliterationRecordLanguageError(str(corpus), language, record["language"])
    rule_file = ipa_rule_file(language)
    installed = rule_digest(_RULE_DIR / rule_file)
    if record["rule_file"] != rule_file or record["rule_digest"] != installed:
        raise TransliteratedByOtherRulesError(
            str(corpus),
            f"{record['rule_file']} {record['rule_digest']}",
            f"{rule_file} {installed}",
        )
    found = sha256_of_file(corpus)
    if found != record["output_sha256"]:
        raise TransliterationOutputChangedError(str(corpus), record["output_sha256"], found)
    return record


__all__ = [
    "TransliterationRecord",
    "decode_transliteration_record",
    "encode_transliteration_record",
    "ipa_rule_file",
    "read_transliteration_record",
    "require_written_by_installed_rules",
    "write_transliteration_record",
]
