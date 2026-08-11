"""Traceable error types for rule-file provenance.

Every failure here carries a stable, greppable code, so a caller can tell
"this rule file declares no source at all" from "it declares one whose
year is not a number" without reading prose. Codes are never reused or
renumbered.
"""

from __future__ import annotations

from typing import Final

ERR_SOURCE_FIELD_MISSING: Final = "TURKIC_RULESRC_001_FIELD_MISSING"
ERR_SOURCE_LINE_MALFORMED: Final = "TURKIC_RULESRC_002_LINE_MALFORMED"


class RuleSourceError(Exception):
    """Base class for provenance failures.

    Args:
        code: Stable error code from this module.
        message: Human-readable description naming the offending file.
    """

    def __init__(self, code: str, message: str) -> None:
        """Store the code and render ``code: message`` as the string form."""
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class RuleSourceFieldMissingError(RuleSourceError):
    """A required provenance field is absent from a rule file's header.

    Args:
        origin: Name of the rule file.
        field: Name of the field that was expected.
    """

    def __init__(self, origin: str, field: str) -> None:
        """Report which file is missing which field."""
        super().__init__(
            ERR_SOURCE_FIELD_MISSING,
            f"{origin} declares no '# Source-{field}:' line; "
            f"a rule set whose provenance cannot be read cannot be checked "
            f"against the description it implements",
        )
        self.origin = origin
        self.field = field


class RuleSourceMalformedLineError(RuleSourceError):
    """A provenance line is present but cannot be read.

    Args:
        origin: Name of the rule file.
        line: The offending line, quoted back to the caller.
    """

    def __init__(self, origin: str, line: str) -> None:
        """Report which file carries which unreadable line."""
        super().__init__(
            ERR_SOURCE_LINE_MALFORMED,
            f"{origin} carries an unreadable provenance line: {line!r}",
        )
        self.origin = origin
        self.line = line


__all__ = [
    "ERR_SOURCE_FIELD_MISSING",
    "ERR_SOURCE_LINE_MALFORMED",
    "RuleSourceError",
    "RuleSourceFieldMissingError",
    "RuleSourceMalformedLineError",
]
