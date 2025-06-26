"""Language utility helpers shared across UI and CLI modules."""

from __future__ import annotations

from functools import lru_cache

__all__ = ["pretty_lang"]

_OVERRIDES = {
    "bh": "Bhojpuri",
}


@lru_cache(maxsize=512)
def pretty_lang(code: str) -> str:
    """Return human-friendly label like "Persian (fa)" for an ISO code.

    Falls back gracefully when *pycountry* is missing, when the code is unknown,
    or when its *name* field is empty.
    """
    # Manual overrides first
    if code in _OVERRIDES:
        return f"{_OVERRIDES[code]} ({code})"

    try:
        import pycountry

        rec = pycountry.languages.get(alpha_2=code) or pycountry.languages.get(
            alpha_3=code
        )
        if rec is not None:
            name = getattr(rec, "name", "").strip()
            if name:
                return f"{name} ({code})"
    except Exception:
        # Any import or lookup problem – just return the code.
        pass
    return code
