from __future__ import annotations


def url_ok(url: str, timeout: float = 5.0) -> bool:
    """Return True if the remote host answers 2xx/3xx within *timeout* sec."""
    try:
        import requests  # Lazy import for minimal core dependencies

        r = requests.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code < 400:
            return True
        # some hosts (e.g. Leipzig) block HEAD; try a lightweight GET
        r = requests.get(url, stream=True, timeout=timeout)
        return r.status_code < 400
    except requests.RequestException:
        return False
