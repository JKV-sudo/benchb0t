"""
framework/utils.py
~~~~~~~~~~~~~~~~~~
Shared utility functions: response builders, URL normalization, truncation, etc.
Consolidates duplicated logic across dashboard.py, api.py, and other modules.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse


def ok_response(
    message: str | None = None,
    data: dict[str, object] | None = None,
    **kwargs: object
) -> JSONResponse:
    """
    Build a successful API response.

    Parameters
    ----------
    message : str, optional
        Optional status message to include.
    data : dict, optional
        Optional payload dict to merge into response.
    **kwargs
        Additional fields to include in response.

    Returns
    -------
    JSONResponse
        Response with {"ok": True, "message": ..., "data": ..., ...kwargs}
    """
    payload = {"ok": True}
    if message:
        payload["message"] = message
    if data:
        payload["data"] = data
    payload.update(kwargs)
    return JSONResponse(payload)


def error_response(
    message: str,
    status_code: int = 400,
    **kwargs: object
) -> JSONResponse:
    """
    Build an error API response.

    Parameters
    ----------
    message : str
        Error message.
    status_code : int
        HTTP status code (default: 400).
    **kwargs
        Additional fields to include in response.

    Returns
    -------
    JSONResponse
        Response with {"error": message, ...kwargs}
    """
    payload = {"error": message}
    payload.update(kwargs)
    return JSONResponse(payload, status_code=status_code)


def normalize_url(url: str) -> str:
    """
    Normalize an OpenAI-compatible endpoint URL.

    Ensures URL has protocol prefix and ends with /v1.
    Used for Ollama, vLLM, OpenRouter, OpenAI, etc.

    Parameters
    ----------
    url : str
        Base URL, may or may not include protocol/path.

    Returns
    -------
    str
        Normalized URL with http:// prefix and /v1 suffix.

    Examples
    --------
    >>> normalize_url("localhost:11434")
    'http://localhost:11434/v1'
    >>> normalize_url("https://api.openai.com")
    'https://api.openai.com/v1'
    """
    url = str(url).strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    if not url.rstrip("/").endswith("/v1"):
        url = url.rstrip("/") + "/v1"
    return url


def truncate(value: object, limit: int = 80) -> str:
    """
    Truncate a value to a short string for logging.

    Parameters
    ----------
    value : object
        Value to stringify.
    limit : int
        Maximum length before ellipsis (default: 80).

    Returns
    -------
    str
        Truncated string representation.
    """
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"
