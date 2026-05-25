"""Thin wrapper around the official `engram` Python SDK.

Isolates the SDK surface so the provider depends only on this module's
contract (constructor, add_memory, search_memories). If the SDK shape
changes, only this file needs to track it.

The wrapper exposes three operations: construction, add_memory, and
search_memories. Engram's SDK also offers `delete` and `get`, but we
deliberately do not expose them as agent tools — Engram is built
around purposeful forgetting (correction-by-write), so the agent
"forgets" by storing a correcting memory rather than issuing a delete.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.engram.weaviate.io"


class EngramNotAvailableError(RuntimeError):
    """Raised when the `engram` SDK can't be imported.

    Install with: pip install weaviate-engram
    """


def is_sdk_available() -> bool:
    """Return True iff the `engram` SDK is importable. Cheap, no network."""
    try:
        import engram  # noqa: F401
    except Exception:
        return False
    return True


class EngramClient:
    """Wrapper around the Engram SDK's `EngramClient`.

    The underlying `engram.EngramClient` only needs an API key. We accept
    an optional `base_url` for staging / custom endpoints and a `timeout`.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        try:
            from engram import EngramClient as _SDKClient
        except Exception as exc:
            raise EngramNotAvailableError(
                "Engram SDK not installed. Run: pip install weaviate-engram"
            ) from exc
        kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._sdk = _SDKClient(**kwargs)
        self._api_key = api_key
        self._base_url = base_url or _DEFAULT_BASE_URL
        self._timeout = timeout

    def close(self) -> None:
        """Release the underlying HTTP client. Safe to call multiple times."""
        sdk_close = getattr(self._sdk, "close", None)
        if callable(sdk_close):
            try:
                sdk_close()
            except Exception:
                logger.debug("Engram SDK close raised", exc_info=True)

    def add_memory(
        self,
        text: str,
        *,
        user_id: str,
        properties: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Ingest a single text payload into Engram for the given user.

        Engram processes the payload asynchronously via its server-side
        pipelines, so this call returns immediately with a Run handle.

        ``properties`` maps to Engram's ``properties: dict[str, str]`` —
        soft-isolation tags the pipeline can filter on at search time.
        Callers must pre-stringify both keys and values (the SDK enforces
        ``str -> str``).
        """
        kwargs: Dict[str, Any] = {"user_id": user_id}
        if properties:
            kwargs["properties"] = properties
        run = self._sdk.memories.add(text, **kwargs)
        return _run_to_dict(run)

    def search_memories(
        self, query: str, *, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Semantic recall. Returns at most `limit` memory dicts.

        The SDK's SearchResults is list-like; we slice here so the
        caller never has to know whether `limit` is honoured server-side.
        """
        results = self._sdk.memories.search(query=query, user_id=user_id)
        out: List[Dict[str, Any]] = []
        for memory in results:
            out.append(_memory_to_dict(memory))
            if len(out) >= limit:
                break
        return out


def _run_to_dict(run: Any) -> Dict[str, Any]:
    if run is None:
        return {}
    if isinstance(run, dict):
        return {
            "run_id": str(run.get("run_id") or run.get("id") or ""),
            "status": str(run.get("status") or ""),
        }
    return {
        "run_id": str(getattr(run, "run_id", "") or ""),
        "status": str(getattr(run, "status", "") or ""),
    }


def _memory_to_dict(item: Any) -> Dict[str, Any]:
    """Normalise an engram.Memory into a stable dict.

    SDK Memory fields available: id, content, topic, group, created_at,
    updated_at, user_id, tags, score, properties, project_id.

    Phase 1 intentionally surfaces only id / content / score / topic to keep
    the tool payloads (and the model's attention) lean — neither the agent
    nor the prefetch formatter needs the rest yet. When a future tool needs
    timestamps, tags, or properties (e.g. property-scoped search), extend
    this normaliser rather than reaching into the raw SDK object.
    """
    if isinstance(item, dict):
        score = item.get("score")
        return {
            "id": str(item.get("id", "") or ""),
            "content": str(item.get("content", "") or ""),
            "score": _maybe_float(score),
            "topic": item.get("topic"),
        }
    return {
        "id": str(getattr(item, "id", "") or ""),
        "content": str(getattr(item, "content", "") or ""),
        "score": _maybe_float(getattr(item, "score", None)),
        "topic": getattr(item, "topic", None),
    }


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
