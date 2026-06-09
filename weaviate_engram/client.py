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

# Canonical list of properties the Engram backend actually accepts.
# Any property not in this list causes an "invalid scope property" error
# at the backend. Only conversation_id is supported as of 2026-06-06.
# Extend this set when the backend schema grows new keys.
_ALLOWED_PROPERTIES: frozenset[str] = frozenset({"conversation_id"})


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
        conversation_id: Optional[str] = None,
        properties: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Ingest a single text payload into Engram for the given user.

        Engram processes the payload asynchronously via its server-side
        pipelines, so this call returns immediately with a Run handle.

        ``properties`` maps to Engram's ``properties: dict[str, str]`` —
        soft-isolation tags the pipeline can filter on at search time.
        Callers must pre-stringify both keys and values (the SDK enforces
        ``str -> str``).

        Only ``conversation_id`` is a valid property key as far as the
        Engram backend is concerned. Unknown keys cause "invalid scope
        property" errors at the backend, so unknown keys are stripped
        before sending. The ``conversation_id`` wins over any conflicting
        key in ``properties``.

        ``conversation_id`` is a Hermes-side concept that scopes a memory
        to a specific chat session. As of ``weaviate-engram==0.6.0`` the
        SDK does NOT accept ``conversation_id`` as a top-level kwarg on
        ``memories.add`` (signature: ``(input_data, *, user_id, group,
        properties)``). Until the SDK grows a first-class field, this
        client folds the value into ``properties["conversation_id"]`` so
        it is at least persisted on the memory object. Whether the
        Engram backend filters search results on this property is
        server-side behavior and out of scope for this client — do not
        assume conversation-scoped recall works without verifying
        against the deployed Engram version.
        """
        filtered = _filter_properties(properties, _ALLOWED_PROPERTIES)
        merged_props = _merge_conversation_id(filtered, conversation_id)
        run = self._sdk.memories.add(
            text,
            user_id=user_id,
            properties=merged_props,
        )
        return _run_to_dict(run)

    def search_memories(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 10,
        conversation_id: Optional[str] = None,
        properties: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic recall. Returns at most `limit` memory dicts.

        The SDK's SearchResults is list-like; we slice here so the
        caller never has to know whether `limit` is honoured server-side.

        ``properties`` and ``conversation_id`` follow the same rules as
        ``add_memory``: unknown property keys are stripped against
        ``_ALLOWED_PROPERTIES`` before being sent, and an explicit
        ``conversation_id`` wins over any conflicting key in ``properties``.

        ``conversation_id`` narrows results to a specific conversation.
        As of ``weaviate-engram==0.6.0`` the SDK does NOT accept it as
        a top-level kwarg on ``memories.search`` (signature: ``(*,
        query, topics, user_id, group, retrieval_config, properties)``).
        Until the SDK grows a first-class field, this client forwards
        it as ``properties["conversation_id"]`` so the value is at
        least sent to the backend. Whether the backend actually
        filters on it is server-side behavior and not guaranteed —
        treat conversation-scoped recall as best-effort, not as
        hard isolation, until the SDK exposes a real field.
        """
        filtered = _filter_properties(properties, _ALLOWED_PROPERTIES)
        merged_props = _merge_conversation_id(filtered, conversation_id)
        results = self._sdk.memories.search(
            query=query,
            user_id=user_id,
            properties=merged_props,
        )
        out: List[Dict[str, Any]] = []
        for memory in results:
            out.append(_memory_to_dict(memory))
            if len(out) >= limit:
                break
        return out


def _filter_properties(
    properties: Optional[Dict[str, str]],
    allowed: frozenset[str],
) -> Optional[Dict[str, str]]:
    """Strip unknown property keys, keeping only those in ``allowed``.

    The Engram backend rejects any property key not in its schema with
    "invalid scope property". Until the schema is extended this function
    acts as a firewall so callers can pass arbitrary metadata without
    triggering errors — unknown keys are dropped, and dropped keys are
    logged at DEBUG so the loss is debuggable when callers wonder why
    their tag didn't land on the memory.
    """
    if not properties:
        return None
    filtered = {k: v for k, v in properties.items() if k in allowed}
    dropped = set(properties) - filtered.keys()
    if dropped:
        logger.debug(
            "Dropped unknown engram property keys: %s (allowed=%s)",
            sorted(dropped),
            sorted(allowed),
        )
    return filtered or None


def _merge_conversation_id(
    properties: Optional[Dict[str, str]],
    conversation_id: Optional[str],
) -> Optional[Dict[str, str]]:
    """Fold an explicit ``conversation_id`` into the ``properties`` dict.

    Returns a fresh dict so callers can keep reusing their own. Empty /
    whitespace-only values are dropped. If the result has no entries we
    return ``None`` so the SDK call stays slim.
    """
    merged: Dict[str, str] = dict(properties or {})
    if conversation_id is not None:
        cid = str(conversation_id).strip()
        if cid:
            merged["conversation_id"] = cid
    return merged or None


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
