"""Weaviate Engram memory provider plugin for Hermes Agent.

Distributed as a standalone repo per Hermes CONTRIBUTING.md — install by
cloning this repo to ``$HERMES_HOME/plugins/weaviate_engram/`` and then
``pip install weaviate-engram`` for the runtime SDK.

Phase 1 walking skeleton:
- Required ABC methods + sync_turn + prefetch + system_prompt_block.
- Three tools: engram_search, engram_store, engram_fetch.
- Synchronous-recall prefetch (no queued background prefetch yet).

Deliberately no engram_forget tool — Engram treats forgetting as a
first-class server-side concern (purposeful forgetting). The agent
corrects memories by storing a new correcting memory; reconcile
pipelines supersede the old one. See README.md for details.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .client import EngramClient, EngramNotAvailableError, is_sdk_available

logger = logging.getLogger(__name__)

_DEFAULT_USER_ID_TEMPLATE = "{identity}"
_DEFAULT_MAX_RECALL_RESULTS = 10
_DEFAULT_MIN_CAPTURE_CHARS = 10
_DEFAULT_API_TIMEOUT = 10.0
_TRIVIAL_RE = re.compile(
    r"^(ok|okay|thanks|thank you|got it|sure|yes|no|yep|nope|k|ty|thx|np)\.?$",
    re.IGNORECASE,
)
# Strip our own injected memory context out of assistant content before
# re-ingesting it — otherwise every turn would re-add the previous turn's
# recalled memories as if they were new.
_CONTEXT_STRIP_RE = re.compile(
    r"<engram-context>[\s\S]*?</engram-context>\s*", re.DOTALL
)


def _default_config() -> dict:
    return {
        "user_id_template": _DEFAULT_USER_ID_TEMPLATE,
        "auto_recall": True,
        "auto_capture": True,
        "max_recall_results": _DEFAULT_MAX_RECALL_RESULTS,
        "min_capture_chars": _DEFAULT_MIN_CAPTURE_CHARS,
        "api_timeout": _DEFAULT_API_TIMEOUT,
        "pipeline_hint": "",
    }


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _sanitize_user_id(raw: str) -> str:
    """Restrict user_id to a conservative charset until preview docs confirm limits.

    Weaviate tenant names allow [A-Za-z0-9_-]; we follow that.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_\-]", "_", raw or "")
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
    return cleaned or "default"


def _load_config(hermes_home: str) -> dict:
    config = _default_config()
    config_path = Path(hermes_home) / "weaviate_engram.json"
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                config.update({k: v for k, v in raw.items() if v is not None})
        except Exception:
            logger.debug("Failed to parse %s", config_path, exc_info=True)

    config["user_id_template"] = str(config.get("user_id_template") or _DEFAULT_USER_ID_TEMPLATE)
    config["auto_recall"] = _as_bool(config.get("auto_recall"), True)
    config["auto_capture"] = _as_bool(config.get("auto_capture"), True)
    try:
        config["max_recall_results"] = max(1, min(20, int(config.get("max_recall_results", _DEFAULT_MAX_RECALL_RESULTS))))
    except Exception:
        config["max_recall_results"] = _DEFAULT_MAX_RECALL_RESULTS
    try:
        config["min_capture_chars"] = max(0, int(config.get("min_capture_chars", _DEFAULT_MIN_CAPTURE_CHARS)))
    except Exception:
        config["min_capture_chars"] = _DEFAULT_MIN_CAPTURE_CHARS
    try:
        config["api_timeout"] = max(0.5, min(60.0, float(config.get("api_timeout", _DEFAULT_API_TIMEOUT))))
    except Exception:
        config["api_timeout"] = _DEFAULT_API_TIMEOUT
    config["pipeline_hint"] = str(config.get("pipeline_hint") or "").strip()
    return config


def _save_config_file(values: dict, hermes_home: str) -> None:
    config_path = Path(hermes_home) / "weaviate_engram.json"
    existing: dict = {}
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw
        except Exception:
            existing = {}
    existing.update(values)
    config_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_for_capture(text: str) -> str:
    return _CONTEXT_STRIP_RE.sub("", text or "").strip()


def _is_trivial(text: str) -> bool:
    return bool(_TRIVIAL_RE.match((text or "").strip()))


def _format_recall_context(results: List[Dict[str, Any]], limit: int) -> str:
    results = results[:limit]
    if not results:
        return ""
    lines: List[str] = []
    for item in results:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        score = item.get("score")
        if score is not None:
            try:
                lines.append(f"- [{round(float(score) * 100)}%] {content}")
                continue
            except (TypeError, ValueError):
                pass
        lines.append(f"- {content}")
    if not lines:
        return ""
    intro = (
        "Background context from long-term memory (Weaviate Engram). "
        "Use silently when relevant; do not force memories into the conversation."
    )
    return f"<engram-context>\n{intro}\n\n" + "\n".join(lines) + "\n</engram-context>"


STORE_SCHEMA = {
    "name": "engram_store",
    "description": (
        "Store an explicit memory in Weaviate Engram for future recall. "
        "To 'forget' or correct an earlier memory, store a new memory that "
        "explicitly states the correction (e.g. 'Correction: the user no "
        "longer works at X, they now work at Y'). Engram's reconcile "
        "pipeline supersedes older memories with newer correcting ones — "
        "there is no separate delete tool by design."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory content to store."},
            "metadata": {"type": "object", "description": "Optional metadata attached to the memory."},
        },
        "required": ["content"],
    },
}

SEARCH_SCHEMA = {
    "name": "engram_search",
    "description": "Search long-term memory in Weaviate Engram by semantic similarity.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {"type": "integer", "description": "Maximum results to return, 1 to 20."},
        },
        "required": ["query"],
    },
}

FETCH_SCHEMA = {
    "name": "engram_fetch",
    "description": (
        "Profile-shaped recall — 'what do you know about me?'. "
        "Returns top memories scoped to the active user. Pass an optional "
        "query to focus the recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Optional query to focus the profile recall."},
        },
    },
}


class WeaviateEngramMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._config: dict = _default_config()
        self._api_key: str = ""
        self._base_url: str = ""
        self._client: Optional[EngramClient] = None
        self._user_id: str = ""
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._write_enabled: bool = True
        self._active: bool = False
        self._auto_recall: bool = True
        self._auto_capture: bool = True
        self._max_recall_results: int = _DEFAULT_MAX_RECALL_RESULTS
        self._min_capture_chars: int = _DEFAULT_MIN_CAPTURE_CHARS
        self._api_timeout: float = _DEFAULT_API_TIMEOUT
        self._pipeline_hint: str = ""
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "weaviate_engram"

    def is_available(self) -> bool:
        # Cheap. No network. Check env var + SDK presence.
        if not os.environ.get("ENGRAM_API_KEY"):
            return False
        return is_sdk_available()

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "Engram API key",
                "secret": True,
                "required": True,
                "env_var": "ENGRAM_API_KEY",
                "url": "https://weaviate.io/engram",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        # Secrets (api_key) go to .env via the wizard. Only non-secret keys land here.
        sanitized = {k: v for k, v in (values or {}).items() if k != "api_key"}
        _save_config_file(sanitized, hermes_home)

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Called by ``hermes memory setup`` after credentials are saved.

        We do not block on a live API probe — that adds latency and noise
        to the wizard and the user can validate via ``hermes memory status``
        later. Instead, print a short orientation pointing at the things
        they may want to tune next.
        """
        sdk_ok = is_sdk_available()
        lines = ["", "Weaviate Engram plugin configured."]
        if not sdk_ok:
            lines += [
                "",
                "  WARNING: the `engram` SDK is not yet installed.",
                "  Install it with:  pip install weaviate-engram",
            ]
        cfg_path = Path(hermes_home) / "weaviate_engram.json"
        lines += [
            "",
            "Next steps:",
            f"  - Optional settings live in {cfg_path}",
            "  - Pipelines (extraction / reconciliation) are configured in the",
            "    Engram console, not from Hermes — see https://weaviate.io/engram",
            "  - Tools exposed to the agent: engram_search, engram_store, engram_fetch",
            "  - Forgetting is purposeful: store a correcting memory; do not expect a delete tool.",
            "",
        ]
        print("\n".join(lines))

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home

        self._hermes_home = kwargs.get("hermes_home") or str(get_hermes_home())
        self._session_id = session_id

        self._config = _load_config(self._hermes_home)
        self._auto_recall = self._config["auto_recall"]
        self._auto_capture = self._config["auto_capture"]
        self._max_recall_results = self._config["max_recall_results"]
        self._min_capture_chars = self._config["min_capture_chars"]
        self._api_timeout = self._config["api_timeout"]
        self._pipeline_hint = self._config["pipeline_hint"]

        self._api_key = os.environ.get("ENGRAM_API_KEY", "")
        self._base_url = os.environ.get("ENGRAM_BASE_URL", "").strip()

        # user_id resolution: pinned env var > template > "default".
        pinned = os.environ.get("WEAVIATE_ENGRAM_USER_ID", "").strip()
        if pinned:
            self._user_id = _sanitize_user_id(pinned)
        else:
            identity = kwargs.get("agent_identity") or "default"
            # kwargs["user_id"] (gateway sessions) takes priority over template.
            gateway_user = kwargs.get("user_id")
            if gateway_user:
                self._user_id = _sanitize_user_id(str(gateway_user))
            else:
                template = self._config["user_id_template"]
                self._user_id = _sanitize_user_id(template.replace("{identity}", str(identity)))

        # Subagents / cron / flush contexts must not write to user memory.
        agent_context = kwargs.get("agent_context", "")
        self._write_enabled = agent_context not in {"subagent", "cron", "flush"}

        # Only activate if we have the API key AND the SDK is installed.
        self._active = bool(self._api_key and is_sdk_available())
        self._client = None
        if self._active:
            try:
                self._client = EngramClient(
                    api_key=self._api_key,
                    base_url=self._base_url or None,
                    timeout=self._api_timeout,
                )
            except EngramNotAvailableError:
                logger.info(
                    "Weaviate Engram SDK not installed; provider will stay inactive. "
                    "Install with: pip install weaviate-engram"
                )
                self._active = False
            except Exception:
                logger.warning("Weaviate Engram client initialization failed", exc_info=True)
                self._active = False

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        lines = [
            "# Weaviate Engram",
            f"Long-term memory active. User scope: {self._user_id}.",
            "Use engram_search, engram_store, and engram_fetch for explicit memory operations.",
            (
                "Forgetting is server-side (purposeful forgetting): to correct or "
                "remove a memory, store a new memory that states the correction. "
                "There is no separate delete tool."
            ),
        ]
        if self._pipeline_hint:
            lines.append(f"\nActive pipeline: {self._pipeline_hint}")
        return "\n".join(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._active or not self._auto_recall or not self._client or not query.strip():
            return ""
        try:
            results = self._client.search_memories(
                query[:500],
                user_id=self._user_id,
                limit=self._max_recall_results,
            )
            return _format_recall_context(results, self._max_recall_results)
        except Exception:
            logger.debug("Weaviate Engram prefetch failed", exc_info=True)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._active or not self._auto_capture or not self._write_enabled or not self._client:
            return

        clean_user = _clean_for_capture(user_content)
        clean_assistant = _clean_for_capture(assistant_content)
        if not clean_user or not clean_assistant:
            return
        if len(clean_user) < self._min_capture_chars or len(clean_assistant) < self._min_capture_chars:
            return
        if _is_trivial(clean_user):
            return

        # Engram's add() takes a single text payload. Format the turn as a
        # readable two-role transcript so the server-side extraction
        # pipeline can see who said what.
        text = f"User: {clean_user}\nAssistant: {clean_assistant}"
        client = self._client
        user_id = self._user_id

        def _run() -> None:
            try:
                client.add_memory(text, user_id=user_id)
            except Exception:
                logger.debug("Weaviate Engram sync_turn failed", exc_info=True)

        # One in-flight sync at a time — drain previous before launching next.
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=2.0)
        self._sync_thread = threading.Thread(target=_run, daemon=True, name="weaviate-engram-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, STORE_SCHEMA, FETCH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._active or not self._client:
            return tool_error("Weaviate Engram is not configured or SDK unavailable")
        if tool_name == "engram_search":
            return self._tool_search(args)
        if tool_name == "engram_store":
            return self._tool_store(args)
        if tool_name == "engram_fetch":
            return self._tool_fetch(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def _tool_search(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        try:
            limit = max(1, min(20, int(args.get("limit", 5) or 5)))
        except Exception:
            limit = 5
        try:
            results = self._client.search_memories(query, user_id=self._user_id, limit=limit)
        except Exception as exc:
            return tool_error(f"Search failed: {exc}")
        formatted: List[Dict[str, Any]] = []
        for item in results:
            entry: Dict[str, Any] = {"id": item.get("id", ""), "content": item.get("content", "")}
            score = item.get("score")
            if score is not None:
                try:
                    entry["score"] = round(float(score) * 100)
                except (TypeError, ValueError):
                    pass
            formatted.append(entry)
        return json.dumps({"results": formatted, "count": len(formatted)})

    def _tool_store(self, args: Dict[str, Any]) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        try:
            run = self._client.add_memory(content, user_id=self._user_id)
        except Exception as exc:
            return tool_error(f"Store failed: {exc}")
        preview = content[:80] + ("..." if len(content) > 80 else "")
        return json.dumps({"saved": True, "run_id": run.get("run_id", ""), "preview": preview})

    def _tool_fetch(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "user profile facts").strip() or "user profile facts"
        try:
            results = self._client.search_memories(
                query, user_id=self._user_id, limit=self._max_recall_results
            )
        except Exception as exc:
            return tool_error(f"Fetch failed: {exc}")
        sections = [(item.get("content") or "").strip() for item in results if (item.get("content") or "").strip()]
        if not sections:
            return json.dumps({"profile": "", "count": 0})
        body = "\n".join(f"- {line}" for line in sections)
        return json.dumps({"profile": body, "count": len(sections)})

    def shutdown(self) -> None:
        thread = self._sync_thread
        if thread and thread.is_alive():
            thread.join(timeout=5.0)
        self._sync_thread = None
        client = self._client
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.debug("Engram client close raised", exc_info=True)
        self._client = None


def register(ctx) -> None:
    """Plugin discovery entry point."""
    ctx.register_memory_provider(WeaviateEngramMemoryProvider())
