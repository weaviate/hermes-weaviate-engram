"""Tests for the Weaviate Engram memory provider.

Phase 1 coverage:
- is_available env + SDK gating
- initialize: user_id resolution, write_enabled gating
- prefetch: gated by auto_recall, fences output
- sync_turn: background ingest, fence stripping, trivial-message skip
- tools: search/store/fetch behavior, no forget tool exposed (negative test)
- shutdown: joins threads
- save_config: never writes api_key
"""

from __future__ import annotations

import json

import pytest

from weaviate_engram import (
    SEARCH_SCHEMA,
    STORE_SCHEMA,
    FETCH_SCHEMA,
    WeaviateEngramMemoryProvider,
    _clean_for_capture,
    _format_recall_context,
    _is_trivial,
    _load_config,
    _sanitize_user_id,
    _save_config_file,
)


# ---------------------------------------------------------------------------
# Fake Engram client — replaces weaviate_engram.EngramClient
# ---------------------------------------------------------------------------


class FakeEngramClient:
    """Drop-in stand-in for EngramClient. Records calls; returns canned data."""

    def __init__(self, api_key: str, *, base_url=None, timeout: float = 10.0):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.add_calls: list = []
        self.search_calls: list = []
        self.search_results: list = []
        self.raise_on_add: Exception | None = None
        self.raise_on_search: Exception | None = None
        self.closed = False

    def close(self):
        self.closed = True

    def add_memory(self, text, *, user_id, properties=None):
        if self.raise_on_add:
            raise self.raise_on_add
        self.add_calls.append({"text": text, "user_id": user_id, "properties": properties})
        return {"run_id": f"run_{len(self.add_calls)}", "status": "queued"}

    def search_memories(self, query, *, user_id, limit=10):
        if self.raise_on_search:
            raise self.raise_on_search
        self.search_calls.append({"query": query, "user_id": user_id, "limit": limit})
        return self.search_results


@pytest.fixture
def env_credentials(monkeypatch):
    monkeypatch.setenv("ENGRAM_API_KEY", "test-key")
    yield
    monkeypatch.delenv("WEAVIATE_ENGRAM_USER_ID", raising=False)
    monkeypatch.delenv("ENGRAM_BASE_URL", raising=False)


@pytest.fixture
def provider(monkeypatch, env_credentials, tmp_path):
    monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: True)
    monkeypatch.setattr("weaviate_engram.EngramClient", FakeEngramClient)
    p = WeaviateEngramMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_identity="default")
    return p


def _drain(provider, attr="_sync_thread", timeout=2.0):
    thread = getattr(provider, attr)
    if thread:
        thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_false_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ENGRAM_API_KEY", raising=False)
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: True)
        assert WeaviateEngramMemoryProvider().is_available() is False

    def test_false_without_sdk(self, monkeypatch, env_credentials):
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: False)
        assert WeaviateEngramMemoryProvider().is_available() is False

    def test_true_when_key_and_sdk_present(self, monkeypatch, env_credentials):
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: True)
        assert WeaviateEngramMemoryProvider().is_available() is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_sanitize_user_id_strips_unsafe(self):
        assert _sanitize_user_id("alice@example.com") == "alice_example_com"
        assert _sanitize_user_id("hello world!") == "hello_world"
        assert _sanitize_user_id("---abc---") == "abc"
        assert _sanitize_user_id("") == "default"
        assert _sanitize_user_id("Alice_Smith-1") == "Alice_Smith-1"

    def test_is_trivial_matches_short_acknowledgements(self):
        assert _is_trivial("ok") is True
        assert _is_trivial("thanks.") is True
        assert _is_trivial("THX") is True
        assert _is_trivial("This is a real question") is False

    def test_clean_for_capture_strips_engram_fence(self):
        # Regex greedily consumes trailing whitespace after the fence — matches
        # Supermemory's pattern. The fence acts as a separator.
        text = "real question\n<engram-context>past memory\nblah</engram-context>\n\ntail"
        out = _clean_for_capture(text)
        assert "engram-context" not in out
        assert "real question" in out
        assert "tail" in out

    def test_clean_for_capture_preserves_mid_string_content_after_fence(self):
        # Pins down the actual behavior: the \s* after </engram-context> only
        # eats whitespace, never alphanumerics. Any non-whitespace text that
        # follows the fence (in a mid-string fence layout) must be preserved.
        text = (
            "leading sentence.\n"
            "<engram-context>recalled stuff</engram-context>   "
            "more text the assistant wrote AFTER the fence."
        )
        out = _clean_for_capture(text)
        assert "engram-context" not in out
        assert "leading sentence." in out
        assert "more text the assistant wrote AFTER the fence." in out

    def test_clean_for_capture_strips_multiple_fences(self):
        text = (
            "intro\n"
            "<engram-context>first</engram-context>\n"
            "middle paragraph stays\n"
            "<engram-context>second</engram-context>\n"
            "outro"
        )
        out = _clean_for_capture(text)
        assert out.count("engram-context") == 0
        assert "intro" in out
        assert "middle paragraph stays" in out
        assert "outro" in out
        assert "first" not in out and "second" not in out

    def test_coerce_properties_stringifies_values(self):
        from weaviate_engram import _coerce_properties

        assert _coerce_properties({"category": "preference", "weight": 7}) == {
            "category": "preference",
            "weight": "7",
        }
        assert _coerce_properties({}) is None
        assert _coerce_properties(None) is None
        assert _coerce_properties("not a dict") is None
        assert _coerce_properties({"keep": "yes", "drop": None}) == {"keep": "yes"}

    def test_format_recall_context_empty_returns_empty_string(self):
        assert _format_recall_context([], limit=10) == ""

    def test_format_recall_context_includes_fence_and_score(self):
        result = _format_recall_context(
            [{"content": "User likes terse answers", "score": 0.87}],
            limit=10,
        )
        assert "<engram-context>" in result
        assert "</engram-context>" in result
        assert "User likes terse answers" in result
        assert "[87%]" in result

    def test_load_and_save_config_round_trip(self, tmp_path):
        _save_config_file(
            {"user_id_template": "shared", "auto_capture": False, "max_recall_results": 5},
            str(tmp_path),
        )
        cfg = _load_config(str(tmp_path))
        assert cfg["user_id_template"] == "shared"
        assert cfg["auto_capture"] is False
        assert cfg["max_recall_results"] == 5
        assert cfg["auto_recall"] is True  # default preserved

    def test_load_config_clamps_out_of_range_values(self, tmp_path):
        _save_config_file({"max_recall_results": 999, "api_timeout": 999.0}, str(tmp_path))
        cfg = _load_config(str(tmp_path))
        assert cfg["max_recall_results"] == 20
        assert cfg["api_timeout"] == 60.0


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_resolves_user_id_from_identity_template(self, provider):
        # Default config has template "{identity}"; agent_identity="default" was passed.
        assert provider._user_id == "default"
        assert provider._active is True

    def test_resolves_user_id_with_explicit_identity(self, monkeypatch, env_credentials, tmp_path):
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: True)
        monkeypatch.setattr("weaviate_engram.EngramClient", FakeEngramClient)
        p = WeaviateEngramMemoryProvider()
        p.initialize("s", hermes_home=str(tmp_path), agent_identity="coder")
        assert p._user_id == "coder"

    def test_env_var_pins_user_id_over_template(self, monkeypatch, env_credentials, tmp_path):
        monkeypatch.setenv("WEAVIATE_ENGRAM_USER_ID", "pinned-user!")
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: True)
        monkeypatch.setattr("weaviate_engram.EngramClient", FakeEngramClient)
        p = WeaviateEngramMemoryProvider()
        p.initialize("s", hermes_home=str(tmp_path), agent_identity="coder")
        assert p._user_id == "pinned-user"

    def test_gateway_user_id_kwarg_overrides_template(self, monkeypatch, env_credentials, tmp_path):
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: True)
        monkeypatch.setattr("weaviate_engram.EngramClient", FakeEngramClient)
        p = WeaviateEngramMemoryProvider()
        p.initialize("s", hermes_home=str(tmp_path), agent_identity="coder", user_id="telegram:42")
        assert p._user_id == "telegram_42"

    def test_subagent_disables_writes(self, monkeypatch, env_credentials, tmp_path):
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: True)
        monkeypatch.setattr("weaviate_engram.EngramClient", FakeEngramClient)
        p = WeaviateEngramMemoryProvider()
        p.initialize("s", hermes_home=str(tmp_path), agent_context="subagent")
        assert p._write_enabled is False

    @pytest.mark.parametrize("ctx", ["cron", "flush", "subagent"])
    def test_non_primary_context_disables_writes(self, monkeypatch, env_credentials, tmp_path, ctx):
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: True)
        monkeypatch.setattr("weaviate_engram.EngramClient", FakeEngramClient)
        p = WeaviateEngramMemoryProvider()
        p.initialize("s", hermes_home=str(tmp_path), agent_context=ctx)
        assert p._write_enabled is False

    def test_inactive_when_sdk_missing(self, monkeypatch, env_credentials, tmp_path):
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: False)
        p = WeaviateEngramMemoryProvider()
        p.initialize("s", hermes_home=str(tmp_path))
        assert p._active is False
        assert p._client is None


# ---------------------------------------------------------------------------
# system_prompt_block
# ---------------------------------------------------------------------------


class TestSystemPromptBlock:
    def test_empty_when_inactive(self, monkeypatch, env_credentials, tmp_path):
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: False)
        p = WeaviateEngramMemoryProvider()
        p.initialize("s", hermes_home=str(tmp_path))
        assert p.system_prompt_block() == ""

    def test_active_block_mentions_user_id_and_no_forget(self, provider):
        block = provider.system_prompt_block()
        assert "Weaviate Engram" in block
        assert "default" in block  # the user_id
        assert "engram_search" in block
        assert "engram_store" in block
        assert "engram_fetch" in block
        # The block must communicate purposeful forgetting.
        assert "forgetting" in block.lower() or "forget" in block.lower()

    def test_pipeline_hint_appears_when_set(self, monkeypatch, env_credentials, tmp_path):
        _save_config_file({"pipeline_hint": "personalization v2"}, str(tmp_path))
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: True)
        monkeypatch.setattr("weaviate_engram.EngramClient", FakeEngramClient)
        p = WeaviateEngramMemoryProvider()
        p.initialize("s", hermes_home=str(tmp_path), agent_identity="default")
        assert "personalization v2" in p.system_prompt_block()


# ---------------------------------------------------------------------------
# prefetch
# ---------------------------------------------------------------------------


class TestPrefetch:
    def test_returns_empty_when_inactive(self, monkeypatch, env_credentials, tmp_path):
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: False)
        p = WeaviateEngramMemoryProvider()
        p.initialize("s", hermes_home=str(tmp_path))
        assert p.prefetch("hello?") == ""

    def test_returns_empty_when_auto_recall_off(self, monkeypatch, env_credentials, tmp_path):
        _save_config_file({"auto_recall": False}, str(tmp_path))
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: True)
        monkeypatch.setattr("weaviate_engram.EngramClient", FakeEngramClient)
        p = WeaviateEngramMemoryProvider()
        p.initialize("s", hermes_home=str(tmp_path), agent_identity="default")
        assert p.prefetch("query") == ""

    def test_returns_fenced_context_with_results(self, provider):
        provider._client.search_results = [
            {"content": "User prefers terse replies", "score": 0.92},
            {"content": "Working on the engram plugin", "score": 0.78},
        ]
        out = provider.prefetch("what am I doing?")
        assert "<engram-context>" in out
        assert "</engram-context>" in out
        assert "User prefers terse replies" in out
        assert "Working on the engram plugin" in out
        # search was called with the resolved user_id
        assert provider._client.search_calls[-1]["user_id"] == "default"

    def test_swallows_search_exceptions(self, provider):
        provider._client.raise_on_search = RuntimeError("network down")
        assert provider.prefetch("hello?") == ""


# ---------------------------------------------------------------------------
# sync_turn
# ---------------------------------------------------------------------------


class TestSyncTurn:
    def test_ingests_turn_as_single_text_in_background(self, provider):
        provider.sync_turn(
            "Tell me about Hermes.",
            "Hermes is a self-improving agent built by Nous Research.",
        )
        _drain(provider)
        assert len(provider._client.add_calls) == 1
        call = provider._client.add_calls[0]
        assert call["user_id"] == "default"
        assert call["text"] == (
            "User: Tell me about Hermes.\n"
            "Assistant: Hermes is a self-improving agent built by Nous Research."
        )

    def test_strips_engram_fence_from_assistant_content(self, provider):
        provider.sync_turn(
            "What do you remember about me?",
            "<engram-context>recalled stuff</engram-context>\n\nYou like terse replies.",
        )
        _drain(provider)
        assert len(provider._client.add_calls) == 1
        text = provider._client.add_calls[0]["text"]
        assert "engram-context" not in text
        assert text.endswith("Assistant: You like terse replies.")

    def test_skips_trivial_user_messages(self, provider):
        provider.sync_turn("ok", "Glad to hear it. Anything else I can help with?")
        _drain(provider)
        assert provider._client.add_calls == []

    def test_skips_short_messages_below_min_chars(self, provider):
        provider.sync_turn("hi", "hey there friend, how are you doing today?")
        _drain(provider)
        assert provider._client.add_calls == []

    def test_skips_when_write_disabled(self, monkeypatch, env_credentials, tmp_path):
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: True)
        monkeypatch.setattr("weaviate_engram.EngramClient", FakeEngramClient)
        p = WeaviateEngramMemoryProvider()
        p.initialize("s", hermes_home=str(tmp_path), agent_context="subagent", agent_identity="default")
        p.sync_turn("a long enough question here", "a long enough answer here")
        _drain(p)
        assert p._client is None or p._client.add_calls == []

    def test_swallows_add_exceptions(self, provider):
        provider._client.raise_on_add = RuntimeError("boom")
        # Should not raise out of sync_turn.
        provider.sync_turn("a longer user message", "a longer assistant message")
        _drain(provider)
        # Nothing recorded, but no crash.
        assert provider._client.add_calls == []

    def test_sync_turn_skips_when_previous_thread_still_alive(self, provider, monkeypatch):
        """Bounded-thread behavior: if Engram is hung, drop the new ingest
        instead of stacking threads. The next turn will try again once the
        in-flight one drains."""
        import threading
        import time

        # Make add_memory block until we release it. The first sync_turn will
        # start a thread that sits on this event; the second sync_turn should
        # see prev.is_alive() and skip rather than spawn a second thread.
        release = threading.Event()
        original_add = provider._client.add_memory

        def slow_add(text, *, user_id, properties=None):
            release.wait(timeout=5.0)
            return original_add(text, user_id=user_id, properties=properties)

        monkeypatch.setattr(provider._client, "add_memory", slow_add)

        # Speed up the test by clamping the join timeout the provider uses;
        # we don't want to wait the real 2.0s in CI.
        from unittest.mock import patch
        provider.sync_turn("first user message here", "first assistant reply here")
        first_thread = provider._sync_thread
        assert first_thread is not None and first_thread.is_alive()

        # Patch Thread.join to return immediately so the test doesn't wait 2s.
        # The bounded-skip branch is what we actually want to exercise.
        with patch.object(threading.Thread, "join", lambda self, timeout=None: None):
            provider.sync_turn("second user message here", "second assistant reply here")
            # Same thread object — nothing new was spawned.
            assert provider._sync_thread is first_thread

        # Let the first thread complete so we don't leak.
        release.set()
        first_thread.join(timeout=5.0)
        # Only ONE add_memory call landed — the second was dropped on purpose.
        assert len(provider._client.add_calls) == 1


# ---------------------------------------------------------------------------
# get_tool_schemas + handle_tool_call
# ---------------------------------------------------------------------------


class TestTools:
    def test_exposes_three_tools(self, provider):
        names = {schema["name"] for schema in provider.get_tool_schemas()}
        assert names == {"engram_search", "engram_store", "engram_fetch"}

    def test_get_tool_schemas_returns_independent_copies(self, provider):
        """Mutating one returned schema must not corrupt the next call."""
        a = provider.get_tool_schemas()
        b = provider.get_tool_schemas()
        assert a is not b
        for sa, sb in zip(a, b):
            assert sa is not sb
            assert sa["parameters"] is not sb["parameters"]
        # Mutate one copy — the next call must still return pristine schemas.
        a[0]["parameters"]["properties"]["query"]["description"] = "MUTATED"
        c = provider.get_tool_schemas()
        assert c[0]["parameters"]["properties"]["query"]["description"] != "MUTATED"

    def test_all_tool_schemas_disallow_additional_properties(self, provider):
        """Defensive: the model can't smuggle unknown args past us."""
        for schema in provider.get_tool_schemas():
            assert schema["parameters"].get("additionalProperties") is False

    def test_no_forget_or_delete_tool_exposed(self, provider):
        """Locks in the purposeful-forgetting design (no client-side delete)."""
        names = {schema["name"] for schema in provider.get_tool_schemas()}
        forbidden = {"engram_forget", "engram_delete", "engram_remove", "engram_purge"}
        assert names.isdisjoint(forbidden), (
            "Engram uses purposeful (server-side) forgetting — no client-issued "
            "delete tool should ever be exposed."
        )

    def test_engram_store_tool_description_explains_correction_pattern(self):
        """The model needs to learn that storing a new memory is how you 'forget'."""
        desc = STORE_SCHEMA["description"].lower()
        assert "correct" in desc or "supersede" in desc
        assert "forget" in desc or "delete" in desc  # at least mentions the concept

    def test_handle_tool_call_returns_error_when_inactive(self, monkeypatch, env_credentials, tmp_path):
        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: False)
        p = WeaviateEngramMemoryProvider()
        p.initialize("s", hermes_home=str(tmp_path))
        result = json.loads(p.handle_tool_call("engram_search", {"query": "x"}))
        assert "error" in result

    def test_search_returns_formatted_results(self, provider):
        provider._client.search_results = [
            {"id": "m1", "content": "fact one", "score": 0.9},
            {"id": "m2", "content": "fact two", "score": 0.42},
        ]
        result = json.loads(provider.handle_tool_call("engram_search", {"query": "facts", "limit": 5}))
        assert result["count"] == 2
        assert result["results"][0] == {"id": "m1", "content": "fact one", "score": 90}
        assert result["results"][1]["score"] == 42

    def test_search_requires_query(self, provider):
        result = json.loads(provider.handle_tool_call("engram_search", {"query": ""}))
        assert "error" in result

    def test_search_clamps_upper_limit(self, provider):
        """Upper bound matters most — protects Engram from runaway requests."""
        provider._client.search_results = []
        provider.handle_tool_call("engram_search", {"query": "x", "limit": 999})
        assert provider._client.search_calls[-1]["limit"] == 20

    def test_search_clamps_negative_limit_to_one(self, provider):
        provider._client.search_results = []
        provider.handle_tool_call("engram_search", {"query": "x", "limit": -5})
        assert provider._client.search_calls[-1]["limit"] == 1

    def test_store_returns_run_id_and_preview(self, provider):
        result = json.loads(provider.handle_tool_call(
            "engram_store",
            {"content": "User moved to Lisbon in 2026"},
        ))
        assert result["saved"] is True
        assert result["run_id"] == "run_1"
        assert "Lisbon" in result["preview"]
        # The call lands in Engram as plain text scoped to the resolved user_id.
        call = provider._client.add_calls[-1]
        assert call["user_id"] == "default"
        assert call["text"] == "User moved to Lisbon in 2026"
        assert call["properties"] is None  # no metadata → no properties forwarded

    def test_store_forwards_metadata_as_engram_properties(self, provider):
        provider.handle_tool_call(
            "engram_store",
            {
                "content": "User prefers concise responses",
                "metadata": {"category": "preference", "weight": 7, "drop_me": None},
            },
        )
        call = provider._client.add_calls[-1]
        # Engram's `properties` field is dict[str, str]; values get stringified
        # and explicit Nones are dropped.
        assert call["properties"] == {"category": "preference", "weight": "7"}

    def test_store_requires_content(self, provider):
        result = json.loads(provider.handle_tool_call("engram_store", {"content": "  "}))
        assert "error" in result

    def test_fetch_returns_profile_blob(self, provider):
        provider._client.search_results = [
            {"id": "m1", "content": "Lives in Lisbon"},
            {"id": "m2", "content": "Loves Weaviate"},
        ]
        result = json.loads(provider.handle_tool_call("engram_fetch", {}))
        assert result["count"] == 2
        assert "Lives in Lisbon" in result["profile"]
        assert "Loves Weaviate" in result["profile"]

    def test_fetch_handles_empty(self, provider):
        provider._client.search_results = []
        result = json.loads(provider.handle_tool_call("engram_fetch", {}))
        assert result["count"] == 0
        assert result["profile"] == ""

    def test_tool_errors_dont_crash(self, provider):
        provider._client.raise_on_search = RuntimeError("boom")
        result = json.loads(provider.handle_tool_call("engram_search", {"query": "x"}))
        assert "error" in result
        provider._client.raise_on_add = RuntimeError("boom")
        result = json.loads(provider.handle_tool_call("engram_store", {"content": "test content here"}))
        assert "error" in result

    def test_unknown_tool_returns_error(self, provider):
        result = json.loads(provider.handle_tool_call("engram_forget", {}))
        assert "error" in result


# ---------------------------------------------------------------------------
# shutdown + save_config
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_shutdown_joins_sync_thread(self, provider):
        provider.sync_turn("a longer user message", "a longer assistant message")
        provider.shutdown()
        assert provider._sync_thread is None

    def test_save_config_never_writes_api_key(self, tmp_path):
        p = WeaviateEngramMemoryProvider()
        p.save_config({"api_key": "SHOULD-NEVER-LAND", "user_id_template": "shared"}, str(tmp_path))
        on_disk = (tmp_path / "weaviate_engram.json").read_text(encoding="utf-8")
        assert "SHOULD-NEVER-LAND" not in on_disk
        assert "api_key" not in on_disk
        assert "shared" in on_disk


# ---------------------------------------------------------------------------
# MemoryManager smoke test
# ---------------------------------------------------------------------------


class TestMemoryManagerIntegration:
    def test_register_and_run_one_turn(self, monkeypatch, env_credentials, tmp_path):
        from agent.memory_manager import MemoryManager

        monkeypatch.setattr("weaviate_engram.is_sdk_available", lambda: True)
        monkeypatch.setattr("weaviate_engram.EngramClient", FakeEngramClient)

        mgr = MemoryManager()
        provider = WeaviateEngramMemoryProvider()
        mgr.add_provider(provider)
        mgr.initialize_all(session_id="smoke-1", hermes_home=str(tmp_path),
                           platform="cli", agent_identity="default")

        # Schemas wired through to the manager.
        schema_names = {s["name"] for s in mgr.get_all_tool_schemas()}
        assert {"engram_search", "engram_store", "engram_fetch"}.issubset(schema_names)

        # sync_all shouldn't raise.
        mgr.sync_all("a real-ish user question", "a real-ish assistant response")
        _drain(provider)
        assert len(provider._client.add_calls) == 1

        mgr.shutdown_all()
