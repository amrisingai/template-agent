"""Unit tests for deep_agent.src.capability.tool_proxy."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from deep_agent.src.capability.manifest import EXPLICIT, CapabilityManifest
from deep_agent.src.capability.tool_proxy import (
    CAPABILITY_DENIED_RESULT,
    CapabilityToolProxy,
    _get_tool_call_id,
    _make_denied_result,
    enforce_capability,
)


def _make_inner_tool(name="my_tool", description="does stuff"):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.args_schema = None
    return tool


def _manifest(allowed: set[str], agent_name="orchestrator", source=EXPLICIT):
    return CapabilityManifest(
        agent_name=agent_name, allowed_tool_names=frozenset(allowed), source=source
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestGetToolCallId:
    def test_returns_id_from_dict(self):
        assert _get_tool_call_id({"id": "abc-123"}) == "abc-123"

    def test_returns_empty_for_non_dict(self):
        assert _get_tool_call_id("nope") == ""


class TestMakeDeniedResult:
    def test_returns_error_tool_message(self):
        result = _make_denied_result("dangerous_tool", {"id": "call-1"})
        assert isinstance(result, ToolMessage)
        assert result.content == CAPABILITY_DENIED_RESULT
        assert result.name == "dangerous_tool"
        assert result.tool_call_id == "call-1"
        assert result.status == "error"


# ---------------------------------------------------------------------------
# CapabilityToolProxy
# ---------------------------------------------------------------------------


class TestCapabilityToolProxyInit:
    def test_copies_name_and_description(self):
        inner = _make_inner_tool(name="searcher", description="searches stuff")
        proxy = CapabilityToolProxy(inner, _manifest({"searcher"}))
        assert proxy.name == "searcher"
        assert proxy.description == "searches stuff"

    def test_stores_inner_tool_and_manifest(self):
        inner = _make_inner_tool()
        manifest = _manifest({"my_tool"})
        proxy = CapabilityToolProxy(inner, manifest)
        assert proxy._inner is inner
        assert proxy._manifest is manifest


class TestCapabilityToolProxyAinvoke:
    @pytest.mark.asyncio
    async def test_allowed_tool_delegates_to_inner(self):
        inner = _make_inner_tool(name="search")
        safe_result = ToolMessage(content="ok", name="search", tool_call_id="id-1")
        inner.ainvoke = AsyncMock(return_value=safe_result)
        proxy = CapabilityToolProxy(inner, _manifest({"search"}))

        result = await proxy.ainvoke({"id": "id-1"})

        assert result is safe_result
        inner.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_disallowed_tool_is_denied_without_calling_inner(self):
        inner = _make_inner_tool(name="delete_everything")
        inner.ainvoke = AsyncMock(return_value="should never be returned")
        proxy = CapabilityToolProxy(inner, _manifest({"search"}))

        with patch("deep_agent.src.audit.emitter.emit_audit_event") as emit:
            result = await proxy.ainvoke({"id": "call-9"})

        assert isinstance(result, ToolMessage)
        assert result.content == CAPABILITY_DENIED_RESULT
        assert result.status == "error"
        assert result.tool_call_id == "call-9"
        inner.ainvoke.assert_not_called()
        emit.assert_called_once()
        assert emit.call_args.args[0] == "capability_denied"
        assert emit.call_args.kwargs["tool"] == "delete_everything"

    @pytest.mark.asyncio
    async def test_denial_survives_audit_emit_failure(self):
        inner = _make_inner_tool(name="delete_everything")
        inner.ainvoke = AsyncMock()
        proxy = CapabilityToolProxy(inner, _manifest({"search"}))

        with patch(
            "deep_agent.src.audit.emitter.emit_audit_event",
            side_effect=RuntimeError("sink down"),
        ):
            result = await proxy.ainvoke({"id": "call-9"})

        assert isinstance(result, ToolMessage)
        assert result.content == CAPABILITY_DENIED_RESULT
        inner.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_manifest_denies_every_tool(self):
        inner = _make_inner_tool(name="anything")
        inner.ainvoke = AsyncMock()
        proxy = CapabilityToolProxy(inner, _manifest(set()))

        result = await proxy.ainvoke({"id": "call-1"})

        assert result.content == CAPABILITY_DENIED_RESULT
        inner.ainvoke.assert_not_called()


class TestCapabilityToolProxyRun:
    def test_allowed_tool_delegates_to_inner_invoke(self):
        inner = _make_inner_tool(name="search")
        inner.invoke = MagicMock(return_value="sync result")
        proxy = CapabilityToolProxy(inner, _manifest({"search"}))

        result = proxy._run("arg1", key="val")

        inner.invoke.assert_called_once_with("arg1", key="val")
        assert result == "sync result"

    def test_disallowed_tool_denied_without_calling_inner(self):
        inner = _make_inner_tool(name="delete_everything")
        inner.invoke = MagicMock(return_value="should never be returned")
        proxy = CapabilityToolProxy(inner, _manifest({"search"}))

        result = proxy._run()

        assert result == CAPABILITY_DENIED_RESULT
        inner.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# enforce_capability
# ---------------------------------------------------------------------------


class TestEnforceCapability:
    def test_returns_unchanged_when_tools_empty(self):
        manifest = _manifest({"search"})
        assert enforce_capability([], manifest) == []

    def test_wraps_every_tool_with_the_same_manifest(self):
        t1 = _make_inner_tool(name="tool_a")
        t2 = _make_inner_tool(name="tool_b")
        manifest = _manifest({"tool_a", "tool_b"})

        result = enforce_capability([t1, t2], manifest)

        assert len(result) == 2
        assert all(isinstance(r, CapabilityToolProxy) for r in result)
        assert result[0].name == "tool_a"
        assert result[1].name == "tool_b"
        assert all(r._manifest is manifest for r in result)

    @pytest.mark.asyncio
    async def test_wrapped_tool_outside_manifest_is_denied_end_to_end(self):
        """Defense-in-depth: even if a tool sneaks into the list without being
        in the manifest, dispatch is blocked."""
        rogue = _make_inner_tool(name="not_in_manifest")
        rogue.ainvoke = AsyncMock(return_value="leaked!")
        manifest = _manifest({"search"})  # rogue tool is not a member

        wrapped = enforce_capability([rogue], manifest)

        result = await wrapped[0].ainvoke({"id": "x"})

        assert result.content == CAPABILITY_DENIED_RESULT
        rogue.ainvoke.assert_not_called()
