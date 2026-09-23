"""Unit tests for deep_agent.aegra.safety."""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from deep_agent.aegra.safety import (
    SafetyAwareRunnable,
    _build_merged_config,
    safety_refusal,
)
from deep_agent.src.agent.repetition import max_window_chars
from deep_agent.src.guardrails import (
    ContentSafetyError,
    InputContentSafetyError,
    ToolContentSafetyError,
)
from deep_agent.src.guardrails import TOOL_SAFETY_REFUSAL as _TOOL_SAFETY_REFUSAL

_INPUT_SAFETY_REFUSAL = "I can't help with that request due to content safety policy."

# A repeated unit long/frequent enough to trip the default
# REPETITION_LOOP_MIN_UNIT_LEN=20 / REPETITION_LOOP_MIN_REPEATS=4 thresholds.
_REPEATED_UNIT = "I cannot verify that claim. "
_REPEATED_TEXT = _REPEATED_UNIT * 6


# ---------------------------------------------------------------------------
# safety_refusal
# ---------------------------------------------------------------------------


class TestSafetyRefusal:
    def test_tool_content_safety_error_returns_tool_refusal(self):
        exc = ToolContentSafetyError("blocked")
        assert safety_refusal(exc) == _TOOL_SAFETY_REFUSAL

    def test_input_content_safety_error_returns_input_refusal(self):
        exc = InputContentSafetyError("blocked")
        assert safety_refusal(exc) == _INPUT_SAFETY_REFUSAL

    def test_content_safety_error_returns_input_refusal(self):
        exc = ContentSafetyError("blocked")
        assert safety_refusal(exc) == _INPUT_SAFETY_REFUSAL

    def test_string_contains_tool_content_safety_error(self):
        exc = RuntimeError("ToolContentSafetyError: some message")
        assert safety_refusal(exc) == _TOOL_SAFETY_REFUSAL

    def test_string_contains_input_content_safety_error(self):
        exc = RuntimeError("InputContentSafetyError: blocked input")
        assert safety_refusal(exc) == _INPUT_SAFETY_REFUSAL

    def test_string_contains_content_safety_error(self):
        exc = RuntimeError("ContentSafetyError occurred")
        assert safety_refusal(exc) == _INPUT_SAFETY_REFUSAL

    def test_non_safety_exception_returns_none(self):
        exc = ValueError("random error")
        assert safety_refusal(exc) is None

    def test_chained_cause_is_safety_error(self):
        cause = ToolContentSafetyError("cause")
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = cause
        assert safety_refusal(wrapper) == _TOOL_SAFETY_REFUSAL

    def test_chained_context_is_safety_error(self):
        ctx = InputContentSafetyError("context")
        wrapper = RuntimeError("wrapper")
        wrapper.__context__ = ctx
        assert safety_refusal(wrapper) == _INPUT_SAFETY_REFUSAL

    def test_cycle_prevention_returns_none(self):
        exc = RuntimeError("no safety")
        exc.__cause__ = exc  # self-referential cycle
        assert safety_refusal(exc) is None


# ---------------------------------------------------------------------------
# _build_merged_config
# ---------------------------------------------------------------------------


class TestBuildMergedConfig:
    def test_none_config_creates_empty_base(self):
        merged, ctx = _build_merged_config(None)
        assert "_safety_ctx" in merged
        assert merged["_safety_ctx"] is ctx
        assert ctx == {"blocked": False}

    def test_existing_config_is_merged(self):
        merged, ctx = _build_merged_config({"run_name": "test"})
        assert merged["run_name"] == "test"
        assert merged["_safety_ctx"] is ctx

    def test_existing_metadata_is_preserved(self):
        merged, ctx = _build_merged_config({"metadata": {"user": "alice"}})
        assert merged["metadata"]["user"] == "alice"
        assert merged["metadata"]["_safety_ctx"] is ctx

    def test_safety_ctx_shared_between_config_and_metadata(self):
        merged, ctx = _build_merged_config({})
        assert merged["_safety_ctx"] is merged["metadata"]["_safety_ctx"]

    def test_safety_ctx_starts_unblocked(self):
        _, ctx = _build_merged_config(None)
        assert ctx["blocked"] is False


# ---------------------------------------------------------------------------
# SafetyAwareRunnable — sync interface
# ---------------------------------------------------------------------------


class TestSafetyAwareRunnableInit:
    def test_stores_runnable_and_outermost(self):
        inner = MagicMock()
        sar = SafetyAwareRunnable(inner, outermost=True)
        assert sar._runnable is inner
        assert sar._outermost is True

    def test_default_outermost_is_false(self):
        sar = SafetyAwareRunnable(MagicMock())
        assert sar._outermost is False

    def test_getattr_delegates_to_inner(self):
        inner = MagicMock()
        inner.some_attr = "value"
        sar = SafetyAwareRunnable(inner)
        assert sar.some_attr == "value"

    def test_copy_wraps_inner_copy(self):
        inner = MagicMock()
        inner.copy.return_value = MagicMock()
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = sar.copy(update={})
        assert isinstance(result, SafetyAwareRunnable)
        assert result._outermost is True
        inner.copy.assert_called_once_with(update={})

    def test_with_config_none_uses_kwargs_only(self):
        inner = MagicMock()
        inner.with_config.return_value = MagicMock()
        sar = SafetyAwareRunnable(inner)
        result = sar.with_config(None, tags=["x"])
        inner.with_config.assert_called_once_with(tags=["x"])
        assert isinstance(result, SafetyAwareRunnable)

    def test_with_config_non_none_passes_config(self):
        inner = MagicMock()
        inner.with_config.return_value = MagicMock()
        sar = SafetyAwareRunnable(inner)
        cfg = {"run_name": "r"}
        result = sar.with_config(cfg, tags=["x"])
        inner.with_config.assert_called_once_with(cfg, tags=["x"])
        assert isinstance(result, SafetyAwareRunnable)


# ---------------------------------------------------------------------------
# SafetyAwareRunnable.ainvoke
# ---------------------------------------------------------------------------


class TestSafetyAwareRunnableAinvoke:
    @pytest.mark.asyncio
    async def test_safe_result_returned_unchanged(self):
        ai = AIMessage(content="hello")
        result = {"messages": [ai]}
        inner = MagicMock()
        inner.ainvoke = AsyncMock(return_value=result)
        sar = SafetyAwareRunnable(inner)
        out = await sar.ainvoke({"input": "hi"})
        assert out["messages"][0].content == "hello"

    @pytest.mark.asyncio
    async def test_tool_blocked_via_safety_ctx_rewrites_last_ai_message(self):
        ai = AIMessage(content="original response")
        tm = ToolMessage(content="tool output", name="t", tool_call_id="c1")

        async def fake_ainvoke(input, config, **kwargs):
            # Simulate GuardianToolProxy setting blocked=True in safety_ctx
            config["_safety_ctx"]["blocked"] = True
            return {"messages": [tm, ai]}

        inner = MagicMock()
        inner.ainvoke = fake_ainvoke
        sar = SafetyAwareRunnable(inner)
        out = await sar.ainvoke({})
        last_ai = next(m for m in reversed(out["messages"]) if isinstance(m, AIMessage))
        assert last_ai.content == _TOOL_SAFETY_REFUSAL

    @pytest.mark.asyncio
    async def test_tool_blocked_via_sentinel_in_tool_message(self):
        ai = AIMessage(content="should be replaced")
        tm = ToolMessage(
            content=f"...{_TOOL_SAFETY_REFUSAL}...", name="t", tool_call_id="c1"
        )
        inner = MagicMock()
        inner.ainvoke = AsyncMock(return_value={"messages": [tm, ai]})
        sar = SafetyAwareRunnable(inner)
        out = await sar.ainvoke({})
        last_ai = next(m for m in reversed(out["messages"]) if isinstance(m, AIMessage))
        assert last_ai.content == _TOOL_SAFETY_REFUSAL

    @pytest.mark.asyncio
    async def test_non_outermost_reraises_exception(self):
        inner = MagicMock()
        inner.ainvoke = AsyncMock(side_effect=InputContentSafetyError("blocked"))
        sar = SafetyAwareRunnable(inner, outermost=False)
        with pytest.raises(InputContentSafetyError):
            await sar.ainvoke({})

    @pytest.mark.asyncio
    async def test_outermost_safety_exception_returns_refusal(self):
        inner = MagicMock()
        inner.ainvoke = AsyncMock(side_effect=InputContentSafetyError("blocked"))
        sar = SafetyAwareRunnable(inner, outermost=True)
        out = await sar.ainvoke({})
        assert isinstance(out["messages"][0], AIMessage)
        assert out["messages"][0].content == _INPUT_SAFETY_REFUSAL

    @pytest.mark.asyncio
    async def test_outermost_non_safety_exception_reraises(self):
        inner = MagicMock()
        inner.ainvoke = AsyncMock(side_effect=RuntimeError("unexpected"))
        sar = SafetyAwareRunnable(inner, outermost=True)
        with pytest.raises(RuntimeError, match="unexpected"):
            await sar.ainvoke({})

    @pytest.mark.asyncio
    async def test_outermost_tool_safety_exception_returns_tool_refusal(self):
        inner = MagicMock()
        inner.ainvoke = AsyncMock(side_effect=ToolContentSafetyError("tool blocked"))
        sar = SafetyAwareRunnable(inner, outermost=True)
        out = await sar.ainvoke({})
        assert out["messages"][0].content == _TOOL_SAFETY_REFUSAL

    @pytest.mark.asyncio
    async def test_non_dict_result_is_returned_without_rewrite(self):
        inner = MagicMock()
        inner.ainvoke = AsyncMock(return_value="plain string result")
        sar = SafetyAwareRunnable(inner)
        out = await sar.ainvoke({})
        assert out == "plain string result"


# ---------------------------------------------------------------------------
# SafetyAwareRunnable.ainvoke — repetition loop detection
# ---------------------------------------------------------------------------


class TestSafetyAwareRunnableAinvokeRepetition:
    @pytest.mark.asyncio
    async def test_repetition_loop_truncates_final_ai_message(self):
        ai = AIMessage(content="Sure, here is the answer. " + _REPEATED_TEXT)
        inner = MagicMock()
        inner.ainvoke = AsyncMock(return_value={"messages": [ai]})
        sar = SafetyAwareRunnable(inner)
        out = await sar.ainvoke({})
        last_ai = out["messages"][-1]
        assert isinstance(last_ai, AIMessage)
        assert len(last_ai.content) < len(ai.content)
        assert last_ai.content.count(_REPEATED_UNIT.strip()) == 1

    @pytest.mark.asyncio
    async def test_non_looping_response_passes_through_unchanged(self):
        ai = AIMessage(content="A perfectly normal, non-repetitive answer.")
        inner = MagicMock()
        inner.ainvoke = AsyncMock(return_value={"messages": [ai]})
        sar = SafetyAwareRunnable(inner)
        out = await sar.ainvoke({})
        assert out["messages"][-1].content == ai.content

    @pytest.mark.asyncio
    async def test_tool_block_takes_priority_over_repetition_check(self):
        ai = AIMessage(content=_REPEATED_TEXT)
        tm = ToolMessage(content="tool output", name="t", tool_call_id="c1")

        async def fake_ainvoke(input, config, **kwargs):
            config["_safety_ctx"]["blocked"] = True
            return {"messages": [tm, ai]}

        inner = MagicMock()
        inner.ainvoke = fake_ainvoke
        sar = SafetyAwareRunnable(inner)
        out = await sar.ainvoke({})
        last_ai = next(m for m in reversed(out["messages"]) if isinstance(m, AIMessage))
        assert last_ai.content == _TOOL_SAFETY_REFUSAL

    @pytest.mark.asyncio
    async def test_repetition_check_disabled_via_settings(self):
        ai = AIMessage(content=_REPEATED_TEXT)
        inner = MagicMock()
        inner.ainvoke = AsyncMock(return_value={"messages": [ai]})
        sar = SafetyAwareRunnable(inner)
        with patch(
            "deep_agent.aegra.safety.settings.REPETITION_LOOP_DETECTION_ENABLED",
            False,
        ):
            out = await sar.ainvoke({})
        assert out["messages"][-1].content == _REPEATED_TEXT

    @pytest.mark.asyncio
    async def test_repetition_check_runs_on_non_outermost_too(self):
        """Runs at every level, mirroring the tool-block override behavior."""
        ai = AIMessage(content=_REPEATED_TEXT)
        inner = MagicMock()
        inner.ainvoke = AsyncMock(return_value={"messages": [ai]})
        sar = SafetyAwareRunnable(inner, outermost=False)
        out = await sar.ainvoke({})
        assert len(out["messages"][-1].content) < len(_REPEATED_TEXT)

    @pytest.mark.asyncio
    async def test_truncation_preserves_message_identity_and_metadata(self):
        """model_copy() must keep id/tool_calls/response_metadata/usage_metadata
        intact — only `content` should change."""
        ai = AIMessage(
            content=_REPEATED_TEXT,
            id="run-123",
            tool_calls=[
                {"name": "lookup", "args": {"q": "x"}, "id": "call-1"},
            ],
            response_metadata={"model": "gemini-2-5-pro", "finish_reason": "stop"},
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 500,
                "total_tokens": 510,
            },
        )
        inner = MagicMock()
        inner.ainvoke = AsyncMock(return_value={"messages": [ai]})
        sar = SafetyAwareRunnable(inner)
        out = await sar.ainvoke({})
        last_ai = out["messages"][-1]
        assert last_ai is not ai  # a copy, not a mutation
        assert len(last_ai.content) < len(ai.content)
        assert last_ai.id == "run-123"
        assert last_ai.tool_calls == ai.tool_calls
        assert last_ai.response_metadata == ai.response_metadata
        assert last_ai.usage_metadata == ai.usage_metadata


# ---------------------------------------------------------------------------
# SafetyAwareRunnable.astream
# ---------------------------------------------------------------------------


async def _collect(agen):
    items = []
    async for item in agen:
        items.append(item)
    return items


class TestSafetyAwareRunnableAstream:
    @pytest.mark.asyncio
    async def test_yields_chunks_normally(self):
        chunks = [{"event": "chunk", "data": i} for i in range(3)]

        async def gen(*a, **kw):
            for c in chunks:
                yield c

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner)
        result = await _collect(sar.astream({}))
        assert result == chunks

    @pytest.mark.asyncio
    async def test_non_outermost_reraises_on_stream_exception(self):
        async def gen(*a, **kw):
            yield {"data": 1}
            raise InputContentSafetyError("blocked")

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner, outermost=False)
        with pytest.raises(InputContentSafetyError):
            await _collect(sar.astream({}))

    @pytest.mark.asyncio
    async def test_outermost_yields_refusal_on_safety_exception(self):
        async def gen(*a, **kw):
            yield {"data": 1}
            raise InputContentSafetyError("blocked")

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream({}))
        assert len(result) == 2
        event_type, (ai_msg, _) = result[-1]
        assert event_type == "messages"
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.content == _INPUT_SAFETY_REFUSAL

    @pytest.mark.asyncio
    async def test_outermost_reraises_non_safety_stream_exception(self):
        async def gen(*a, **kw):
            yield {"data": 1}
            raise RuntimeError("crash")

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        with pytest.raises(RuntimeError, match="crash"):
            await _collect(sar.astream({}))


# ---------------------------------------------------------------------------
# SafetyAwareRunnable.astream — repetition loop detection
# ---------------------------------------------------------------------------


class TestSafetyAwareRunnableAstreamRepetition:
    @pytest.mark.asyncio
    async def test_repetition_loop_truncates_and_stops_stream(self):
        # Split the repeated text into several "messages"-mode chunks, as a
        # real streaming model call would.
        parts = [_REPEATED_UNIT] * 6

        async def gen(*a, **kw):
            for part in parts:
                yield ("messages", (AIMessageChunk(content=part), {}))
            # Should never be reached — the proxy stops after detecting the loop.
            yield ("messages", (AIMessageChunk(content="should not appear"), {}))

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream({}))

        assert "should not appear" not in "".join(
            str(r[1][0].content) for r in result if isinstance(r, tuple)
        )
        # Chunks are buffered per-invocation, not forwarded live: none of the
        # individual repeats leak out ahead of the truncation decision, so
        # exactly one ("messages", ...) event — the truncated one — is ever
        # yielded, and its content contains exactly one copy of the unit.
        assert len(result) == 1
        last_event_type, (last_msg, _) = result[-1]
        assert last_event_type == "messages"
        assert last_msg.content.count(_REPEATED_UNIT.strip()) == 1
        combined = "".join(str(r[1][0].content) for r in result)
        assert combined.count(_REPEATED_UNIT.strip()) == 1

    @pytest.mark.asyncio
    async def test_repetition_buffer_scoped_per_model_invocation(self):
        """A short, non-repetitive first invocation must be flushed through
        unmodified once a second invocation (different run_id) starts, and
        the second invocation's loop must be detected using only its own
        text — mirroring the run_id-partitioning used in astream_events."""

        async def gen(*a, **kw):
            yield (
                "messages",
                (AIMessageChunk(content="Sure, one moment."), {"run_id": "run-1"}),
            )
            for part in [_REPEATED_UNIT] * 6:
                yield ("messages", (AIMessageChunk(content=part), {"run_id": "run-2"}))

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream({}))

        # First invocation's chunk flushed unchanged at the run_id boundary.
        first_type, (first_msg, first_meta) = result[0]
        assert first_type == "messages"
        assert first_msg.content == "Sure, one moment."
        assert first_meta == {"run_id": "run-1"}

        # Second invocation's loop truncated to a single copy of the unit.
        last_type, (last_msg, _) = result[-1]
        assert last_type == "messages"
        assert last_msg.content.count(_REPEATED_UNIT.strip()) == 1

    @pytest.mark.asyncio
    async def test_repetition_text_does_not_leak_across_invocations(self):
        """A prior invocation's near-threshold trailing text must not combine
        with an unrelated later invocation's text to produce a false-positive
        loop — each is a distinct model call, not one continuous generation."""

        async def gen(*a, **kw):
            # First invocation: 3 copies of the unit (below MIN_REPEATS=4) —
            # not a loop on its own.
            yield (
                "messages",
                (
                    AIMessageChunk(content=_REPEATED_UNIT * 3),
                    {"run_id": "run-1"},
                ),
            )
            # Second, unrelated invocation happens to start with one copy of
            # the same phrase — genuinely not a loop by itself.
            yield (
                "messages",
                (
                    AIMessageChunk(
                        content=_REPEATED_UNIT + "Anyway, the summary is X."
                    ),
                    {"run_id": "run-2"},
                ),
            )

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream({}))

        combined = "".join(str(r[1][0].content) for r in result if isinstance(r, tuple))
        assert combined.count(_REPEATED_UNIT.strip()) == 4  # 3 + 1, none truncated
        assert "Anyway, the summary is X." in combined

    @pytest.mark.asyncio
    async def test_non_looping_messages_mode_chunks_pass_through(self):
        chunks = [
            ("messages", (AIMessageChunk(content="Hello "), {})),
            ("messages", (AIMessageChunk(content="world."), {})),
        ]

        async def gen(*a, **kw):
            for c in chunks:
                yield c

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream({}))
        assert result == chunks

    @pytest.mark.asyncio
    async def test_non_outermost_does_not_check_repetition(self):
        parts = [_REPEATED_UNIT] * 6

        async def gen(*a, **kw):
            for part in parts:
                yield ("messages", (AIMessageChunk(content=part), {}))

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner, outermost=False)
        result = await _collect(sar.astream({}))
        # All chunks pass through unmodified — no truncation at non-outermost level.
        assert len(result) == len(parts)


# ---------------------------------------------------------------------------
# SafetyAwareRunnable.astream — incremental delivery
#
# Regression coverage: with repetition detection enabled, a normal single
# invocation must not be buffered in full and released only once it ends.
# The confirmed (non-repeating) prefix must reach the caller as it streams
# in; only the detector-sized undecided suffix may remain buffered.
# ---------------------------------------------------------------------------


def _unique_chunks(n: int, size: int = 64) -> list[str]:
    """Return ``n`` distinct, non-repeating strings of ``size`` chars each.

    Built from per-index hash digests so concatenating them never produces a
    consecutively-repeated substring long enough to be mistaken for a
    degenerate loop by ``detect_repetition_loop``.
    """
    return [
        (hashlib.sha256(str(i).encode()).hexdigest() * ((size // 64) + 1))[:size]
        for i in range(n)
    ]


class TestSafetyAwareRunnableAstreamIncrementalDelivery:
    @pytest.mark.asyncio
    async def test_non_repeating_stream_yields_before_invocation_completes(self):
        """A normal (non-repeating) response must start reaching the caller
        while the model is still streaming, not only once the whole
        invocation has finished."""
        window = max_window_chars()  # default settings: 400 * 4 = 1600 chars
        chunk_size = 64
        # Comfortably more chunks than needed to exceed the detector's window,
        # so a flush must happen well before the source is exhausted.
        num_chunks = (window // chunk_size) + 15
        parts = _unique_chunks(num_chunks, size=chunk_size)

        produced = 0

        async def gen(*a, **kw):
            nonlocal produced
            for part in parts:
                produced += 1
                yield ("messages", (AIMessageChunk(content=part), {"run_id": "run-1"}))

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner, outermost=True)

        stream = sar.astream({})
        first_event = await stream.__anext__()

        # A chunk was already forwarded to the caller before the source
        # finished producing every chunk — i.e. delivery is incremental,
        # not withheld until the whole invocation completes.
        assert produced < num_chunks
        assert first_event[0] == "messages"
        assert first_event[1][0].content == parts[0]

        # Draining the rest must reproduce the full, unmodified text in order
        # — nothing lost, duplicated, or reordered by the windowed buffering.
        rest = [first_event]
        async for item in stream:
            rest.append(item)
        combined = "".join(str(r[1][0].content) for r in rest)
        assert combined == "".join(parts)

    @pytest.mark.asyncio
    async def test_repetition_still_detected_after_incremental_prefix_flushed(self):
        """A loop that starts only *after* a long non-repeating prefix has
        already been flushed incrementally must still be detected and
        truncated — the windowed buffering must not weaken detection."""
        window = max_window_chars()
        chunk_size = 64
        prefix_chunks = (window // chunk_size) + 15
        prefix_parts = _unique_chunks(prefix_chunks, size=chunk_size)
        looping_parts = [_REPEATED_UNIT] * 6

        async def gen(*a, **kw):
            for part in prefix_parts:
                yield ("messages", (AIMessageChunk(content=part), {"run_id": "run-1"}))
            for part in looping_parts:
                yield ("messages", (AIMessageChunk(content=part), {"run_id": "run-1"}))
            # Should never be reached — the proxy stops after detecting the loop.
            yield (
                "messages",
                (AIMessageChunk(content="should not appear"), {"run_id": "run-1"}),
            )

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream({}))

        combined = "".join(str(r[1][0].content) for r in result)
        assert "should not appear" not in combined
        assert combined.count(_REPEATED_UNIT.strip()) == 1
        # The incrementally-flushed prefix must still be present in full.
        assert combined.startswith("".join(prefix_parts))

    @pytest.mark.asyncio
    async def test_buffer_never_exceeds_detector_window_plus_one_chunk(self):
        """At no point should more than ``max_window_chars()`` (plus the
        latest chunk) of confirmed non-repeating text be withheld — proving
        the fix bounds memory/latency instead of buffering the whole
        invocation."""
        window = max_window_chars()
        chunk_size = 64
        num_chunks = (window // chunk_size) + 15
        parts = _unique_chunks(num_chunks, size=chunk_size)
        max_gap = 0
        produced = 0
        consumed = 0

        async def gen(*a, **kw):
            nonlocal produced
            for part in parts:
                produced += 1
                yield ("messages", (AIMessageChunk(content=part), {"run_id": "run-1"}))

        inner = MagicMock()
        inner.astream = gen
        sar = SafetyAwareRunnable(inner, outermost=True)

        async for _ in sar.astream({}):
            consumed += 1
            # How far ahead the source has produced relative to what's been
            # forwarded is bounded by the detector's window, not unbounded.
            max_gap = max(max_gap, produced - consumed)

        assert max_gap * chunk_size <= window + chunk_size


# ---------------------------------------------------------------------------
# SafetyAwareRunnable.astream_events
# ---------------------------------------------------------------------------


class TestSafetyAwareRunnableAstreamEvents:
    @pytest.mark.asyncio
    async def test_non_ai_events_pass_through_immediately(self):
        events = [
            {"event": "on_tool_start", "data": {}},
            {"event": "on_tool_end", "data": {"output": "ok"}},
            {"event": "on_chain_end", "data": {}},
        ]

        async def gen(*a, **kw):
            for e in events:
                yield e

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))
        # tool_start and chain_end passed through; tool_end also yielded
        assert any(e.get("event") == "on_tool_start" for e in result)
        assert any(e.get("event") == "on_chain_end" for e in result)

    @pytest.mark.asyncio
    async def test_ai_chunks_buffered_and_flushed_when_safe(self):
        chunk_event = {"event": "on_chat_model_stream", "data": {"chunk": "hi"}}
        other_event = {"event": "on_chain_end", "data": {}}

        async def gen(*a, **kw):
            yield chunk_event
            yield other_event

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))
        # chunk should be flushed at end (safe path)
        assert chunk_event in result

    @pytest.mark.asyncio
    async def test_blocked_tool_via_sentinel_emits_refusal_event(self):
        async def gen(*a, **kw):
            config = a[1] if len(a) > 1 else kw.get("config", {})
            yield {"event": "on_tool_start", "data": {}}
            yield {
                "event": "on_tool_end",
                "data": {"output": f"prefix {_TOOL_SAFETY_REFUSAL} suffix"},
            }
            # This should not be yielded — loop breaks after tool batch completes
            yield {"event": "on_chat_model_stream", "data": {"chunk": "dropped"}}

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))
        refusal_events = [e for e in result if e.get("name") == "guardian_refusal"]
        assert len(refusal_events) == 1
        assert isinstance(refusal_events[0]["data"]["chunk"], AIMessage)

    @pytest.mark.asyncio
    async def test_non_outermost_does_not_track_tool_calls(self):
        chunk = {"event": "on_chat_model_stream", "data": {"chunk": "x"}}

        async def gen(*a, **kw):
            yield chunk

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=False)
        result = await _collect(sar.astream_events({}))
        assert chunk in result

    @pytest.mark.asyncio
    async def test_outermost_safety_exception_yields_refusal_event(self):
        async def gen(*a, **kw):
            raise InputContentSafetyError("blocked")
            yield  # noqa: unreachable — makes this an async generator

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))
        assert len(result) == 1
        assert result[0]["name"] == "guardian_refusal"

    @pytest.mark.asyncio
    async def test_outermost_non_safety_exception_reraises(self):
        async def gen(*a, **kw):
            raise RuntimeError("crash")
            yield  # noqa: unreachable

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        with pytest.raises(RuntimeError, match="crash"):
            await _collect(sar.astream_events({}))

    @pytest.mark.asyncio
    async def test_blocked_via_safety_ctx_emits_refusal_instead_of_ai_chunks(self):
        chunk_event = {"event": "on_chat_model_stream", "data": {"chunk": "response"}}

        async def gen(*a, **kw):
            config = a[1]
            config["_safety_ctx"]["blocked"] = True
            yield chunk_event

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))
        refusal_events = [e for e in result if e.get("name") == "guardian_refusal"]
        assert len(refusal_events) == 1
        assert chunk_event not in result


# ---------------------------------------------------------------------------
# SafetyAwareRunnable.astream_events — repetition loop detection
# ---------------------------------------------------------------------------


class TestSafetyAwareRunnableAstreamEventsRepetition:
    @pytest.mark.asyncio
    async def test_repetition_loop_breaks_stream_and_emits_truncated_chunk(self):
        parts = [_REPEATED_UNIT] * 6

        async def gen(*a, **kw):
            for part in parts:
                yield {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": AIMessageChunk(content=part)},
                }
            # Should never be reached — the proxy breaks out before this.
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="should not appear")},
            }

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))

        truncated_events = [
            e for e in result if e.get("name") == "repetition_loop_truncated"
        ]
        assert len(truncated_events) == 1
        chunk = truncated_events[0]["data"]["chunk"]
        assert isinstance(chunk, AIMessage)
        assert chunk.content.count(_REPEATED_UNIT.strip()) == 1
        assert "should not appear" not in chunk.content
        # No guardian refusal — this is a repetition-loop event, not a safety block.
        assert not any(e.get("name") == "guardian_refusal" for e in result)

    @pytest.mark.asyncio
    async def test_repetition_text_does_not_leak_across_run_ids(self):
        """A prior model invocation's near-threshold trailing text must not
        combine with a later, unrelated invocation's text to produce a
        false-positive loop — each on_chat_model_stream run_id is a distinct
        model call."""

        async def gen(*a, **kw):
            # First invocation: 3 copies of the unit (below MIN_REPEATS=4).
            yield {
                "event": "on_chat_model_stream",
                "run_id": "run-1",
                "data": {"chunk": AIMessageChunk(content=_REPEATED_UNIT * 3)},
            }
            # Second, unrelated invocation starts with one copy of the same
            # phrase — genuinely not a loop by itself.
            yield {
                "event": "on_chat_model_stream",
                "run_id": "run-2",
                "data": {
                    "chunk": AIMessageChunk(
                        content=_REPEATED_UNIT + "Anyway, the summary is X."
                    )
                },
            }

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))

        assert not any(e.get("name") == "repetition_loop_truncated" for e in result)

    @pytest.mark.asyncio
    async def test_repetition_detected_within_single_run_id_after_boundary(self):
        """The run_id boundary reset must not prevent detecting a genuine
        loop that occurs entirely within a single (later) invocation."""

        async def gen(*a, **kw):
            yield {
                "event": "on_chat_model_stream",
                "run_id": "run-1",
                "data": {"chunk": AIMessageChunk(content="short, unrelated reply")},
            }
            for part in [_REPEATED_UNIT] * 6:
                yield {
                    "event": "on_chat_model_stream",
                    "run_id": "run-2",
                    "data": {"chunk": AIMessageChunk(content=part)},
                }

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))

        truncated_events = [
            e for e in result if e.get("name") == "repetition_loop_truncated"
        ]
        assert len(truncated_events) == 1
        assert (
            truncated_events[0]["data"]["chunk"].content.count(_REPEATED_UNIT.strip())
            == 1
        )

    @pytest.mark.asyncio
    async def test_non_looping_ai_chunks_flushed_unchanged(self):
        events = [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="Hello ")},
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="world.")},
            },
        ]

        async def gen(*a, **kw):
            for e in events:
                yield e

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))
        assert result == events

    @pytest.mark.asyncio
    async def test_safety_block_takes_priority_over_repetition(self):
        async def gen(*a, **kw):
            config = a[1] if len(a) > 1 else kw.get("config", {})
            config["_safety_ctx"]["blocked"] = True
            for part in [_REPEATED_UNIT] * 6:
                yield {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": AIMessageChunk(content=part)},
                }

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        result = await _collect(sar.astream_events({}))
        assert any(e.get("name") == "guardian_refusal" for e in result)
        assert not any(e.get("name") == "repetition_loop_truncated" for e in result)

    @pytest.mark.asyncio
    async def test_repetition_check_disabled_via_settings(self):
        parts = [_REPEATED_UNIT] * 6

        async def gen(*a, **kw):
            for part in parts:
                yield {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": AIMessageChunk(content=part)},
                }

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=True)
        with patch(
            "deep_agent.aegra.safety.settings.REPETITION_LOOP_DETECTION_ENABLED",
            False,
        ):
            result = await _collect(sar.astream_events({}))
        assert not any(e.get("name") == "repetition_loop_truncated" for e in result)
        assert len(result) == len(parts)

    @pytest.mark.asyncio
    async def test_non_outermost_does_not_check_repetition(self):
        parts = [_REPEATED_UNIT] * 6

        async def gen(*a, **kw):
            for part in parts:
                yield {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": AIMessageChunk(content=part)},
                }

        inner = MagicMock()
        inner.astream_events = gen
        sar = SafetyAwareRunnable(inner, outermost=False)
        result = await _collect(sar.astream_events({}))
        # All chunks pass through unmodified — no truncation at non-outermost level.
        assert len(result) == len(parts)
