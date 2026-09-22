"""Degenerate repetition-loop detection for LLM output (OFFSEC-380).

Gemini (and other autoregressive models) can occasionally get stuck emitting
the same sentence/phrase dozens of times in a single completion — most often
when the model is triggered to refuse a prompt. This wastes tokens/cost and
degrades latency and UX, and can even restart the entire response block
mid-generation.

``detect_repetition_loop`` is a small, pure, dependency-free utility that
looks for a short unit of text repeated many times *consecutively* at the end
of a (possibly still-growing) string and, if found, returns a truncated
version with the repeats collapsed to a single copy. It is intentionally
conservative (exact-match, bounded window) so it is fast enough to run on
every streamed token chunk and does not need to reason about semantics.

Used by :mod:`deep_agent.aegra.safety` (``SafetyAwareRunnable``) to break a
generation loop early when streaming, and to clean up the final message when
not streaming.
"""

from __future__ import annotations

from deep_agent.src.settings import settings

# Hard cap on how long a single repeated "unit" is allowed to be. Bounds the
# cost of each check to O(_MAX_UNIT_LEN ** 2) regardless of how much text has
# accumulated, since only the tail of the text is ever inspected.
_MAX_UNIT_LEN = 400


def max_window_chars(
    min_repeats: int | None = None, max_unit_len: int = _MAX_UNIT_LEN
) -> int:
    """Largest number of trailing characters ``detect_repetition_loop`` ever inspects.

    For any ``unit_len`` up to ``max_unit_len``, the detector only looks at the
    last ``unit_len * min_repeats`` characters of the text it's given — it
    never reasons about anything further back. This helper returns that
    absolute upper bound (``max_unit_len * min_repeats``), letting streaming
    callers know how many trailing characters they must keep buffered (as an
    "undecided suffix") in order to preserve *exactly* the same detection
    behavior while still being able to forward everything older than that
    boundary immediately, incrementally, as it streams in.

    Args:
        min_repeats: Minimum consecutive repeats required to flag a loop.
            Defaults to ``settings.REPETITION_LOOP_MIN_REPEATS``.
        max_unit_len: Largest unit length considered by the detector.

    Returns:
        The number of trailing characters that may still influence a future
        ``detect_repetition_loop`` call.
    """
    if min_repeats is None:
        min_repeats = settings.REPETITION_LOOP_MIN_REPEATS
    return max_unit_len * max(min_repeats, 1)


def detect_repetition_loop(
    text: str,
    min_unit_len: int | None = None,
    min_repeats: int | None = None,
    max_unit_len: int = _MAX_UNIT_LEN,
) -> tuple[bool, str]:
    """Detect a unit of text repeated consecutively at the end of ``text``.

    Args:
        text: The (possibly partial/streaming) completion text to inspect.
        min_unit_len: Minimum character length of the repeated unit to
            consider. Defaults to ``settings.REPETITION_LOOP_MIN_UNIT_LEN``.
        min_repeats: Minimum number of consecutive repeats required to flag
            a loop. Defaults to ``settings.REPETITION_LOOP_MIN_REPEATS``.
        max_unit_len: Largest unit length to consider. Bounds the cost of
            the check; repeated units longer than this are not detected.

    Returns:
        ``(False, text)`` if no loop is detected (text is returned
        unchanged). ``(True, truncated_text)`` if a loop is detected —
        ``truncated_text`` is ``text`` with all but one copy of the repeated
        unit dropped from the end.
    """
    if min_unit_len is None:
        min_unit_len = settings.REPETITION_LOOP_MIN_UNIT_LEN
    if min_repeats is None:
        min_repeats = settings.REPETITION_LOOP_MIN_REPEATS

    if not text or min_unit_len <= 0 or min_repeats <= 1:
        return False, text

    n = len(text)
    largest_unit_len = min(max_unit_len, n // min_repeats)

    for unit_len in range(min_unit_len, largest_unit_len + 1):
        window_len = unit_len * min_repeats
        if window_len > n:
            break

        tail = text[-window_len:]
        unit = tail[:unit_len]
        if not unit.strip():
            # Skip whitespace-only "units" — not a meaningful loop.
            continue

        is_loop = all(
            tail[i * unit_len : (i + 1) * unit_len] == unit
            for i in range(1, min_repeats)
        )
        if not is_loop:
            continue

        # Extend backward to find every consecutive repeat present (there
        # may be more than min_repeats), so we can collapse all of them.
        total_repeats = min_repeats
        while True:
            start = n - (total_repeats + 1) * unit_len
            if start < 0:
                break
            if text[start : start + unit_len] != unit:
                break
            total_repeats += 1

        keep_len = n - (total_repeats - 1) * unit_len
        return True, text[:keep_len]

    return False, text
