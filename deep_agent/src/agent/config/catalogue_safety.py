"""Startup-time safety scan over catalogue-sourced subagent and skill content.

Subagent descriptions/bodies and skill ``SKILL.md`` content come from the
shared package catalogue, which may be authored by a different org than the
one deploying this agent — a cross-tenant prompt-injection distribution
channel that, unlike user input or tool output, never passes through
Granite Guardian before entering the agent's context. This module runs
Guardian's ``check_safety``/``check_injection`` checks (see
``deep_agent.src.guardrails.client``) over that content and excludes —
rather than blocking the whole agent on — any subagent or skill flagged
unsafe or as a prompt-injection attempt.

``AgentConfig`` reloads subagent/skill config from disk on every access
under ``CONFIG_AUTO_RELOAD`` (the default), which would otherwise let
new/modified content reach graph construction without ever being scanned.
To avoid a Guardian round-trip on every reload, each entry is checked
against a content fingerprint (``_content_fingerprint``) of the last
version scanned safe; unchanged content is skipped.
``ensure_catalogue_safety_scanned`` wires this into the per-request graph
path (see ``deep_agent/aegra/graph.py``) and shares its scan logic with the
one-time startup scan (``scan_catalogue_safety``).

Functions:
    scan_catalogue_safety: One-time full scan at startup.
    ensure_catalogue_safety_scanned: Per-request reload + incremental rescan.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deep_agent.src.guardrails import get_guardrails_config
from deep_agent.utils.pylogger import get_python_logger

from .parser import parse_frontmatter

if TYPE_CHECKING:
    from .loader import AgentConfig

logger = get_python_logger()

# Serializes ensure_catalogue_safety_scanned so concurrent requests don't
# each redundantly Guardian-scan the same new/changed entry. Not used by
# scan_catalogue_safety, which has no concurrent callers.
_rescan_lock = asyncio.Lock()


def _content_fingerprint(content: str) -> str:
    """Stable fingerprint of scanned content, used to detect catalogue changes.

    Used to skip Guardian's ``check_safety``/``check_injection`` calls for
    content that hasn't changed since it was last scanned as safe.

    Args:
        content: The scanned text (e.g. a subagent's description + body).

    Returns:
        A hex SHA-256 digest of *content*.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def _check_content_safety(content: str, context: str) -> tuple[bool, str, str]:
    """Run the harm and injection checks for one piece of catalogue content.

    Args:
        content: The text to scan (e.g. a subagent's description + body).
        context: Label passed through to Guardian logging for traceability.

    Returns:
        A tuple of ``(is_safe, failed_check, verdict)``. ``failed_check`` and
        ``verdict`` are empty strings when ``is_safe`` is True.
    """
    from deep_agent.src.guardrails.client import check_injection, check_safety

    if not content.strip():
        return True, "", ""

    is_safe, verdict = await check_safety(content, context=context)
    if not is_safe:
        return False, "safety", verdict

    is_safe, verdict = await check_injection(content, context=context)
    if not is_safe:
        return False, "injection", verdict

    return True, "", ""


async def _scan_subagents(
    config: "AgentConfig", subagent_configs: dict[str, dict[str, Any]]
) -> list[str]:
    """Scan subagents whose content is new or changed. Returns excluded names.

    Subagents whose fingerprint matches the last successful scan are
    skipped (no Guardian calls); new or changed content is scanned and its
    fingerprint recorded on success.

    Args:
        config: The ``AgentConfig`` singleton to mutate via ``exclude_subagent``.
        subagent_configs: The caller's pinned snapshot to scan — not
            re-fetched here, to avoid a TOCTOU gap from re-reading through
            an auto-reloading getter after this function's ``await`` points.
    """
    excluded: list[str] = []

    for name, cfg in list(subagent_configs.items()):
        text = "\n\n".join(
            part for part in (cfg.get("description", ""), cfg.get("body", "")) if part
        )
        fingerprint = _content_fingerprint(text)
        if config.get_subagent_scan_fingerprint(name) == fingerprint:
            # Unchanged since the last successful safe scan — skip Guardian.
            continue

        is_safe, failed_check, verdict = await _check_content_safety(
            text, context=f"subagent_config:{name}"
        )
        if is_safe:
            config.set_subagent_scan_fingerprint(name, fingerprint)
            continue

        logger.warning(
            "catalogue_content_unsafe",
            kind="subagent",
            name=name,
            check=failed_check,
            verdict=verdict,
        )
        # exclude_subagent() triggers a full synchronous reload under
        # CONFIG_AUTO_RELOAD, not just a dict pop — offload it so one
        # flagged entry doesn't block the event loop.
        await asyncio.to_thread(
            config.exclude_subagent,
            name,
            reason=f"{failed_check} check flagged: {verdict}",
        )
        excluded.append(name)

    return excluded


def _read_skill_frontmatter(skill_md: Path) -> dict[str, Any] | None:
    """Sync helper for ``_scan_skills``: read+parse one ``SKILL.md`` off the event loop.

    Bundles the file check and frontmatter read into one synchronous unit so
    ``_scan_skills`` can offload both with a single ``asyncio.to_thread`` call.

    Args:
        skill_md: Path to the skill's ``SKILL.md`` file.

    Returns:
        The parsed frontmatter dict, or ``None`` if *skill_md* doesn't exist.

    Raises:
        Exception: Whatever ``parse_frontmatter`` raises on malformed content.
    """
    if not skill_md.is_file():
        return None
    return parse_frontmatter(skill_md)


async def _scan_skills(
    config: "AgentConfig", available_skills: dict[str, Path]
) -> list[str]:
    """Scan skills whose SKILL.md content is new or changed. Returns excluded names.

    Skills whose fingerprint matches the last successful scan are skipped
    (no Guardian calls); new or changed content is scanned and its
    fingerprint recorded on success.

    Args:
        config: The ``AgentConfig`` singleton to mutate via ``exclude_skill``.
        available_skills: The caller's pinned skill-name -> directory-path
            snapshot to scan — not re-fetched here, so availability stays
            consistent with what the caller will use to build the graph.
    """
    excluded: list[str] = []

    for name, path in list(available_skills.items()):
        skill_md = path / "SKILL.md"
        try:
            # Both filesystem calls — offload to a worker thread so a
            # large/slow catalogue doesn't block the event loop.
            skill_cfg: dict[str, Any] | None = await asyncio.to_thread(
                _read_skill_frontmatter, skill_md
            )
        except Exception as exc:
            # Fail closed: unlike subagents (dropped on parse failure by
            # _load_all_subagents), the skill directory index still lists a
            # skill even if its SKILL.md fails to parse. Leaving it
            # available here would let malformed frontmatter bypass the scan.
            logger.warning(
                "catalogue_content_unscannable",
                kind="skill",
                name=name,
                error=str(exc),
            )
            await asyncio.to_thread(
                config.exclude_skill, name, reason=f"SKILL.md failed to parse: {exc}"
            )
            excluded.append(name)
            continue

        if skill_cfg is None:
            # Nothing to scan (e.g. malformed/empty skill dir) — leave as-is,
            # the existing skill loading path already tolerates this.
            continue

        text = "\n\n".join(
            part
            for part in (skill_cfg.get("description", ""), skill_cfg.get("body", ""))
            if part
        )
        fingerprint = _content_fingerprint(text)
        if config.get_skill_scan_fingerprint(name) == fingerprint:
            # Unchanged since the last successful safe scan — skip Guardian.
            continue

        is_safe, failed_check, verdict = await _check_content_safety(
            text, context=f"skill_config:{name}"
        )
        if is_safe:
            config.set_skill_scan_fingerprint(name, fingerprint)
            continue

        logger.warning(
            "catalogue_content_unsafe",
            kind="skill",
            name=name,
            check=failed_check,
            verdict=verdict,
        )
        # exclude_skill() calls AgentConfig._ensure_loaded() internally,
        # which — under CONFIG_AUTO_RELOAD — performs a full synchronous
        # reload from disk. Offload it, same as exclude_subagent() above.
        await asyncio.to_thread(
            config.exclude_skill,
            name,
            reason=f"{failed_check} check flagged: {verdict}",
        )
        excluded.append(name)

    return excluded


async def scan_catalogue_safety(config: "AgentConfig") -> dict[str, list[str]]:
    """Scan all loaded subagent and skill metadata for unsafe/injected content.

    Run once at startup, before any fingerprints exist, so every entry gets
    a full scan. See ``ensure_catalogue_safety_scanned`` for the incremental
    rescan used on every subsequent reload. No-op when guardrails are
    disabled.

    Args:
        config: The loaded ``AgentConfig`` singleton to scan and mutate.

    Returns:
        Summary dict with ``subagents_excluded`` and ``skills_excluded`` name lists.
    """
    if get_guardrails_config() is None:
        logger.info("catalogue_safety_scan_skipped", reason="guardrails_disabled")
        return {"subagents_excluded": [], "skills_excluded": []}

    # Offloaded to a worker thread: reads/parses every subagent .md and
    # SKILL.md file from disk, which would otherwise block the startup
    # event loop.
    subagent_configs, available_skills = await asyncio.to_thread(
        config.get_catalogue_snapshot
    )
    subagents_excluded = await _scan_subagents(config, subagent_configs)
    skills_excluded = await _scan_skills(config, available_skills)

    logger.info(
        "catalogue_safety_scan_complete",
        subagents_excluded=len(subagents_excluded),
        skills_excluded=len(skills_excluded),
    )

    return {
        "subagents_excluded": subagents_excluded,
        "skills_excluded": skills_excluded,
    }


async def ensure_catalogue_safety_scanned(
    config: "AgentConfig",
) -> dict[str, Any]:
    """Reload-and-rescan hook for the per-request/per-invocation code path.

    ``AgentConfig._ensure_loaded()`` reloads subagent/skill config from disk
    on every access under ``CONFIG_AUTO_RELOAD`` (the default), but a reload
    alone only re-applies previously recorded exclusions — it doesn't
    rescan content that's new or changed since the last scan. Call this
    before handing subagent/skill data to graph construction (e.g.
    ``graph.py``'s ``agent()`` factory) to close that gap.

    Concretely, this:

    1. Takes exactly one snapshot (``AgentConfig.get_catalogue_snapshot``,
       off-thread) instead of letting ``_scan_subagents``/``_scan_skills``
       each independently call their own getter (which would each trigger
       their own reload under ``CONFIG_AUTO_RELOAD``), so what gets scanned
       is provably the same content returned to the caller (avoids a
       TOCTOU gap).
    2. Runs the incremental scan against that snapshot. Only subagents/
       skills with a new or changed content fingerprint hit Guardian.
    3. Excludes anything newly flagged and scrubs any excluded skill's path
       out of the *pinned* snapshot's ``skill_paths`` too — not just out of
       ``available_skills`` — since each subagent's ``skill_paths`` was
       already baked in at config-load time and ``load_subagents()`` reads
       that list directly (CWE-74: without this, an excluded skill's path
       would still reach the subagent loader through this snapshot).

    A lock serializes steps 1-2 so concurrent requests can't each
    redundantly scan the same new/changed entry. Guardian is never touched
    when guardrails are disabled (the reload still runs; it's cheap disk
    I/O).

    Args:
        config: The ``AgentConfig`` singleton to reload and rescan.

    Returns:
        Dict with ``subagents_excluded``/``skills_excluded`` name lists,
        plus ``subagent_configs``/``available_skills`` — the pinned
        snapshot that was scanned. Callers building a graph (e.g.
        ``graph.py``) should use these two directly instead of calling
        ``AgentConfig`` getters again.
    """
    async with _rescan_lock:
        # Single reload-and-fetch for both sections (see docstring point 1),
        # offloaded to a worker thread so the disk I/O doesn't block the
        # event loop. Serialized by _rescan_lock since AgentConfig's reload
        # isn't safe for concurrent mutation.
        subagent_configs, available_skills = await asyncio.to_thread(
            config.get_catalogue_snapshot
        )

        if get_guardrails_config() is None:
            logger.debug(
                "catalogue_safety_rescan_skipped", reason="guardrails_disabled"
            )
            return {
                "subagents_excluded": [],
                "skills_excluded": [],
                "subagent_configs": subagent_configs,
                "available_skills": available_skills,
            }

        subagents_excluded = await _scan_subagents(config, subagent_configs)
        skills_excluded = await _scan_skills(config, available_skills)

        # exclude_subagent()/exclude_skill() each trigger their own
        # AgentConfig._ensure_loaded() reload, which reassigns AgentConfig's
        # internal dicts rather than mutating the snapshot captured above.
        # Strip newly-excluded names explicitly so the returned snapshot
        # doesn't still contain something AgentConfig itself just excluded.
        for name in subagents_excluded:
            subagent_configs.pop(name, None)
        for name in skills_excluded:
            skill_path = available_skills.pop(name, None)
            # exclude_skill() only scrubs AgentConfig's own live state, not
            # the subagent_configs snapshot pinned above (whose
            # skill_paths were already baked in at config-load time) —
            # scrub it here too (CWE-74; see docstring point 3).
            if skill_path is not None:
                excluded_path = str(skill_path)
                for subagent_cfg in subagent_configs.values():
                    paths = subagent_cfg.get("skill_paths")
                    if paths:
                        subagent_cfg["skill_paths"] = [
                            p for p in paths if str(p) != excluded_path
                        ]

    if subagents_excluded or skills_excluded:
        logger.info(
            "catalogue_safety_rescan_excluded",
            subagents_excluded=subagents_excluded,
            skills_excluded=skills_excluded,
        )

    return {
        "subagents_excluded": subagents_excluded,
        "skills_excluded": skills_excluded,
        "subagent_configs": subagent_configs,
        "available_skills": available_skills,
    }
