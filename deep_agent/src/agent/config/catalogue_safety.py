"""Startup-time safety scan over catalogue-sourced subagent and skill content.

Subagent descriptions/bodies and skill ``SKILL.md`` content originate in the
shared package catalogue and may be authored by a different user or org than
the one deploying this agent (see OFFSEC-379: "The catalogue is a cross
tenant prompt injection distribution channel"). Unlike user input, tool
results, LLM output, and memory writes, this content was never passed
through Granite Guardian before entering the agent's context — it is
trusted purely because it passed structural validation at publish time in
the registry.

This module runs the existing Guardian ``check_safety`` / ``check_injection``
checks (see ``deep_agent.src.guardrails.client``) over that content once, at
startup, and excludes — rather than blocks the whole agent on — any subagent
or skill whose metadata is flagged unsafe or as a prompt-injection attempt.
This preserves availability: a single compromised or malicious catalogue
package cannot take down an agent that otherwise depends on it.

``AgentConfig`` reloads all subagent/skill config from disk on every access
when ``settings.CONFIG_AUTO_RELOAD`` is set (the default) — see
``AgentConfig._ensure_loaded``. That reload previously bypassed this scan
entirely: new/modified catalogue content could reach graph construction
without ever going through Guardian. To close that gap without turning every
reload into 2xN additional Guardian HTTP round-trips, every subagent/skill is
scanned against a content fingerprint (``_content_fingerprint``) of the last
version that was successfully scanned as safe. Unchanged content is skipped;
Guardian is only called for entries that are new or whose content actually
changed. ``ensure_catalogue_safety_scanned`` wires this into the per-request
graph-construction path (see ``deep_agent/aegra/graph.py``); the one-time
``scan_catalogue_safety`` startup call and this incremental rescan share the
same underlying ``_scan_subagents``/``_scan_skills`` — at startup there are
no fingerprints yet, so everything is treated as new (a full scan), exactly
matching the previous startup-only behavior.

Functions:
    scan_catalogue_safety: One-time full scan of all loaded subagents/skills,
        run once at startup (see ``deep_agent/aegra/startup.py``).
    ensure_catalogue_safety_scanned: Per-request/per-invocation hook that
        triggers AgentConfig's normal reload-from-disk and then incrementally
        rescans only new/changed catalogue content before it becomes
        available to graph construction (OFFSEC-379).
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING, Any

from deep_agent.src.guardrails import get_guardrails_config
from deep_agent.utils.pylogger import get_python_logger

from .parser import parse_frontmatter

if TYPE_CHECKING:
    from .loader import AgentConfig

logger = get_python_logger()

# Serializes ensure_catalogue_safety_scanned so that a burst of concurrent
# requests arriving while a subagent/skill's content is new/changed doesn't
# each independently fire the same 2 Guardian calls for that entry before any
# of them has recorded its fingerprint. Not used by scan_catalogue_safety —
# the one-time startup scan has no concurrent callers.
_rescan_lock = asyncio.Lock()


def _content_fingerprint(content: str) -> str:
    """Stable fingerprint of scanned content, used to detect catalogue changes.

    Two calls with the same *content* always produce the same fingerprint;
    any change to the content (a single character) produces a different one.
    Used to skip re-running Guardian's ``check_safety``/``check_injection``
    HTTP calls for subagent/skill content that hasn't changed since it was
    last successfully scanned as safe (OFFSEC-379).

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


async def _scan_subagents(config: "AgentConfig") -> list[str]:
    """Scan subagents whose content is new or changed. Returns excluded names.

    Subagents whose ``description``+``body`` fingerprint matches the last
    successful scan (``AgentConfig.get_subagent_scan_fingerprint``) are
    skipped entirely — no Guardian calls. New subagents (never scanned) and
    subagents whose content changed since the last scan are always scanned;
    on success their fingerprint is recorded so the next call can skip them.
    """
    excluded: list[str] = []

    for name, cfg in list(config.get_all_subagent_configs().items()):
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
        config.exclude_subagent(name, reason=f"{failed_check} check flagged: {verdict}")
        excluded.append(name)

    return excluded


async def _scan_skills(config: "AgentConfig") -> list[str]:
    """Scan skills whose SKILL.md content is new or changed. Returns excluded names.

    Skills whose ``description``+``body`` fingerprint matches the last
    successful scan (``AgentConfig.get_skill_scan_fingerprint``) are skipped
    entirely — no Guardian calls. New skills (never scanned) and skills whose
    ``SKILL.md`` content changed since the last scan are always scanned; on
    success their fingerprint is recorded so the next call can skip them.
    """
    excluded: list[str] = []

    for name, path in list(config.get_available_skills().items()):
        skill_md = path / "SKILL.md"
        if not skill_md.is_file():
            # Nothing to scan (e.g. malformed/empty skill dir) — leave as-is,
            # the existing skill loading path already tolerates this.
            continue

        try:
            skill_cfg: dict[str, Any] = parse_frontmatter(skill_md)
        except Exception as exc:
            # Fail closed: a SKILL.md that can't be parsed can't be scanned,
            # but the directory-based skill index (_scan_available_skills)
            # loads it regardless of parse success — unlike subagents, which
            # are dropped entirely on a parse failure (_load_all_subagents).
            # Leaving it available here would let a deliberately malformed
            # frontmatter (with a malicious body) bypass this scan entirely.
            logger.warning(
                "catalogue_content_unscannable",
                kind="skill",
                name=name,
                error=str(exc),
            )
            config.exclude_skill(name, reason=f"SKILL.md failed to parse: {exc}")
            excluded.append(name)
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
        config.exclude_skill(name, reason=f"{failed_check} check flagged: {verdict}")
        excluded.append(name)

    return excluded


async def scan_catalogue_safety(config: "AgentConfig") -> dict[str, list[str]]:
    """Scan all loaded subagent and skill metadata for unsafe/injected content.

    Run once at process startup (``run_startup()`` -> ``_validate_catalogue_safety()``),
    before any fingerprints have been recorded — every subagent/skill is
    therefore "new" and gets a full scan, same as before fingerprinting was
    introduced. See ``ensure_catalogue_safety_scanned`` for the incremental
    rescan that runs on every subsequent reload.

    No-op when guardrails are disabled or not yet initialised — mirrors the
    short-circuit in ``deep_agent.src.guardrails.client._call_guardian``, so
    this scan never fails startup or blocks when Guardian is not configured.

    Args:
        config: The loaded ``AgentConfig`` singleton to scan and mutate.

    Returns:
        Summary dict with ``subagents_excluded`` and ``skills_excluded`` name lists.
    """
    if get_guardrails_config() is None:
        logger.info("catalogue_safety_scan_skipped", reason="guardrails_disabled")
        return {"subagents_excluded": [], "skills_excluded": []}

    subagents_excluded = await _scan_subagents(config)
    skills_excluded = await _scan_skills(config)

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
) -> dict[str, list[str]]:
    """Reload-and-rescan hook for the per-request/per-invocation code path.

    ``AgentConfig._ensure_loaded()`` reloads all subagent/skill config from
    disk on every access when ``settings.CONFIG_AUTO_RELOAD`` is set (the
    default) — but a reload alone only re-applies previously recorded
    exclusions (``_reapply_exclusions``); it does not re-run the Guardian
    scan over content that changed or was added since the last scan. Call
    this early in any path that is about to hand ``AgentConfig`` subagent/
    skill data to graph construction (e.g. ``deep_agent/aegra/graph.py``'s
    ``agent()`` factory, which Aegra invokes per-request) so that gap is
    closed *before* that data is read for that purpose (OFFSEC-379).

    Concretely, this:

    1. Triggers ``AgentConfig``'s normal synchronous reload-from-disk (by
       calling an existing getter) and lets ``_reapply_exclusions()`` remove
       anything already flagged by a prior scan.
    2. Runs the incremental scan (``_scan_subagents`` / ``_scan_skills``),
       which only calls Guardian for subagents/skills whose content
       fingerprint is new or changed since the last successful scan of that
       name — unchanged entries never touch Guardian, so a request arriving
       while nothing in the catalogue has changed costs zero extra HTTP
       calls, not 2xN.
    3. Excludes (``exclude_subagent`` / ``exclude_skill``, same sticky
       mechanism as the startup scan) anything newly flagged, before
       returning — so the caller's subsequent reads of subagent/skill
       config already reflect the exclusion.

    A process-wide lock serializes step 2 so a burst of concurrent requests
    that all observe the same brand-new/changed entry don't each
    independently fire Guardian for it before the first one finishes and
    records its fingerprint.

    Guardian is never touched when guardrails are disabled, mirroring
    ``scan_catalogue_safety``'s short-circuit (the reload trigger itself
    still runs — it's cheap disk I/O with no Guardian calls either way).

    Args:
        config: The ``AgentConfig`` singleton to reload and rescan.

    Returns:
        Summary dict with ``subagents_excluded`` and ``skills_excluded`` name
        lists (empty lists when guardrails are disabled or nothing new/changed
        was found).
    """
    # Trigger AgentConfig's normal sync reload path. This is a no-op read
    # when CONFIG_AUTO_RELOAD is false and configs are already loaded, and a
    # full reload-from-disk (including _reapply_exclusions()) otherwise.
    # _scan_subagents()/_scan_skills() below also read through this same
    # reloaded state.
    config.get_all_subagent_configs()

    if get_guardrails_config() is None:
        logger.debug("catalogue_safety_rescan_skipped", reason="guardrails_disabled")
        return {"subagents_excluded": [], "skills_excluded": []}

    async with _rescan_lock:
        subagents_excluded = await _scan_subagents(config)
        skills_excluded = await _scan_skills(config)

    if subagents_excluded or skills_excluded:
        logger.info(
            "catalogue_safety_rescan_excluded",
            subagents_excluded=subagents_excluded,
            skills_excluded=skills_excluded,
        )

    return {
        "subagents_excluded": subagents_excluded,
        "skills_excluded": skills_excluded,
    }
