"""Mode-gated evolution engine orchestration.

The concrete packet, decision, admission, authoring, validation, candidate, and
commit collaborators are intentionally injected. This module owns the hard
mode boundary so no caller can accidentally bypass audit_only/fix_only/autonomous
semantics.
"""

from __future__ import annotations

import inspect
import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from openspace.skill_engine.patch import SKILL_FILENAME, collect_skill_snapshot
from openspace.skill_engine.registry import write_skill_id
from openspace.skill_engine.skill_utils import validate_skill_dir
from openspace.skill_engine.types import (
    EvolutionType,
    SkillCategory,
    SkillLineage,
    SkillOrigin,
    SkillRecord,
)
from .audit import EvolutionActionRecord
from .behavior_eval import (
    SkillBehaviorEvalResult,
    behavior_eval_feedback,
    _replay_result_has_verified_executable_evidence,
    _replay_task_result_failures,
)
from openspace.utils.logging import Logger

logger = Logger.get_logger(__name__)

_EVOLUTION_MODES = {"audit_only", "fix_only", "autonomous"}


@dataclass(frozen=True, slots=True)
class EvolutionRunResult:
    job_id: str
    status: str
    decisions: list[Any] = field(default_factory=list)
    admissions: list[Any] = field(default_factory=list)
    candidates: list[Any] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    behavior_evals: list[SkillBehaviorEvalResult] = field(default_factory=list)
    evolved_skill_records: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvolutionMutationOutcome:
    action: Any | None = None
    candidate: Any | None = None
    behavior_evals: list[SkillBehaviorEvalResult] = field(default_factory=list)
    blocked_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def committed_action(self) -> Any | None:
        if self.action is None:
            return None
        return self.action


class EvolutionEngine:
    """Process TriggerJobs through mode-gated evolution stages."""

    def __init__(
        self,
        *,
        packet_builder: Any | None = None,
        decision_engine: Any | None = None,
        admission_policy: Any | None = None,
        candidate_store: Any | None = None,
        authoring_backend: Any | None = None,
        validator: Any | None = None,
        behavior_evaluator: Any | None = None,
        committer: Any | None = None,
        evolution_mode: str = "autonomous",
        behavior_eval_max_revisions: int = 2,
    ) -> None:
        self.packet_builder = packet_builder
        self.decision_engine = decision_engine
        self.admission_policy = admission_policy
        self.candidate_store = candidate_store
        self.authoring_backend = authoring_backend
        self.validator = validator
        self.behavior_evaluator = behavior_evaluator
        self.committer = committer
        self.evolution_mode = _normalize_evolution_mode(evolution_mode)
        self.behavior_eval_max_revisions = max(0, int(behavior_eval_max_revisions))

    async def process_job(
        self,
        job: Any,
        *,
        evolution_mode: str | None = None,
    ) -> EvolutionRunResult:
        mode = _normalize_evolution_mode(evolution_mode or self.evolution_mode)
        job_id = str(getattr(job, "job_id", None) or getattr(job, "id", None) or "")
        if not job_id:
            job_id = "unknown"
        job_status = str(getattr(job, "status", "") or "").strip().lower()
        if job_status and job_status != "running":
            return EvolutionRunResult(
                job_id=job_id,
                status="failed_unclaimed_job",
                errors=[
                    "trigger job must be claimed/running before processing: "
                    f"{job_status}"
                ],
            )

        decisions: list[Any] = []
        admissions: list[Any] = []
        candidates: list[Any] = []
        actions: list[Any] = []
        behavior_evals: list[SkillBehaviorEvalResult] = []
        evolved: list[Any] = []
        errors: list[str] = []
        packet: Any | None = None

        try:
            packet_result = await self._build_packet_result(job)
            packet = _unwrap_packet_result(packet_result)
            if packet is None:
                decision = self._persist_packet_noop_decision(job, packet_result)
                if decision is not None:
                    decisions.append(decision)
                result = EvolutionRunResult(
                    job_id=job_id,
                    status="completed_noop",
                    decisions=decisions,
                )
                await self._record_candidate_recheck_result(job, None, mode, result)
                return result

            decisions = list(await self._decide(packet, job))
            if not decisions:
                result = EvolutionRunResult(
                    job_id=job_id,
                    status="completed_noop",
                    decisions=[],
                )
                await self._record_candidate_recheck_result(job, packet, mode, result)
                return result

            for decision in decisions:
                admission = await self._admit(decision, packet, job)
                admissions.append(admission)

                if mode == "audit_only":
                    continue

                outcome = _admission_outcome(admission)
                if outcome in {
                    "noop",
                    "reject",
                    "rejected",
                    "human_review",
                    "needs_human_review",
                }:
                    continue

                action_type = _decision_action_type(decision)
                if mode == "fix_only" and action_type != EvolutionType.FIX.value:
                    candidate = await self._create_candidate(
                        decision,
                        admission,
                        packet,
                        job,
                        reason="fix_only_mode_non_fix",
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                    continue

                if outcome == "candidate":
                    candidate = await self._create_candidate(
                        decision,
                        admission,
                        packet,
                        job,
                        reason="admission_candidate",
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                    continue

                if mode == "fix_only" and outcome != "direct":
                    continue

                mutation = await self._author_validate_commit(
                    decision,
                    admission,
                    packet,
                    job,
                )
                if mutation.candidate is not None:
                    candidates.append(mutation.candidate)
                behavior_evals.extend(mutation.behavior_evals)
                if mutation.errors:
                    errors.extend(mutation.errors)
                action = mutation.committed_action
                if action is None:
                    if mutation.blocked_reason:
                        logger.info(
                            "Evolution direct action blocked before commit: %s",
                            mutation.blocked_reason,
                        )
                    else:
                        errors.append(
                            "direct action did not produce a committed action record"
                        )
                    continue
                actions.append(action)
                commit_status = _commit_status(action)
                commit_succeeded = commit_status in {
                    "committed",
                    "committed_reconciled",
                }
                if commit_status and not commit_succeeded:
                    reason = (
                        getattr(action, "failure_reason", None)
                        or _mapping_get(action, "failure_reason")
                        or commit_status
                    )
                    errors.append(f"commit {commit_status}: {reason}")
                elif commit_succeeded:
                    candidates.extend(
                        await self._mark_promoted_candidates(job, packet, action)
                    )
                skill_record = _extract_skill_record(action)
                if skill_record is not None:
                    evolved.append(skill_record)

            result = EvolutionRunResult(
                job_id=job_id,
                status="failed" if errors else "completed",
                decisions=decisions,
                admissions=admissions,
                candidates=candidates,
                actions=actions,
                behavior_evals=behavior_evals,
                evolved_skill_records=evolved,
                errors=errors,
            )
            await self._record_candidate_recheck_result(job, packet, mode, result)
            return result
        except Exception as exc:
            logger.debug("EvolutionEngine job failed: %s", job_id, exc_info=True)
            errors.append(str(exc))
            result = EvolutionRunResult(
                job_id=job_id,
                status="failed",
                decisions=decisions,
                admissions=admissions,
                candidates=candidates,
                actions=actions,
                behavior_evals=behavior_evals,
                evolved_skill_records=evolved,
                errors=errors,
            )
            await self._record_candidate_recheck_result(job, packet, mode, result)
            return result

    async def _build_packet_result(self, job: Any) -> Any:
        builder = self.packet_builder
        if builder is None:
            return getattr(job, "packet", None)
        for name in ("build_trigger_packet", "build_packet", "build"):
            method = getattr(builder, name, None)
            if callable(method):
                return await _maybe_await(method(job))
        if callable(builder):
            return await _maybe_await(builder(job))
        return None

    def _persist_packet_noop_decision(self, job: Any, packet_result: Any) -> Any | None:
        store = _evidence_store_for(self)
        if store is None:
            return None
        try:
            from openspace.skill_engine.decision.types import DecisionRationale

            status = (
                getattr(packet_result, "status", None)
                or _mapping_get(packet_result, "status")
                or "packet_unavailable"
            )
            noop_reason = (
                getattr(packet_result, "noop_reason", None)
                or _mapping_get(packet_result, "noop_reason")
                or str(status)
            )
            missing = _str_list(
                getattr(packet_result, "missing_ref_types", None)
                or _mapping_get(packet_result, "missing_ref_types")
            )
            decision = DecisionRationale(
                decision_id=f"dec_{uuid.uuid4().hex}",
                trigger_job_id=str(getattr(job, "job_id", "") or ""),
                proposed_action="NOOP",
                candidate_policy="never",
                target_skill_ids=[],
                reason_summary=f"Evidence packet unavailable: {noop_reason}",
                reason_tags=[
                    "packet_unavailable",
                    f"packet_status:{status}",
                    *[f"missing_ref:{item}" for item in missing],
                ],
                evidence_claims=[],
                confidence=0.0,
                risks=missing,
                source_analysis_id=None,
                noop_reason=str(noop_reason),
                analyzed_by="evolution_engine",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            persist = getattr(store, "persist_decision", None)
            if callable(persist):
                persist(decision, packet_id="")
            return decision
        except Exception:
            logger.debug("Failed to persist packet NOOP decision", exc_info=True)
            return None

    async def _decide(self, packet: Any, job: Any) -> list[Any]:
        engine = self.decision_engine
        if engine is None:
            return list(getattr(packet, "decisions", []) or [])
        for name in ("decide", "process_packet"):
            method = getattr(engine, name, None)
            if callable(method):
                result = await _maybe_await(method(packet))
                return _as_list(result)
        if callable(engine):
            return _as_list(await _maybe_await(engine(packet, job)))
        return []

    async def _admit(self, decision: Any, packet: Any, job: Any) -> Any:
        policy = self.admission_policy
        if policy is None:
            return getattr(decision, "admission", None) or {
                "outcome": getattr(decision, "admission_outcome", "noop")
            }
        for name in ("admit", "evaluate"):
            method = getattr(policy, name, None)
            if callable(method):
                if _accepts_positional_count(method, 3):
                    return await _maybe_await(method(decision, packet, job))
                return await _maybe_await(method(decision, packet))
        if callable(policy):
            return await _maybe_await(policy(decision, packet, job))
        return {"outcome": "noop"}

    async def _create_candidate(
        self,
        decision: Any,
        admission: Any,
        packet: Any,
        job: Any,
        *,
        reason: str,
    ) -> Any:
        store = self.candidate_store
        if store is None:
            return {
                "decision": decision,
                "admission": admission,
                "job": job,
                "reason": reason,
            }
        for name in ("create_or_merge", "create"):
            method = getattr(store, name, None)
            if callable(method):
                kwargs = {
                    "decision": decision,
                    "admission": admission,
                    "job": job,
                    "reason": reason,
                }
                if _accepts_keyword(method, "packet"):
                    kwargs["packet"] = packet
                return await _maybe_await(method(**kwargs))
        if callable(store):
            if _accepts_positional_count(store, 5):
                return await _maybe_await(
                    store(decision, admission, packet, job, reason)
                )
            return await _maybe_await(store(decision, admission, job, reason))
        return None

    async def _mark_promoted_candidates(
        self,
        job: Any,
        packet: Any,
        action: Any,
    ) -> list[Any]:
        store = self.candidate_store
        if store is None:
            return []
        action_id = _action_id(action)
        if not action_id:
            return []
        candidate_ids = _explicit_promoted_candidate_ids(job, action)
        if not candidate_ids:
            return []
        marker = getattr(store, "mark_promoted", None)
        if not callable(marker):
            return []

        promoted: list[Any] = []
        for candidate_id in candidate_ids:
            try:
                promoted_candidate = await _maybe_await(
                    marker(candidate_id, action_id, commit_succeeded=True)
                )
            except Exception:
                logger.debug(
                    "Failed to mark evolution candidate promoted: %s",
                    candidate_id,
                    exc_info=True,
                )
                continue
            if promoted_candidate is not None:
                promoted.append(promoted_candidate)
        return promoted

    async def _record_candidate_recheck_result(
        self,
        job: Any,
        packet: Any | None,
        mode: str,
        result: EvolutionRunResult,
    ) -> None:
        if str(getattr(job, "trigger_type", "") or "").upper() != "CANDIDATE_RECHECK":
            return
        store = self.candidate_store
        recorder = getattr(store, "record_recheck_result", None)
        if store is None or not callable(recorder):
            return
        candidate_ids = _explicit_candidate_ids_from_job(job)
        if not candidate_ids:
            return
        payload = _candidate_recheck_result_payload(result, mode=mode)
        blocked_reason = _candidate_recheck_blocked_reason(result, mode=mode)
        needed_evidence = _candidate_recheck_needed_evidence(result)
        for candidate_id in candidate_ids:
            try:
                await _maybe_await(
                    recorder(
                        candidate_id,
                        result=payload,
                        blocked_reason=blocked_reason,
                        needed_evidence=needed_evidence,
                    )
                )
            except Exception:
                logger.debug(
                    "Failed to record candidate recheck result: %s",
                    candidate_id,
                    exc_info=True,
                )

    async def _author_validate_commit(
        self,
        decision: Any,
        admission: Any,
        packet: Any,
        job: Any,
    ) -> EvolutionMutationOutcome:
        authoring = self.authoring_backend
        if authoring is None:
            logger.warning("Evolution authoring skipped: no authoring backend available")
            return EvolutionMutationOutcome(errors=["missing_authoring_backend"])
        method = getattr(authoring, "author_from_action_packet", None)
        if not callable(method):
            logger.warning(
                "Evolution authoring backend does not expose "
                "author_from_action_packet; skipping mutation"
            )
            return EvolutionMutationOutcome(errors=["invalid_authoring_backend"])

        action_packet = await self._build_action_packet(decision, admission, packet)
        if action_packet is None:
            return EvolutionMutationOutcome(errors=["missing_action_packet"])

        behavior_evals: list[SkillBehaviorEvalResult] = []
        eval_feedback: Any | None = None
        previous_authoring: Any | None = None
        last_validation: Any | None = None
        last_authoring: Any | None = None
        for attempt in range(self.behavior_eval_max_revisions + 1):
            authoring_result = await self._call_authoring_backend(
                method,
                action_packet,
                eval_feedback=eval_feedback,
                previous_authoring=previous_authoring,
            )
            last_authoring = authoring_result
            if authoring_result is None:
                return EvolutionMutationOutcome(
                    behavior_evals=behavior_evals,
                    errors=["authoring_returned_none"],
                )
            if not _authoring_staged(authoring_result):
                errors = ["authoring_not_staged"]
                failure_reason = _authoring_failure_reason(authoring_result)
                if failure_reason:
                    errors.append(f"authoring_failure:{failure_reason}")
                return EvolutionMutationOutcome(
                    behavior_evals=behavior_evals,
                    errors=errors,
                )

            validator = self.validator
            validation_result = None
            if validator is None:
                logger.warning("Evolution validation skipped: no validator available")
                return EvolutionMutationOutcome(
                    behavior_evals=behavior_evals,
                    errors=["missing_validator"],
                )
            validator_packet = await self._build_validator_packet(
                authoring_result,
                action_packet,
            )
            if validator_packet is None:
                logger.warning("Evolution validation skipped: no validator packet available")
                return EvolutionMutationOutcome(
                    behavior_evals=behavior_evals,
                    errors=["missing_validator_packet"],
                )
            validation_result = await self._call_validator(
                validator,
                authoring_result,
                validator_packet,
                decision,
                admission,
                job,
            )
            last_validation = validation_result
            if not _validation_passed(validation_result):
                return EvolutionMutationOutcome(
                    behavior_evals=behavior_evals,
                    blocked_reason="validation_failed",
                )

            behavior_result = await self._run_behavior_eval(
                authoring_result,
                validation_result,
                decision,
                admission,
                action_packet,
            )
            if behavior_result is not None:
                behavior_evals.append(behavior_result)
                if not behavior_result.passed:
                    if attempt < self.behavior_eval_max_revisions:
                        eval_feedback = behavior_eval_feedback(behavior_result)
                        previous_authoring = authoring_result
                        continue
                    candidate = await self._create_candidate(
                        decision,
                        admission,
                        packet,
                        job,
                        reason=_behavior_eval_candidate_reason(behavior_result),
                    )
                    return EvolutionMutationOutcome(
                        candidate=candidate,
                        behavior_evals=behavior_evals,
                        blocked_reason=_behavior_eval_blocked_reason(behavior_result),
                    )
                validation_result = _attach_behavior_eval_ref(
                    validation_result,
                    behavior_result,
                )
            else:
                candidate = await self._create_candidate(
                    decision,
                    admission,
                    packet,
                    job,
                    reason="missing_behavior_eval",
                )
                return EvolutionMutationOutcome(
                    candidate=candidate,
                    behavior_evals=behavior_evals,
                    blocked_reason="missing_behavior_eval",
                    errors=["missing_behavior_eval"],
                )

            committer = self.committer
            if committer is None:
                logger.warning("Evolution commit skipped: no committer available")
                return EvolutionMutationOutcome(
                    behavior_evals=behavior_evals,
                    errors=["missing_committer"],
                )
            for name in ("commit", "apply"):
                commit_method = getattr(committer, name, None)
                if callable(commit_method):
                    action = await _maybe_await(
                        commit_method(
                            authoring_result,
                            validation_result,
                            decision,
                            admission,
                            action_packet,
                        )
                    )
                    return EvolutionMutationOutcome(
                        action=action,
                        behavior_evals=behavior_evals,
                    )
            if callable(committer):
                action = await _maybe_await(
                    committer(
                        authoring_result,
                        validation_result,
                        decision,
                        admission,
                        action_packet,
                    )
                )
                return EvolutionMutationOutcome(
                    action=action,
                    behavior_evals=behavior_evals,
                )
            return EvolutionMutationOutcome(
                action=authoring_result,
                behavior_evals=behavior_evals,
            )
        return EvolutionMutationOutcome(
            behavior_evals=behavior_evals,
            blocked_reason="behavior_eval_revision_exhausted",
            errors=[] if last_authoring is not None and last_validation is not None else [
                "mutation_loop_exhausted_without_result"
            ],
        )

    async def _call_authoring_backend(
        self,
        method: Any,
        action_packet: Any,
        *,
        eval_feedback: Any | None,
        previous_authoring: Any | None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if eval_feedback is not None and _accepts_keyword(method, "eval_feedback"):
            kwargs["eval_feedback"] = eval_feedback
        if previous_authoring is not None and _accepts_keyword(method, "previous_authoring"):
            kwargs["previous_authoring"] = previous_authoring
        if kwargs:
            return await _maybe_await(method(action_packet, **kwargs))
        return await _maybe_await(method(action_packet))

    async def _run_behavior_eval(
        self,
        authoring_result: Any,
        validation_result: Any,
        decision: Any,
        admission: Any,
        action_packet: Any,
    ) -> SkillBehaviorEvalResult | None:
        evaluator = self.behavior_evaluator
        if evaluator is None:
            logger.warning("Behavior eval skipped: no behavior evaluator available")
            return None
        for name in ("evaluate", "run"):
            method = getattr(evaluator, name, None)
            if callable(method):
                return await _maybe_await(
                    method(
                        authoring_result,
                        validation_result,
                        decision,
                        admission,
                        action_packet,
                    )
                )
        if callable(evaluator):
            return await _maybe_await(
                evaluator(
                    authoring_result,
                    validation_result,
                    decision,
                    admission,
                    action_packet,
                )
            )
        return None

    async def _build_action_packet(
        self,
        decision: Any,
        admission: Any,
        packet: Any,
    ) -> Any | None:
        builder = self.packet_builder
        if builder is not None:
            method = getattr(builder, "build_action_packet", None)
            if callable(method):
                action_scope = _decision_with_action_scope(decision, admission, packet)
                result = await _maybe_await(method(action_scope))
                return _unwrap_packet_result(result)

        if str(getattr(packet, "packet_type", "") or "").lower() == "action":
            return packet
        logger.warning("Evolution authoring skipped: no action packet builder available")
        return None

    async def _build_validator_packet(
        self,
        authoring_result: Any,
        action_packet: Any,
    ) -> Any | None:
        builder = self.packet_builder
        if builder is not None:
            method = getattr(builder, "build_validator_packet", None)
            if callable(method):
                result = await _maybe_await(method(authoring_result))
                return _unwrap_packet_result(result)

        if str(getattr(action_packet, "packet_type", "") or "").lower() == "validator":
            return action_packet
        if builder is None:
            return action_packet
        return None

    async def _call_validator(
        self,
        validator: Any,
        authoring_result: Any,
        validator_packet: Any,
        decision: Any,
        admission: Any,
        job: Any,
    ) -> Any:
        for name in ("validate", "run"):
            method = getattr(validator, name, None)
            if callable(method):
                return await _invoke_validator(
                    method,
                    authoring_result,
                    validator_packet,
                    decision,
                    admission,
                    job,
                )
        if callable(validator):
            return await _invoke_validator(
                validator,
                authoring_result,
                validator_packet,
                decision,
                admission,
                job,
            )
        return None


class EvolutionCommitter:
    """Commit approved staged edits to active skill storage with audit lineage."""

    def __init__(
        self,
        *,
        evidence_store: Any,
        skill_store: Any,
        registry: Any,
        trigger_store: Any | None = None,
        trigger_engine: Any | None = None,
        backup_root: str | Path | None = None,
    ) -> None:
        self.evidence_store = evidence_store
        self.skill_store = skill_store
        self.registry = registry
        self.trigger_store = trigger_store or getattr(trigger_engine, "store", None)
        self.trigger_engine = trigger_engine
        if backup_root is not None:
            self.backup_root = Path(backup_root).expanduser().resolve()
        else:
            db_path = getattr(evidence_store, "db_path", None)
            if db_path is not None:
                self.backup_root = Path(db_path).expanduser().resolve().parent / "evolution" / "backups"
            else:
                self.backup_root = Path.cwd() / ".openspace" / "evolution" / "backups"

    async def commit(
        self,
        authoring: Any,
        validation: Any,
        decision: Any,
        admission: Any,
        action_packet: Any,
    ) -> EvolutionActionRecord:
        action_type = _commit_action_type(decision, _attr(authoring, "staged_edit"))
        self._check_preconditions(
            authoring=authoring,
            validation=validation,
            decision=decision,
            admission=admission,
            action_packet=action_packet,
            action_type=action_type,
        )
        staged = _attr(authoring, "staged_edit")
        if staged is None:
            raise ValueError("commit requires staged_edit")

        target_dir = Path(str(_attr(staged, "target_dir") or "")).expanduser().resolve()
        proposed_dir = _proposed_dir(staged, target_dir)
        parent_skill_ids = _parent_skill_ids(staged, decision, action_type)
        changed_files = _str_list(_attr(validation, "changed_files")) or _str_list(
            _attr(staged, "changed_files")
        )
        evidence_refs = _str_list(_attr(validation, "provenance_refs")) or _str_list(
            _attr(staged, "evidence_refs")
        )
        proposed_skill_id = self._resolve_commit_skill_id(
            staged=staged,
            action_type=action_type,
            target_dir=target_dir,
            parent_skill_ids=parent_skill_ids,
        )
        action_id = f"act_{uuid.uuid4().hex}"
        backup_dir: Path | None = None

        if action_type == "FIX":
            backup_dir = self.backup_root / action_id / "before"
        action = self.evidence_store.begin_action(
            action_id=action_id,
            decision_id=str(_attr(decision, "decision_id") or _attr(authoring, "decision_id") or ""),
            trigger_job_id=str(
                _attr(decision, "trigger_job_id")
                or _attr(action_packet, "trigger_job_id")
                or ""
            ),
            authoring_id=str(_attr(authoring, "authoring_id") or ""),
            validation_id=str(_attr(validation, "validation_id") or ""),
            action_type=action_type,
            skill_id=proposed_skill_id,
            parent_skill_ids=parent_skill_ids,
            changed_files=changed_files,
            evidence_refs=evidence_refs,
            staging_dir=str(_attr(staged, "staging_dir") or ""),
            active_target_dir=str(target_dir),
            backup_dir=str(backup_dir) if backup_dir is not None else None,
            session_id=_packet_scope_value(action_packet, "session_id"),
            task_id=_packet_scope_value(action_packet, "task_id"),
            raw_backrefs=_commit_raw_backrefs(
                authoring,
                validation,
                decision,
                admission,
                action_packet,
                evidence_refs,
            ),
        )
        if backup_dir is not None:
            backup_dir = Path(action.backup_dir or backup_dir)

        phase = "begin"
        store_written = False
        new_record: SkillRecord | None = None
        try:
            phase = "backup"
            if action_type == "FIX":
                if backup_dir is None:
                    raise RuntimeError("FIX commit missing backup_dir")
                _backup_target(target_dir, backup_dir)

            phase = "active_copy"
            _apply_proposed_dir(action_type, proposed_dir, target_dir)

            phase = "disk_validation"
            validation_error = validate_skill_dir(target_dir)
            if validation_error:
                raise RuntimeError(f"disk structural validation failed: {validation_error}")

            phase = "skill_store"
            new_record = self._build_skill_record(
                action=action,
                staged=staged,
                authoring=authoring,
                decision=decision,
                action_packet=action_packet,
                action_type=action_type,
                target_dir=target_dir,
                parent_skill_ids=parent_skill_ids,
                evidence_refs=evidence_refs,
                skill_id=proposed_skill_id,
            )
            if action_type in {"FIX", "DERIVED"}:
                await _maybe_await(
                    self.skill_store.evolve_skill(new_record, parent_skill_ids)
                )
            else:
                await _maybe_await(self.skill_store.save_record(new_record))
            store_written = True

            phase = "skill_id_sidecar"
            write_skill_id(target_dir, new_record.skill_id, raise_on_error=True)

            phase = "local_category_tree"
            materialized_dir = self._materialize_local_category_tree(
                action_type=action_type,
                target_dir=target_dir,
                record=new_record,
                decision=decision,
                parent_skill_ids=parent_skill_ids,
            )
            if materialized_dir != target_dir:
                target_dir = materialized_dir
                new_record.path = str(target_dir / SKILL_FILENAME)
                await _maybe_await(self.skill_store.save_record(new_record))

            phase = "registry_refresh"
            self._refresh_registry(action_type, target_dir, new_record, parent_skill_ids)

            phase = "evidence_finalize"
            finalized = self.evidence_store.finalize_action(
                action.action_id,
                status="committed",
                skill_id=new_record.skill_id,
                changed_files=changed_files,
                backup_dir=str(backup_dir) if backup_dir is not None else None,
                session_id=_packet_scope_value(action_packet, "session_id"),
                task_id=_packet_scope_value(action_packet, "task_id"),
                raw_backrefs=_commit_raw_backrefs(
                    authoring,
                    validation,
                    decision,
                    admission,
                    action_packet,
                    evidence_refs,
                ),
            )
            return _attach_skill_record(finalized, new_record)
        except Exception as exc:
            reason = f"{phase}: {exc}"
            self._record_failure(action.action_id, phase, "failed", reason)
            if store_written:
                return _attach_skill_record(
                    _replace_action(
                        action,
                        skill_id=new_record.skill_id if new_record else proposed_skill_id,
                        failure_reason=reason,
                    ),
                    new_record,
                )
            failed_status = "failed"
            try:
                _rollback_disk(action_type, target_dir, backup_dir)
            except Exception as rollback_exc:
                failed_status = "failed_needs_review"
                reason = f"{reason}; rollback failed: {rollback_exc}"
                self._record_failure(
                    action.action_id,
                    "rollback",
                    "failed_needs_review",
                    str(rollback_exc),
                )
            finalized = self.evidence_store.finalize_action(
                action.action_id,
                status=failed_status,
                skill_id=proposed_skill_id,
                changed_files=changed_files,
                backup_dir=str(backup_dir) if backup_dir is not None else None,
                failure_reason=reason,
                session_id=_packet_scope_value(action_packet, "session_id"),
                task_id=_packet_scope_value(action_packet, "task_id"),
                raw_backrefs=_commit_raw_backrefs(
                    authoring,
                    validation,
                    decision,
                    admission,
                    action_packet,
                    evidence_refs,
                ),
            )
            return finalized

    async def recover_committing_actions(
        self,
        *,
        limit: int = 100,
    ) -> list[EvolutionActionRecord]:
        from .recovery import EvolutionRecovery

        recovered: list[EvolutionActionRecord] = []
        actions = self.evidence_store.list_actions(status="committing", limit=limit)
        recovery = EvolutionRecovery(
            evidence_store=self.evidence_store,
            skill_store=self.skill_store,
            registry=self.registry,
        )
        for action in actions:
            try:
                recovery._reconcile_action(action)  # pylint: disable=protected-access
                finalized = self.evidence_store.load_action(action.action_id) or action
                record = (
                    self.skill_store.load_record(finalized.skill_id)
                    if finalized.skill_id
                    else None
                )
                recovered.append(
                    _attach_skill_record(finalized, record)
                    if record is not None
                    else finalized
                )
            except Exception as exc:
                reason = f"recovery: {exc}"
                self._record_failure(
                    action.action_id,
                    "recovery",
                    "failed_retryable",
                    reason,
                )
                recovered.append(_replace_action(action, failure_reason=reason))
        return recovered

    def _check_preconditions(
        self,
        *,
        authoring: Any,
        validation: Any,
        decision: Any | None = None,
        admission: Any,
        action_packet: Any | None = None,
        action_type: str,
    ) -> None:
        if str(_attr(authoring, "status") or "").strip().lower() != "staged":
            raise ValueError("authoring result is not staged")
        if str(_attr(validation, "outcome") or "").strip().lower() != "approve":
            raise ValueError("validation result is not approved")
        if str(_attr(admission, "outcome") or "").strip().lower() != "direct":
            raise ValueError("admission result is not direct")
        if action_type not in {"FIX", "DERIVED", "CAPTURED"}:
            raise ValueError(f"unsupported commit action type: {action_type or '(missing)'}")
        _require_approved_behavior_eval(
            self.evidence_store,
            authoring=authoring,
            validation=validation,
            decision=decision,
            action_packet=action_packet,
            action_type=action_type,
        )

    def _build_skill_record(
        self,
        *,
        action: EvolutionActionRecord,
        staged: Any,
        authoring: Any,
        decision: Any,
        action_packet: Any,
        action_type: str,
        target_dir: Path,
        parent_skill_ids: list[str],
        evidence_refs: list[str],
        skill_id: str,
    ) -> SkillRecord:
        parent_records = _load_parent_records(self.skill_store, parent_skill_ids)
        if action_type == "FIX" and len(parent_records) != 1:
            raise RuntimeError("FIX commit requires exactly one SkillStore parent")
        if action_type == "DERIVED" and not parent_records:
            raise RuntimeError("DERIVED commit requires at least one SkillStore parent")

        first_parent = parent_records[0] if parent_records else None
        proposed_name = str(_attr(staged, "proposed_name") or target_dir.name)
        proposed_description = str(
            _attr(staged, "proposed_description")
            or getattr(first_parent, "description", "")
            or proposed_name
        )
        snapshot = _mapping_str_str(_attr(staged, "content_snapshot"))
        if not snapshot:
            snapshot = collect_skill_snapshot(target_dir)
        content_diff = str(_attr(staged, "content_diff") or "")
        source_task_id = (
            _packet_scope_value(action_packet, "task_id")
            or _none_or_str(_attr(decision, "source_task_id"))
            or _none_or_str(_attr(decision, "source_analysis_id"))
        )
        change_summary = _change_summary(staged, decision, action)
        provenance_refs = _dedupe_strs(evidence_refs)
        content_hash = _content_snapshot_hash(snapshot)
        revision_metadata = _revision_metadata(
            evidence_store=self.evidence_store,
            evidence_refs=provenance_refs,
            revision_id=skill_id,
            parent_revision_ids=parent_skill_ids,
            content_hash=content_hash,
        )

        if action_type == "FIX":
            assert first_parent is not None
            generation = first_parent.lineage.generation + 1
            origin = SkillOrigin.FIXED
            category = first_parent.category
            tags = list(first_parent.tags)
            visibility = first_parent.visibility
            creator_id = first_parent.creator_id
            tool_dependencies = _str_list(_attr(staged, "tool_dependencies")) or list(
                first_parent.tool_dependencies
            )
            critical_tools = _str_list(_attr(staged, "critical_tools")) or list(
                first_parent.critical_tools
            )
            path = first_parent.path or str(target_dir / SKILL_FILENAME)
        elif action_type == "DERIVED":
            assert first_parent is not None
            generation = max(record.lineage.generation for record in parent_records) + 1
            origin = SkillOrigin.DERIVED
            category = _decision_category(decision) or first_parent.category
            tags = sorted({tag for record in parent_records for tag in record.tags})
            visibility = first_parent.visibility
            creator_id = first_parent.creator_id
            tool_dependencies = _str_list(_attr(staged, "tool_dependencies")) or sorted(
                {tool for record in parent_records for tool in record.tool_dependencies}
            )
            critical_tools = _str_list(_attr(staged, "critical_tools")) or sorted(
                {tool for record in parent_records for tool in record.critical_tools}
            )
            path = str(target_dir / SKILL_FILENAME)
        else:
            generation = 0
            origin = SkillOrigin.CAPTURED
            category = _decision_category(decision) or SkillCategory.WORKFLOW
            tags = []
            visibility = getattr(first_parent, "visibility", None)
            if visibility is None:
                from openspace.skill_engine.types import SkillVisibility

                visibility = SkillVisibility.PRIVATE
            creator_id = getattr(first_parent, "creator_id", "")
            tool_dependencies = _str_list(_attr(staged, "tool_dependencies"))
            critical_tools = _str_list(_attr(staged, "critical_tools"))
            path = str(target_dir / SKILL_FILENAME)

        return SkillRecord(
            skill_id=skill_id,
            name=proposed_name,
            description=proposed_description,
            path=path,
            is_active=True,
            category=category,
            tags=tags,
            visibility=visibility,
            creator_id=creator_id,
            lineage=SkillLineage(
                origin=origin,
                revision_id=skill_id,
                generation=generation,
                parent_skill_ids=list(parent_skill_ids),
                parent_revision_ids=list(parent_skill_ids),
                source_task_id=source_task_id,
                change_summary=change_summary,
                content_hash=content_hash,
                content_diff=content_diff,
                content_snapshot=snapshot,
                evolution_action_id=action.action_id,
                provenance_refs=provenance_refs,
                revision_metadata=revision_metadata,
                created_by=str(_attr(authoring, "model") or ""),
            ),
            tool_dependencies=tool_dependencies,
            critical_tools=critical_tools,
        )

    def _resolve_commit_skill_id(
        self,
        *,
        staged: Any,
        action_type: str,
        target_dir: Path,
        parent_skill_ids: list[str],
    ) -> str:
        proposed_skill_id = _none_or_str(_attr(staged, "proposed_skill_id"))
        if proposed_skill_id:
            return proposed_skill_id
        parent_records = _load_parent_records(self.skill_store, parent_skill_ids)
        first_parent = parent_records[0] if parent_records else None
        proposed_name = str(_attr(staged, "proposed_name") or target_dir.name)
        return _new_skill_id(proposed_name, action_type, first_parent)

    def _materialize_local_category_tree(
        self,
        *,
        action_type: str,
        target_dir: Path,
        record: SkillRecord,
        decision: Any,
        parent_skill_ids: list[str],
    ) -> Path:
        try:
            from openspace.cloud.local_mapping import CloudLocalMappingStore
            from openspace.cloud.skill_classification import (
                build_local_category_path,
                classify_skill_dir,
                initialize_local_skill_taxonomy,
                materialize_skill_category_tree,
                persist_skill_classification,
            )

            db_path = getattr(self.skill_store, "db_path", None)
            if db_path is None and getattr(self.skill_store, "base", None) is not None:
                db_path = getattr(self.skill_store.base, "db_path", None)
            mapping_store = CloudLocalMappingStore(db_path)
            try:
                parent_records = _load_parent_records(self.skill_store, parent_skill_ids)
                if parent_records:
                    initialize_local_skill_taxonomy(
                        mapping_store=mapping_store,
                        skills=parent_records,
                    )
                parent_classification = None
                parent_cloud_path = ""
                for parent_id in parent_skill_ids:
                    parent_classification = mapping_store.get_skill_local_classification(parent_id)
                    parent_binding = mapping_store.get_skill_cloud_binding_by_local(parent_id)
                    if parent_binding is not None and not parent_cloud_path:
                        parent_cloud_path = (
                            parent_binding.current_package_path
                            or parent_binding.package_path_at_pull
                            or ""
                        )
                    if parent_classification is not None:
                        break

                decision_path = _decision_local_category_path(decision)
                inherited_path = ""
                if (
                    parent_classification is not None
                    and parent_classification.local_category_path
                    and action_type in {"FIX", "DERIVED"}
                ):
                    inherited_path = parent_classification.local_category_path
                selected_path = decision_path or inherited_path
                classification = classify_skill_dir(
                    target_dir,
                    local_skill_id=record.skill_id,
                    cloud_package_path=parent_cloud_path or None,
                    local_category=record.category.value,
                    local_category_path=selected_path,
                    origin=_classification_origin(action_type),
                )
                category = record.category.value
                local_category_path = build_local_category_path(
                    category,
                    local_category_path=selected_path,
                    cloud_package_path=parent_cloud_path or None,
                    local_path=str(target_dir),
                    name=record.name,
                )
                if not decision_path and parent_classification is not None:
                    if action_type == "FIX":
                        category = parent_classification.category
                        local_category_path = parent_classification.local_category_path
                    elif action_type == "DERIVED" and inherited_path:
                        local_category_path = parent_classification.local_category_path

                classification = replace(
                    classification,
                    category=category,
                    local_category_path=local_category_path,
                    evidence={
                        **dict(classification.evidence or {}),
                        "origin": _classification_origin(action_type),
                        "evolution_action_type": action_type,
                        "parent_skill_ids": list(parent_skill_ids),
                    },
                )
                saved = persist_skill_classification(mapping_store, classification)
                return materialize_skill_category_tree(
                    target_dir,
                    saved,
                    skills_root=target_dir.parent,
                )
            finally:
                mapping_store.close()
        except Exception as exc:
            logger.debug("local category tree materialization skipped: %s", exc)
            return target_dir

    def _refresh_registry(
        self,
        action_type: str,
        target_dir: Path,
        record: SkillRecord,
        parent_skill_ids: list[str],
    ) -> None:
        meta = self.registry.load_skill_from_dir(target_dir)
        if meta is None:
            raise RuntimeError(f"registry could not load skill from {target_dir}")
        if action_type == "FIX":
            old_skill_id = parent_skill_ids[0] if parent_skill_ids else record.skill_id
            self.registry.update_skill(old_skill_id, meta)
        else:
            self.registry.add_skill(meta)


    def _record_failure(
        self,
        action_id: str,
        phase: str,
        status: str,
        reason: str,
    ) -> None:
        recorder = getattr(self.evidence_store, "record_action_failure", None)
        if not callable(recorder):
            return
        try:
            recorder(action_id, phase=phase, status=status, error=reason)
        except Exception:
            logger.debug("Failed to record evolution action failure", exc_info=True)


def _commit_action_type(decision: Any, staged: Any = None) -> str:
    raw = (
        _attr(decision, "proposed_action")
        or _attr(decision, "action_type")
        or _attr(decision, "evolution_type")
        or _attr(staged, "action_type")
        or ""
    )
    return str(getattr(raw, "value", raw) or "").strip().upper()


def _classification_origin(action_type: str) -> str:
    return {
        "FIX": "fix",
        "DERIVED": "derive",
        "CAPTURED": "capture",
    }.get(action_type, "imported")


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _proposed_dir(staged: Any, target_dir: Path) -> Path:
    staging_dir = Path(str(_attr(staged, "staging_dir") or "")).expanduser().resolve()
    proposed_root = staging_dir / "proposed"
    preferred = proposed_root / target_dir.name
    if preferred.is_dir():
        return preferred
    proposed_name = str(_attr(staged, "proposed_name") or "")
    if proposed_name and (proposed_root / proposed_name).is_dir():
        return proposed_root / proposed_name
    children = [path for path in proposed_root.iterdir()] if proposed_root.is_dir() else []
    dirs = [path for path in children if path.is_dir()]
    if len(dirs) == 1:
        return dirs[0]
    raise RuntimeError(f"Cannot resolve proposed staging dir under {proposed_root}")


def _parent_skill_ids(staged: Any, decision: Any, action_type: str) -> list[str]:
    ids = _str_list(_attr(staged, "parent_skill_ids"))
    if not ids:
        ids = _str_list(_attr(staged, "target_skill_ids"))
    if not ids:
        ids = _str_list(_attr(decision, "target_skill_ids")) or _str_list(
            _attr(decision, "target_skills")
        )
    return [] if action_type == "CAPTURED" else _dedupe_strs(ids)


def _backup_target(target_dir: Path, backup_dir: Path) -> None:
    if not target_dir.is_dir():
        raise RuntimeError(f"active target dir not found: {target_dir}")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(target_dir, backup_dir)


def _apply_proposed_dir(action_type: str, proposed_dir: Path, target_dir: Path) -> None:
    if not proposed_dir.is_dir():
        raise RuntimeError(f"proposed staging dir not found: {proposed_dir}")
    if action_type == "FIX":
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(proposed_dir, target_dir)
        return
    if target_dir.exists():
        raise RuntimeError(f"active target already exists: {target_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(proposed_dir, target_dir)


def _rollback_disk(
    action_type: str,
    target_dir: Path,
    backup_dir: Path | None,
) -> None:
    if action_type == "FIX":
        if backup_dir is None or not backup_dir.is_dir():
            raise RuntimeError("missing backup for FIX rollback")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(backup_dir, target_dir)
        return
    if target_dir.exists():
        shutil.rmtree(target_dir)


def _load_parent_records(skill_store: Any, parent_skill_ids: list[str]) -> list[SkillRecord]:
    records: list[SkillRecord] = []
    for skill_id in parent_skill_ids:
        record = skill_store.load_record(skill_id)
        if record is None:
            raise RuntimeError(f"missing parent SkillRecord: {skill_id}")
        records.append(record)
    return records


def _new_skill_id(
    proposed_name: str,
    action_type: str,
    parent: SkillRecord | None,
) -> str:
    if action_type == "FIX" and parent is not None:
        generation = parent.lineage.generation + 1
        return f"{proposed_name}__v{generation}_{uuid.uuid4().hex[:8]}"
    return f"{proposed_name}__v0_{uuid.uuid4().hex[:8]}"


def _mapping_str_str(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _packet_scope_value(packet: Any, key: str) -> str | None:
    scope = _attr(packet, "scope")
    return _none_or_str(_attr(scope, key))


def _commit_raw_backrefs(
    authoring: Any,
    validation: Any,
    decision: Any,
    admission: Any,
    action_packet: Any,
    evidence_refs: list[str],
) -> list[str]:
    refs = [
        f"authoring:{_attr(authoring, 'authoring_id')}" if _attr(authoring, "authoring_id") else "",
        f"validation:{_attr(validation, 'validation_id')}" if _attr(validation, "validation_id") else "",
        f"decision:{_attr(decision, 'decision_id')}" if _attr(decision, "decision_id") else "",
        f"admission:{_attr(admission, 'admission_id')}" if _attr(admission, "admission_id") else "",
        f"packet:{_attr(action_packet, 'packet_id')}" if _attr(action_packet, "packet_id") else "",
        *evidence_refs,
    ]
    return _dedupe_strs(refs)


def _change_summary(staged: Any, decision: Any, action: EvolutionActionRecord) -> str:
    apply_metadata = _attr(staged, "apply_metadata")
    if isinstance(apply_metadata, Mapping):
        summary = str(apply_metadata.get("change_summary") or "").strip()
    else:
        summary = ""
    if not summary:
        summary = str(_attr(decision, "reason_summary") or "").strip()
    audit = f"evolution_action={action.action_id}; decision={action.decision_id}"
    return f"{summary}\n\nAudit: {audit}" if summary else f"Audit: {audit}"


def _content_snapshot_hash(snapshot: Mapping[str, str]) -> str:
    payload = json.dumps(
        {str(key): str(value) for key, value in sorted(snapshot.items())},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _revision_metadata(
    *,
    evidence_store: Any,
    evidence_refs: list[str],
    revision_id: str,
    parent_revision_ids: list[str],
    content_hash: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "revision_id": revision_id,
        "parent_revision_ids": list(parent_revision_ids),
        "content_hash": content_hash,
        "behavior_eval_refs": [
            ref for ref in evidence_refs if str(ref).startswith("behavior_eval:")
        ],
    }
    loader = getattr(evidence_store, "load_behavior_eval", None)
    if not callable(loader):
        return metadata
    evals: list[dict[str, Any]] = []
    for ref in metadata["behavior_eval_refs"]:
        eval_id = str(ref).split(":", 1)[1]
        try:
            result = loader(eval_id)
        except Exception:
            logger.debug("Failed to load behavior eval for revision metadata", exc_info=True)
            continue
        if result is None:
            continue
        replay = getattr(result, "replay_eval", None)
        evals.append(
            {
                "eval_id": getattr(result, "eval_id", eval_id),
                "outcome": getattr(result, "outcome", ""),
                "replay_run_id": getattr(replay, "replay_run_id", ""),
                "sandbox_run_id": getattr(replay, "sandbox_run_id", ""),
                "judge_result_id": getattr(replay, "judge_result_id", ""),
                "baseline_revision_set": list(
                    getattr(replay, "baseline_revision_set", []) or []
                ),
                "candidate_revision_set": list(
                    getattr(replay, "candidate_revision_set", []) or []
                ),
                "baseline_score": getattr(replay, "baseline_score", None),
                "candidate_score": getattr(replay, "candidate_score", None),
                "artifact_refs": list(getattr(replay, "artifact_refs", []) or []),
            }
        )
    if evals:
        metadata["behavior_evals"] = evals
        metadata["latest_behavior_eval"] = evals[-1]
    return metadata


def _decision_category(decision: Any) -> SkillCategory | None:
    value = _attr(decision, "category")
    if not value:
        return None
    try:
        return value if isinstance(value, SkillCategory) else SkillCategory(str(value))
    except ValueError:
        return None


def _decision_local_category_path(decision: Any) -> str:
    return str(_attr(decision, "local_category_path") or "").strip()


def _attach_skill_record(
    action: EvolutionActionRecord,
    record: SkillRecord | None,
) -> EvolutionActionRecord:
    if record is None:
        return action
    from dataclasses import replace

    return replace(action, skill_record=record)


def _replace_action(action: EvolutionActionRecord, **changes: Any) -> EvolutionActionRecord:
    from dataclasses import replace

    return replace(action, **changes)


def _dedupe_strs(values: list[Any]) -> list[str]:
    return [
        text
        for text in dict.fromkeys(str(item) for item in values if item is not None)
        if text
    ]


def _none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _strict_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return None


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _normalize_evolution_mode(value: str) -> str:
    mode = str(value or "autonomous").strip().lower()
    if mode not in _EVOLUTION_MODES:
        raise ValueError(
            "evolution_mode must be one of: audit_only, fix_only, autonomous"
        )
    return mode


def _decision_action_type(decision: Any) -> str:
    raw = (
        getattr(decision, "proposed_action", None)
        or getattr(decision, "action_type", None)
        or getattr(decision, "evolution_type", None)
        or _mapping_get(decision, "proposed_action")
        or _mapping_get(decision, "action_type")
        or _mapping_get(decision, "evolution_type")
        or ""
    )
    if isinstance(raw, EvolutionType):
        return raw.value
    return str(raw).strip().lower()


def _admission_outcome(admission: Any) -> str:
    raw = getattr(admission, "outcome", None) or _mapping_get(admission, "outcome")
    return str(raw or "noop").strip().lower()


def _validation_passed(validation: Any) -> bool:
    if validation is None:
        return False
    if isinstance(validation, bool):
        return validation
    outcome = getattr(validation, "outcome", None) or _mapping_get(validation, "outcome")
    if outcome is not None:
        return str(outcome).strip().lower() == "approve"
    raw = (
        getattr(validation, "passed", None)
        if getattr(validation, "passed", None) is not None
        else _mapping_get(validation, "passed")
    )
    if raw is None:
        raw = getattr(validation, "status", None) or _mapping_get(validation, "status")
        return str(raw).strip().lower() in {"passed", "success", "ok"}
    return _strict_bool(raw) is True


def _attach_behavior_eval_ref(
    validation: Any,
    behavior_result: SkillBehaviorEvalResult,
) -> Any:
    if validation is None:
        return validation
    ref_id = behavior_result.ref_id
    current = _str_list(
        getattr(validation, "provenance_refs", None)
        or _mapping_get(validation, "provenance_refs")
    )
    refs = _dedupe_strs([*current, ref_id])
    try:
        return replace(validation, provenance_refs=refs)
    except Exception:
        if isinstance(validation, dict):
            updated = dict(validation)
            updated["provenance_refs"] = refs
            return updated
        if hasattr(validation, "provenance_refs"):
            try:
                setattr(validation, "provenance_refs", refs)
                return validation
            except Exception:
                logger.debug(
                    "Failed to attach behavior eval ref to mutable validation object",
                    exc_info=True,
                )
    return validation


def _has_behavior_eval_ref(validation: Any, staged: Any = None) -> bool:
    del staged
    return bool(_validation_behavior_eval_refs(validation))


def _require_approved_behavior_eval(
    evidence_store: Any,
    *,
    authoring: Any,
    validation: Any,
    decision: Any | None,
    action_packet: Any | None,
    action_type: str,
) -> SkillBehaviorEvalResult:
    refs = _validation_behavior_eval_refs(validation)
    if not refs:
        raise ValueError("commit requires approved behavior eval provenance ref")
    loader = getattr(evidence_store, "load_behavior_eval", None)
    if not callable(loader):
        raise ValueError("commit requires behavior eval evidence store lookup")

    checked_failures: list[str] = []
    for ref in refs:
        eval_id = _behavior_eval_id_from_ref(ref)
        if not eval_id:
            continue
        try:
            result = loader(eval_id)
        except Exception as exc:
            checked_failures.append(f"{ref}:load_failed:{str(exc)[:120]}")
            logger.debug("Failed to load behavior eval ref %s", ref, exc_info=True)
            continue
        failures = _behavior_eval_commit_gate_failures(
            result,
            evidence_store=evidence_store,
            authoring=authoring,
            validation=validation,
            decision=decision,
            action_packet=action_packet,
            action_type=action_type,
        )
        if not failures:
            return result
        checked_failures.append(f"{ref}:{','.join(failures[:4])}")

    detail = f" ({'; '.join(checked_failures[:3])})" if checked_failures else ""
    raise ValueError(f"commit requires approved behavior eval provenance ref{detail}")


def _validation_behavior_eval_refs(validation: Any) -> list[str]:
    refs = _str_list(
        getattr(validation, "provenance_refs", None)
        or _mapping_get(validation, "provenance_refs")
    )
    return [
        ref
        for ref in refs
        if _behavior_eval_id_from_ref(ref)
    ]


def _behavior_eval_id_from_ref(ref: Any) -> str:
    text = str(ref or "").strip()
    if not text.startswith("behavior_eval:"):
        return ""
    eval_id = text.split(":", 1)[1].strip()
    return eval_id if eval_id else ""


def _behavior_eval_commit_gate_failures(
    result: Any,
    *,
    evidence_store: Any | None = None,
    authoring: Any,
    validation: Any,
    decision: Any | None,
    action_packet: Any | None,
    action_type: str,
) -> list[str]:
    if result is None:
        return ["missing_behavior_eval_result"]
    failures: list[str] = []
    if str(_attr(result, "outcome") or "").strip().lower() != "approve":
        failures.append("behavior_eval_not_approved")
    behavior_failures = _str_list(_attr(result, "failures"))
    if behavior_failures:
        failures.extend(f"behavior_eval_failure:{item}" for item in behavior_failures)
    optional_replay_allowed = _behavior_eval_allows_optional_replay(result)
    replay = _attr(result, "replay_eval")
    if not replay:
        if optional_replay_allowed:
            pass
        else:
            failures.append("missing_replay_eval")
    elif optional_replay_allowed:
        replay_mapping = _replay_eval_gate_mapping(replay)
        failures.extend(_replay_task_result_failures(replay_mapping))
    else:
        if _strict_bool(_attr(replay, "attempted")) is not True:
            failures.append("replay_eval_not_attempted")
        if _strict_bool(_attr(replay, "passed")) is not True:
            failures.append("replay_eval_not_passed")
        replay_mapping = _replay_eval_gate_mapping(replay)
        if not _replay_result_has_verified_executable_evidence(
            replay_mapping,
            evidence_store,
        ):
            failures.append("missing_executable_eval_evidence")
        failures.extend(_replay_task_result_failures(replay_mapping))

    expected = {
        "authoring_id": str(_attr(authoring, "authoring_id") or ""),
        "validation_id": str(_attr(validation, "validation_id") or ""),
        "decision_id": str(
            _attr(decision, "decision_id")
            or _attr(authoring, "decision_id")
            or ""
        ),
        "packet_id": str(_attr(action_packet, "packet_id") or ""),
        "action_type": str(action_type or "").strip().upper(),
    }
    for field_name, expected_value in expected.items():
        if not expected_value:
            failures.append(f"missing_behavior_eval_binding:{field_name}")
            continue
        actual = str(_attr(result, field_name) or "").strip()
        if field_name == "action_type":
            actual = actual.upper()
        if actual != expected_value:
            failures.append(f"behavior_eval_binding_mismatch:{field_name}")
    return _dedupe_strs(failures)


def _behavior_eval_allows_optional_replay(result: Any) -> bool:
    warnings = _str_list(_attr(result, "warnings"))
    if any(str(item).startswith("optional_replay_eval_") for item in warnings):
        return True
    replay = _attr(result, "replay_eval")
    if not replay:
        return False
    replay_warnings = _str_list(_attr(replay, "warnings"))
    return any(
        str(item).startswith("optional_replay_eval_") for item in replay_warnings
    )


def _replay_eval_gate_mapping(replay: Any) -> dict[str, Any]:
    if hasattr(replay, "to_dict") and callable(replay.to_dict):
        data = replay.to_dict()
        if isinstance(data, Mapping):
            return dict(data)
    if isinstance(replay, Mapping):
        return dict(replay)
    result: dict[str, Any] = {}
    for key in (
        "attempted",
        "passed",
        "runner",
        "replay_run_id",
        "sandbox_run_id",
        "judge_result_id",
        "baseline_revision_set",
        "candidate_revision_set",
        "baseline_score",
        "candidate_score",
        "artifact_refs",
        "details",
        "failures",
        "warnings",
    ):
        value = _attr(replay, key)
        if value is not None:
            result[key] = value
    return result


def _behavior_eval_blocked_reason(result: SkillBehaviorEvalResult) -> str:
    if result.failures:
        return f"behavior_eval_failed:{result.failures[0]}"
    return f"behavior_eval_{result.outcome}"


def _behavior_eval_candidate_reason(result: SkillBehaviorEvalResult) -> str:
    if result.failures:
        return f"behavior_eval_failed:{result.failures[0]}"
    return "behavior_eval_failed"


def _authoring_staged(authoring_result: Any) -> bool:
    status = getattr(authoring_result, "status", None) or _mapping_get(
        authoring_result, "status"
    )
    if status is None:
        return False
    return str(status).strip().lower() == "staged"


def _authoring_failure_reason(authoring_result: Any) -> str | None:
    for key in ("failure_reason", "error", "message"):
        raw = getattr(authoring_result, key, None) or _mapping_get(authoring_result, key)
        if raw:
            return str(raw).strip()[:500]
    return None


def _extract_skill_record(action: Any) -> Any | None:
    return (
        getattr(action, "skill_record", None)
        or getattr(action, "record", None)
        or _mapping_get(action, "skill_record")
        or _mapping_get(action, "record")
    )


def _commit_status(action: Any) -> str:
    raw = getattr(action, "commit_status", None) or _mapping_get(action, "commit_status")
    return str(raw or "").strip().lower()


def _action_id(action: Any) -> str:
    raw = getattr(action, "action_id", None) or _mapping_get(action, "action_id")
    return str(raw or "").strip()


def _explicit_promoted_candidate_ids(job: Any, action: Any) -> list[str]:
    ids = _explicit_candidate_ids_from_job(job)

    for key in (
        "candidate_id",
        "consumed_candidate_id",
        "promoted_candidate_id",
        "evolution_candidate_id",
    ):
        value = str(getattr(action, key, None) or _mapping_get(action, key) or "")
        if value:
            ids.append(value)

    for key in ("candidate_ids", "consumed_candidate_ids", "promoted_candidate_ids"):
        ids.extend(
            _str_list(getattr(action, key, None) or _mapping_get(action, key))
        )
    return _dedupe_strs(ids)


def _explicit_candidate_ids_from_job(job: Any) -> list[str]:
    ids: list[str] = []
    for tag in _str_list(
        getattr(job, "reason_tags", None) or _mapping_get(job, "reason_tags")
    ):
        if tag.startswith("candidate_id:"):
            ids.append(tag.split(":", 1)[1])
    return _dedupe_strs(ids)


def _candidate_ids_from_job_and_packet(job: Any, packet: Any) -> list[str]:
    del packet
    return _explicit_candidate_ids_from_job(job)


def _candidate_recheck_result_payload(
    result: EvolutionRunResult,
    *,
    mode: str,
) -> dict[str, Any]:
    return {
        "job_id": result.job_id,
        "run_status": result.status,
        "mode": mode,
        "decision_ids": [
            str(_attr(item, "decision_id") or "")
            for item in result.decisions
            if _attr(item, "decision_id")
        ],
        "admission_ids": [
            str(_attr(item, "admission_id") or "")
            for item in result.admissions
            if _attr(item, "admission_id")
        ],
        "admission_outcomes": [
            str(_attr(item, "outcome") or "")
            for item in result.admissions
            if _attr(item, "outcome")
        ],
        "candidate_ids": [
            str(_attr(item, "candidate_id") or "")
            for item in result.candidates
            if _attr(item, "candidate_id")
        ],
        "action_ids": [
            str(_attr(item, "action_id") or "")
            for item in result.actions
            if _attr(item, "action_id")
        ],
        "action_statuses": [
            str(_attr(item, "commit_status") or "")
            for item in result.actions
            if _attr(item, "commit_status")
        ],
        "behavior_eval_ids": [
            str(_attr(item, "eval_id") or "")
            for item in result.behavior_evals
            if _attr(item, "eval_id")
        ],
        "behavior_eval_outcomes": [
            str(_attr(item, "outcome") or "")
            for item in result.behavior_evals
            if _attr(item, "outcome")
        ],
        "errors": [str(item) for item in result.errors],
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _candidate_recheck_blocked_reason(
    result: EvolutionRunResult,
    *,
    mode: str,
) -> str | None:
    if any(_commit_status(action) in {"committed", "committed_reconciled"} for action in result.actions):
        return None
    if any(_commit_status(action) == "committing" for action in result.actions):
        return "commit_needs_recovery"
    for candidate in result.candidates:
        reason = _none_or_str(_attr(candidate, "blocked_reason")) or _none_or_str(
            _attr(candidate, "reason")
        )
        if reason:
            return reason
    error_reason = _candidate_recheck_error_reason(result.errors)
    if error_reason:
        return error_reason
    if result.errors:
        return "engine_failed"
    for admission in result.admissions:
        outcome = str(_attr(admission, "outcome") or "").lower()
        if outcome == "candidate":
            reason = _candidate_admission_reason(admission)
            return reason or "admission_candidate"
        if outcome in {"needs_human_review", "human_review"}:
            return "needs_human_review"
        if outcome in {"reject", "rejected"}:
            failures = _str_list(_attr(admission, "hard_failures"))
            return f"admission_rejected:{failures[0]}" if failures else "admission_rejected"
        if outcome == "noop":
            warnings = _str_list(_attr(admission, "warnings"))
            return f"admission_noop:{warnings[0]}" if warnings else "admission_noop"
    for decision in result.decisions:
        if mode == "fix_only" and _decision_action_type(decision) != EvolutionType.FIX.value:
            return "policy_blocked:fix_only_non_fix"
    if result.status.startswith("completed"):
        return None
    return "no_direct_action"


def _candidate_admission_reason(admission: Any) -> str | None:
    tags = [
        *_str_list(_attr(admission, "hard_failures")),
        *_str_list(_attr(admission, "warnings")),
    ]
    lowered = {str(tag).lower() for tag in tags}
    if "single_observation" in lowered:
        return "needs_more_evidence:additional_recurrence"
    if "no_derived_divergence" in lowered:
        return "needs_more_evidence:derived_divergence"
    if "reusable_boundary_uncertain" in lowered:
        return "needs_more_evidence:reusable_boundary"
    if "workflow_trivial_or_uncertain" in lowered:
        return "needs_more_evidence:workflow_significance"
    if "low_signal_capture" in lowered:
        return "needs_more_evidence:stronger_capture_signal"
    if "fallback_only_capture_evidence" in lowered:
        return "needs_more_evidence:primary_execution_evidence"
    if tags:
        return f"admission_candidate:{str(tags[0])}"
    return None


def _candidate_recheck_error_reason(errors: list[Any]) -> str | None:
    normalized = [str(error or "").strip() for error in errors if str(error or "").strip()]
    if "missing_behavior_eval" in normalized:
        return "missing_behavior_eval"
    for error in normalized:
        if error.startswith("behavior_eval_failed:"):
            return error
        if error in {
            "missing_behavior_eval",
            "missing_behavior_evaluator",
            "missing_required_replay_runner",
        }:
            return error
    return None


def _candidate_recheck_needed_evidence(result: EvolutionRunResult) -> list[str]:
    for candidate in result.candidates:
        needed = _str_list(_attr(candidate, "needed_evidence"))
        if needed:
            return needed
    needed: list[str] = []
    for admission in result.admissions:
        for tag in [
            *_str_list(_attr(admission, "hard_failures")),
            *_str_list(_attr(admission, "warnings")),
        ]:
            lower = str(tag).lower()
            if lower == "single_observation":
                needed.append("additional_recurrence")
            elif lower == "no_derived_divergence":
                needed.append("derived_divergence_evidence")
            elif lower in {"reusable_boundary_uncertain", "workflow_trivial_or_uncertain"}:
                needed.append("reusable_workflow_boundary_evidence")
            elif lower == "low_signal_capture":
                needed.append("stronger_successful_workflow_evidence")
            elif lower == "fallback_only_capture_evidence":
                needed.append("primary_runtime_or_transcript_evidence")
            elif lower.startswith("missing_") or lower.startswith("missing_ref:"):
                needed.append(str(tag))
    for error in result.errors:
        lower = str(error or "").strip().lower()
        if lower == "missing_behavior_eval":
            needed.append("behavior_eval_result")
        elif lower == "missing_behavior_evaluator":
            needed.append("behavior_evaluator")
        elif lower == "missing_required_replay_runner":
            needed.append("replay_runner")
    return _dedupe_strs(needed)


def _mapping_get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _decision_with_action_scope(
    decision: Any,
    admission: Any,
    packet: Any,
) -> Any:
    decision_id = getattr(decision, "decision_id", None) or _mapping_get(
        decision, "decision_id"
    )
    admission_id = getattr(admission, "admission_id", None) or _mapping_get(
        admission, "admission_id"
    )
    trigger_job_id = getattr(decision, "trigger_job_id", None) or _mapping_get(
        decision, "trigger_job_id"
    )
    return SimpleNamespace(
        decision_id=decision_id,
        admission_id=admission_id,
        trigger_job_id=trigger_job_id,
        packet=packet,
        packet_id=getattr(packet, "packet_id", None) or _mapping_get(packet, "packet_id"),
    )


def _accepts_positional_count(func: Any, count: int) -> bool:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return True
    positional = 0
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            positional += 1
    return positional >= count


def _accepts_keyword(func: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return True
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == name:
            return True
    return False


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    decisions = getattr(value, "decisions", None)
    if decisions is not None:
        return list(decisions or [])
    if isinstance(value, dict) and "decisions" in value:
        return list(value.get("decisions") or [])
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _evidence_store_for(engine: EvolutionEngine) -> Any | None:
    for owner in (
        getattr(engine, "packet_builder", None),
        getattr(engine, "decision_engine", None),
        getattr(engine, "admission_policy", None),
        getattr(engine, "candidate_store", None),
        getattr(engine, "validator", None),
        getattr(engine, "committer", None),
    ):
        store = getattr(owner, "evidence_store", None)
        if store is not None:
            return store
    return None


def _unwrap_packet_result(value: Any) -> Any:
    if value is None:
        return None
    status = getattr(value, "status", None) or _mapping_get(value, "status")
    packet = getattr(value, "packet", None) or _mapping_get(value, "packet")
    if status is not None and packet is not None:
        return packet if str(status).lower() == "ok" else None
    if status is not None and packet is None:
        return None
    return value


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _invoke_validator(
    method: Any,
    authoring_result: Any,
    validator_packet: Any,
    decision: Any,
    admission: Any,
    job: Any,
) -> Any:
    if _accepts_positional_count(method, 5):
        return await _maybe_await(
            method(authoring_result, validator_packet, decision, admission, job)
        )
    if _accepts_positional_count(method, 4):
        return await _maybe_await(
            method(authoring_result, validator_packet, decision, admission)
        )
    if _accepts_positional_count(method, 3):
        return await _maybe_await(method(authoring_result, decision, admission))
    if _accepts_positional_count(method, 2):
        return await _maybe_await(method(authoring_result, validator_packet))
    return await _maybe_await(method(authoring_result))
