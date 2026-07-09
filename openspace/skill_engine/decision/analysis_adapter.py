"""Adapter from analyzer proposals to DecisionRationale."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from openspace.skill_engine.evidence import EvidencePacket, ResourceRef
from openspace.skill_engine.types import (
    EvolutionSuggestion,
    EvolutionType,
    ExecutionAnalysis,
)

from .types import DecisionRationale, EvidenceClaim

_PRIMARY_CAPTURE_REF_TYPES = {
    "manual_request_ref",
    "tool_event",
    "tool_result",
    "tool_incident",
    "file_history",
    "skill_event",
}
_FALLBACK_CAPTURE_REF_TYPES = {
    "memory_ref",
    "recording_ref",
    "background_task_result",
}


class AnalyzerDecisionAdapter:
    """Convert analyzer proposals into evidence-backed decisions."""

    def from_analysis(
        self,
        analysis: ExecutionAnalysis,
        packet: EvidencePacket,
        *,
        source_analysis_id: str | None = None,
    ) -> list[DecisionRationale]:
        if not analysis.evolution_suggestions:
            fallback = _fallback_capture_suggestion(analysis, packet)
            if fallback is None:
                return []
            return [
                self._captured_decision(
                    analysis,
                    packet,
                    fallback,
                    index=0,
                    source_analysis_id=source_analysis_id,
                    extra_reason_tags=["fallback_no_evolution_suggestion"],
                )
            ]

        decisions: list[DecisionRationale] = []
        for index, suggestion in enumerate(analysis.evolution_suggestions):
            if suggestion.evolution_type == EvolutionType.FIX:
                decisions.append(
                    self._fix_decision(
                        analysis,
                        packet,
                        suggestion,
                        index=index,
                        source_analysis_id=source_analysis_id,
                    )
                )
            elif suggestion.evolution_type == EvolutionType.DERIVED:
                decisions.append(
                    self._derived_decision(
                        analysis,
                        packet,
                        suggestion,
                        index=index,
                        source_analysis_id=source_analysis_id,
                    )
                )
            elif suggestion.evolution_type == EvolutionType.CAPTURED:
                decision = self._captured_decision(
                    analysis,
                    packet,
                    suggestion,
                    index=index,
                    source_analysis_id=source_analysis_id,
                )
                decisions.append(decision)
        return decisions

    def _fix_decision(
        self,
        analysis: ExecutionAnalysis,
        packet: EvidencePacket,
        suggestion: EvolutionSuggestion,
        *,
        index: int,
        source_analysis_id: str | None,
    ) -> DecisionRationale:
        target_ids = list(dict.fromkeys(suggestion.target_skill_ids))
        lifecycle_refs = _skill_lifecycle_ref_ids(packet, target_ids)
        skill_file_refs = _skill_file_ref_ids(packet, target_ids)
        friction_refs = _failure_or_friction_ref_ids(packet)
        manual_refs = _manual_request_ref_ids(packet)
        signal_refs = _quality_signal_ref_ids(packet)
        claims: list[EvidenceClaim] = []
        if signal_refs:
            claims.append(
                EvidenceClaim(
                    "rule-based quality signal selected this packet for semantic attribution",
                    refs=signal_refs[:4],
                    confidence="high",
                )
            )
        if lifecycle_refs:
            claims.append(
                EvidenceClaim(
                    "target skill was selected, invoked, or applied in this scope",
                    refs=lifecycle_refs[:8],
                    confidence="high",
                )
            )
        if friction_refs:
            claims.append(
                EvidenceClaim(
                    "failure or friction is visible in runtime, transcript, or tool evidence",
                    refs=friction_refs[:8],
                    confidence="medium",
                )
            )
        if manual_refs:
            claims.append(
                EvidenceClaim(
                    "user explicitly requested repair of the target skill",
                    refs=manual_refs[:4],
                    confidence="high",
                )
            )
        if skill_file_refs:
            claims.append(
                EvidenceClaim(
                    "target skill source is available for downstream admission",
                    refs=skill_file_refs[:8],
                    confidence="high",
                )
            )
        policy = (
            "direct"
            if target_ids and skill_file_refs and (lifecycle_refs or manual_refs)
            else "reject"
        )
        risks: list[str] = []
        if not target_ids:
            risks.append("FIX proposal did not name a target skill")
        if target_ids and not skill_file_refs:
            risks.append("FIX target skill has no skill_file ref in the packet")
        if target_ids and not lifecycle_refs and not manual_refs:
            risks.append("FIX target skill lacks lifecycle evidence in the packet")
        if not friction_refs and not manual_refs:
            risks.append("FIX suggestion lacks direct failure or friction refs")
        return _decision(
            packet,
            analysis,
            suggestion,
            index=index,
            proposed_action="FIX",
            candidate_policy=policy,
            target_skill_ids=target_ids,
            reason_tags=[
                "analyzer_adapter",
                "fix",
                *_quality_signal_reason_tags(packet),
            ],
            evidence_claims=claims,
            confidence=0.72 if policy == "direct" else 0.25,
            risks=risks,
            source_analysis_id=source_analysis_id,
        )

    def _derived_decision(
        self,
        analysis: ExecutionAnalysis,
        packet: EvidencePacket,
        suggestion: EvolutionSuggestion,
        *,
        index: int,
        source_analysis_id: str | None,
    ) -> DecisionRationale:
        target_ids = list(dict.fromkeys(suggestion.target_skill_ids))
        skill_file_refs = _skill_file_ref_ids(packet, target_ids)
        supporting_refs = _supporting_workflow_ref_ids(packet)
        claims: list[EvidenceClaim] = []
        if skill_file_refs:
            claims.append(
                EvidenceClaim(
                    "parent skill source is available for derived-skill review",
                    refs=skill_file_refs[:8],
                    confidence="high",
                )
            )
        if supporting_refs:
            claims.append(
                EvidenceClaim(
                    "packet contains task evidence for a possible workflow divergence",
                    refs=supporting_refs[:10],
                    confidence="medium",
                )
            )
        policy = "candidate"
        risks: list[str] = []
        if target_ids and not skill_file_refs:
            policy = "reject"
            risks.append("DERIVED target skill has no skill_file ref in the packet")
        return _decision(
            packet,
            analysis,
            suggestion,
            index=index,
            proposed_action="DERIVED",
            candidate_policy=policy,
            target_skill_ids=target_ids,
            reason_tags=[
                "analyzer_adapter",
                "derived",
                "candidate_default",
                *_quality_signal_reason_tags(packet),
            ],
            evidence_claims=claims,
            confidence=0.55 if policy == "candidate" else 0.2,
            risks=risks,
            source_analysis_id=source_analysis_id,
        )

    def _captured_decision(
        self,
        analysis: ExecutionAnalysis,
        packet: EvidencePacket,
        suggestion: EvolutionSuggestion,
        *,
        index: int,
        source_analysis_id: str | None,
        extra_reason_tags: list[str] | None = None,
    ) -> DecisionRationale:
        if not _task_completed(analysis, packet):
            return _noop_from_suggestion(
                packet,
                analysis,
                suggestion,
                index=index,
                noop_reason="captured_from_incomplete_task",
                tags=["analyzer_adapter", "captured", "task_incomplete"],
                source_analysis_id=source_analysis_id,
            )
        primary_refs = _primary_capture_ref_ids(packet)
        if not primary_refs:
            return _noop_from_suggestion(
                packet,
                analysis,
                suggestion,
                index=index,
                noop_reason="insufficient_capture_evidence",
                tags=["analyzer_adapter", "captured", "insufficient_capture_evidence"],
                source_analysis_id=source_analysis_id,
            )

        refs = _supporting_workflow_ref_ids(packet)
        claims = [
            EvidenceClaim(
                "completed task evidence may contain a reusable workflow pattern",
                refs=refs[:12],
                confidence="medium",
            ),
            EvidenceClaim(
                "packet contains primary workflow evidence for admission review",
                refs=primary_refs[:12],
                confidence="medium",
            )
        ]
        refs_types = _ref_types_for_ref_ids(packet, [ref for claim in claims for ref in claim.refs])
        risks: list[str] = []
        if refs_types and refs_types.issubset(_FALLBACK_CAPTURE_REF_TYPES):
            risks.append("CAPTURED decision only has fallback memory/recording evidence")
        return _decision(
            packet,
            analysis,
            suggestion,
            index=index,
            proposed_action="CAPTURED",
            candidate_policy="candidate",
            target_skill_ids=[],
            reason_tags=[
                "analyzer_adapter",
                "captured",
                "candidate_default",
                *(extra_reason_tags or []),
                *_quality_signal_reason_tags(packet),
            ],
            evidence_claims=claims,
            confidence=0.5,
            risks=risks,
            source_analysis_id=source_analysis_id,
        )


def packet_conflicts_with_analysis(
    analysis: ExecutionAnalysis,
    packet: EvidencePacket,
) -> list[str]:
    """Return conflicts where packet facts should dominate analyzer proposals."""

    risks: list[str] = []
    # Packet completion is the runtime fact.  The adapter below already treats
    # ``analysis.task_completed or packet_completed`` as sufficient for captured
    # decisions, so an analyzer false-negative about completion should not
    # discard otherwise usable evolution suggestions.
    if _packet_indicates_completed(packet):
        return risks
    return risks


def _fallback_capture_suggestion(
    analysis: ExecutionAnalysis,
    packet: EvidencePacket,
) -> EvolutionSuggestion | None:
    if not _task_completed(analysis, packet):
        return None
    if not _primary_capture_ref_ids(packet):
        return None

    return EvolutionSuggestion(
        evolution_type=EvolutionType.CAPTURED,
        direction=(
            "Capture any reusable workflow pattern from the completed task. "
            "The analyzer produced no explicit evolution suggestion, but the "
            "runtime packet contains completion evidence and primary workflow "
            "refs for admission review."
        ),
    )


def _decision(
    packet: EvidencePacket,
    analysis: ExecutionAnalysis,
    suggestion: EvolutionSuggestion,
    *,
    index: int,
    proposed_action: str,
    candidate_policy: str,
    target_skill_ids: list[str],
    reason_tags: list[str],
    evidence_claims: list[EvidenceClaim],
    confidence: float,
    risks: list[str],
    source_analysis_id: str | None,
) -> DecisionRationale:
    created_at = _utc_now()
    summary = suggestion.direction.strip() or analysis.execution_note.strip()
    payload = {
        "packet_id": packet.packet_id,
        "trigger_job_id": packet.trigger_job_id,
        "task_id": analysis.task_id,
        "index": index,
        "action": proposed_action,
        "targets": target_skill_ids,
        "local_category_path": suggestion.local_category_path,
        "direction": suggestion.direction,
    }
    return DecisionRationale(
        decision_id=f"dec_{_digest(payload)}",
        trigger_job_id=packet.trigger_job_id,
        proposed_action=proposed_action,
        candidate_policy=candidate_policy,
        target_skill_ids=target_skill_ids,
        reason_summary=summary[:1000],
        reason_tags=list(dict.fromkeys(reason_tags)),
        evidence_claims=evidence_claims,
        confidence=max(0.0, min(1.0, confidence)),
        risks=list(dict.fromkeys(risks)),
        source_analysis_id=source_analysis_id,
        noop_reason=None,
        analyzed_by=analysis.analyzed_by or "execution_analyzer",
        created_at=created_at,
        local_category_path=str(suggestion.local_category_path or ""),
    )


def _noop_from_suggestion(
    packet: EvidencePacket,
    analysis: ExecutionAnalysis,
    suggestion: EvolutionSuggestion,
    *,
    index: int,
    noop_reason: str,
    tags: list[str],
    source_analysis_id: str | None,
) -> DecisionRationale:
    refs = _supporting_workflow_ref_ids(packet)[:8]
    claims = [
        EvidenceClaim(
            "packet evidence does not support a safe skill mutation",
            refs=refs,
            confidence="medium",
        )
    ] if refs else []
    decision = _decision(
        packet,
        analysis,
        suggestion,
        index=index,
        proposed_action="NOOP",
        candidate_policy="reject",
        target_skill_ids=[],
        reason_tags=tags,
        evidence_claims=claims,
        confidence=0.85,
        risks=[],
        source_analysis_id=source_analysis_id,
    )
    return replace(decision, noop_reason=noop_reason)


def _task_completed(analysis: ExecutionAnalysis, packet: EvidencePacket) -> bool:
    return bool(analysis.task_completed or _packet_indicates_completed(packet))


def _packet_indicates_completed(packet: EvidencePacket) -> bool:
    saw_runtime_status = False
    for ref in _refs(packet, "runtime_snapshot"):
        metadata = ref.metadata
        status = str(metadata.get("status") or "").lower()
        final_response = str(metadata.get("final_response_preview") or "").strip()
        if status:
            saw_runtime_status = True
        if status in {"success", "completed", "ok"} and final_response:
            return True
    if saw_runtime_status:
        return False
    return _final_assistant_ref(packet) is not None


def _final_assistant_ref(packet: EvidencePacket) -> ResourceRef | None:
    assistants = [
        ref
        for ref in _refs(packet, "transcript_message")
        if str(ref.metadata.get("role") or "").lower() == "assistant"
        and (ref.preview or ref.metadata.get("preview"))
    ]
    marked = [
        ref
        for ref in assistants
        if ref.metadata.get("is_final_response")
        or ref.metadata.get("final_response")
        or ref.metadata.get("message_kind") in {"final", "final_response"}
    ]
    if marked:
        return marked[-1]
    return assistants[-1] if assistants else None


def _skill_file_ref_ids(packet: EvidencePacket, target_skill_ids: list[str]) -> list[str]:
    target_set = set(target_skill_ids)
    refs: list[str] = []
    for ref in _refs(packet, "skill_file"):
        skill_id = str(ref.metadata.get("skill_id") or "")
        if not target_set or skill_id in target_set or any(t in ref.ref_id for t in target_set):
            refs.append(ref.ref_id)
    return list(dict.fromkeys(refs))


def _skill_lifecycle_ref_ids(packet: EvidencePacket, target_skill_ids: list[str]) -> list[str]:
    target_set = set(target_skill_ids)
    refs: list[str] = []
    for ref_type in ("skill_event", "skill_record"):
        for ref in _refs(packet, ref_type):
            skill_values = _metadata_values(ref.metadata, "skill_id", "skill_ids")
            if not target_set or skill_values.intersection(target_set):
                refs.append(ref.ref_id)
    return list(dict.fromkeys(refs))


def _manual_request_ref_ids(packet: EvidencePacket) -> list[str]:
    return list(dict.fromkeys(ref.ref_id for ref in _refs(packet, "manual_request_ref")))


def _failure_or_friction_ref_ids(packet: EvidencePacket) -> list[str]:
    refs: list[str] = []
    for ref in _refs(packet, "runtime_snapshot"):
        metadata = ref.metadata
        status = str(metadata.get("status") or "").lower()
        stop_reason = str(metadata.get("stop_reason") or "").lower()
        if status not in {"", "success", "completed", "ok"} or stop_reason in {
            "max_iterations",
            "error",
            "cancelled",
        }:
            refs.append(ref.ref_id)
    for ref_type in ("tool_event", "tool_result", "tool_incident"):
        for ref in _refs(packet, ref_type):
            metadata = ref.metadata
            status = str(
                metadata.get("status")
                or metadata.get("outcome")
                or metadata.get("result")
                or ""
            ).lower()
            if status and status not in {"success", "ok", "completed", "passed"}:
                refs.append(ref.ref_id)
            elif metadata.get("error_type") or metadata.get("error_message"):
                refs.append(ref.ref_id)
    for ref in _refs(packet, "skill_event"):
        status = str(ref.metadata.get("event_type") or ref.metadata.get("status") or "").lower()
        if status in {"fallback", "failed", "error", "permission_denied"}:
            refs.append(ref.ref_id)
    return list(dict.fromkeys(refs))


def _supporting_workflow_ref_ids(packet: EvidencePacket) -> list[str]:
    preferred = [
        "manual_request_ref",
        "runtime_snapshot",
        "transcript_message",
        "tool_event",
        "tool_result",
        "file_history",
        "skill_event",
        "skill_file",
    ]
    refs: list[str] = []
    for ref_type in preferred:
        refs.extend(ref.ref_id for ref in _refs(packet, ref_type))
    if not refs:
        for ref_type, items in sorted(packet.selected_refs.items()):
            refs.extend(ref.ref_id for ref in items)
    return list(dict.fromkeys(refs))


def _primary_capture_ref_ids(packet: EvidencePacket) -> list[str]:
    refs: list[str] = []
    for ref_type in sorted(_PRIMARY_CAPTURE_REF_TYPES):
        refs.extend(ref.ref_id for ref in _refs(packet, ref_type))
    return list(dict.fromkeys(refs))


def _ref_types_for_ref_ids(packet: EvidencePacket, ref_ids: list[str]) -> set[str]:
    wanted = set(ref_ids)
    return {
        ref.ref_type
        for refs in packet.selected_refs.values()
        for ref in refs
        if ref.ref_id in wanted
    }


def _refs(packet: EvidencePacket, ref_type: str) -> list[ResourceRef]:
    return list(packet.selected_refs.get(ref_type) or [])


def _quality_signal_reason_tags(packet: EvidencePacket) -> list[str]:
    if str(getattr(packet, "profile_name", "") or "") != "quality_signal":
        return []
    return ["quality_signal"]


def _quality_signal_ref_ids(packet: EvidencePacket) -> list[str]:
    return list(dict.fromkeys(ref.ref_id for ref in _refs(packet, "quality_signal_ref")))


def _metadata_values(metadata: dict[str, Any], *keys: str) -> set[str]:
    values: set[str] = set()
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            values.add(value)
        elif isinstance(value, (list, tuple, set)):
            values.update(str(item) for item in value if str(item))
    return values


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
