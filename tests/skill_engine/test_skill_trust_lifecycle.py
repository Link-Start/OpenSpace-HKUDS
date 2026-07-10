import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from openspace.skill_engine.evidence import EvidenceStore
from openspace.skill_engine.evidence.types import (
    EvidencePacket,
    EvidenceScope,
    PacketBudget,
    ResourceRef,
)
from openspace.skill_engine.evolution.candidates import EvolutionCandidateStore
from openspace.skill_engine.evolution.admission import EvolutionAdmission
from openspace.skill_engine.evolution.audit import EvolutionAuditService
from openspace.skill_engine.evolution.engine import EvolutionCommitter
from openspace.skill_engine.protocol import SkillDiscoveryService
from openspace.skill_engine.store import SkillStore
from openspace.skill_engine.triggers import TriggerJobSpec, TriggerStore, default_policies
from openspace.skill_engine.types import (
    ExecutionAnalysis,
    SkillJudgment,
    SkillLineage,
    SkillOrigin,
    SkillRecord,
    SkillTrustState,
)


def _save(store: SkillStore, record: SkillRecord) -> None:
    asyncio.run(store.save_record(record))


def _captured_record(skill_id: str = "captured__v0_test") -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id,
        name="captured",
        description="Captured workflow",
        enabled=True,
        trust_state=SkillTrustState.PROVISIONAL,
        lineage=SkillLineage(
            origin=SkillOrigin.CAPTURED,
            source_task_id="task_origin",
        ),
    )


def _analysis(
    task_id: str,
    skill_id: str,
    *,
    completed: bool,
    phase_failed: bool = False,
) -> ExecutionAnalysis:
    return ExecutionAnalysis(
        task_id=task_id,
        timestamp=datetime.now(),
        task_completed=completed,
        skill_judgments=[
            SkillJudgment(
                skill_id=skill_id,
                skill_applied=True,
                note="workflow applied",
            )
        ],
        skill_phase_failed_skill_ids=[skill_id] if phase_failed else [],
    )


def _candidate_packet(task_id: str) -> EvidencePacket:
    runtime_ref = ResourceRef(
        ref_id=f"runtime:{task_id}",
        ref_type="runtime_snapshot",
        task_id=task_id,
        preview="task completed",
        metadata={"status": "success", "final_response_preview": "done"},
    )
    tool_ref = ResourceRef(
        ref_id=f"tool:{task_id}",
        ref_type="tool_event",
        task_id=task_id,
        preview="bash workflow completed",
        metadata={"tool_key": "shell:default:bash", "status": "success"},
    )
    return EvidencePacket(
        packet_id=f"packet:{task_id}",
        trigger_job_id=f"trigger:{task_id}",
        packet_type="analysis",
        profile_name="analysis_current_task",
        subprofile="task_finished",
        manifest_watermark=1,
        scope=EvidenceScope(task_id=task_id, source_task_ids=(task_id,)),
        selected_refs={"runtime_snapshot": [runtime_ref], "tool_event": [tool_ref]},
        expanded_snippets=[],
        readable_paths=[],
        instructions={},
        budget=PacketBudget(max_chars=1000, used_chars=10),
        redaction_status="ok",
        build_status="ok",
        missing_ref_types=[],
    )


def _candidate_decision(decision_id: str, summary: str) -> SimpleNamespace:
    return SimpleNamespace(
        decision_id=decision_id,
        proposed_action="CAPTURED",
        target_skill_ids=[],
        reason_summary=summary,
        reason_tags=["captured", "candidate_default"],
        local_category_path="local/workflow/test",
        evidence_claims=[
            SimpleNamespace(claim="workflow evidence", refs=["tool:task_one"])
        ],
    )


def _candidate_admission(admission_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        admission_id=admission_id,
        warnings=["provisional_evolution_disabled"],
        required_refs_checked=[],
    )


def test_origin_plus_one_cross_task_success_promotes_trust(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills.db")
    try:
        record = _captured_record()
        _save(store, record)
        asyncio.run(
            store.record_trust_observation(
                record.skill_id,
                "task:task_origin",
                "success",
                task_id="task_origin",
                source="evolution_origin",
            )
        )

        provisional = store.load_record(record.skill_id)
        assert provisional is not None
        assert provisional.trust_state == SkillTrustState.PROVISIONAL
        assert provisional.trust_successes == 1

        asyncio.run(
            store.record_analysis(
                _analysis("task_reuse", record.skill_id, completed=True)
            )
        )

        trusted = store.load_record(record.skill_id)
        assert trusted is not None
        assert trusted.trust_state == SkillTrustState.TRUSTED
        assert trusted.trust_successes == 2
        observations = store.load_trust_observations(record.skill_id)
        assert [item["task_id"] for item in observations] == [
            "task_origin",
            "task_reuse",
        ]
        assert observations[-1]["evidence_refs"] == ["analysis:task_reuse"]
    finally:
        store.close()


def test_committer_builds_evolved_revision_as_provisional(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills.db")
    try:
        committer = EvolutionCommitter(
            evidence_store=SimpleNamespace(),
            skill_store=store,
            registry=SimpleNamespace(),
        )
        record = committer._build_skill_record(
            action=SimpleNamespace(action_id="action_one", decision_id="decision_one"),
            staged=SimpleNamespace(
                proposed_name="captured",
                proposed_description="Captured workflow",
                content_snapshot={"SKILL.md": "# Captured\n"},
                content_diff="",
                tool_dependencies=[],
                critical_tools=[],
                apply_metadata={},
            ),
            authoring=SimpleNamespace(model="test-model"),
            decision=SimpleNamespace(
                reason_summary="Capture reusable workflow",
                local_category_path="local/workflow/test",
            ),
            action_packet=SimpleNamespace(
                scope=SimpleNamespace(task_id="task_origin")
            ),
            action_type="CAPTURED",
            target_dir=tmp_path / "captured",
            parent_skill_ids=[],
            evidence_refs=["runtime:task_origin"],
            skill_id="captured__v0_test",
        )

        assert record.trust_state == SkillTrustState.PROVISIONAL
        assert record.enabled is True
    finally:
        store.close()


def test_existing_skill_store_schema_migrates_to_trusted_enabled(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    now = datetime.now().isoformat()
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE skill_records (
            skill_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            path TEXT NOT NULL DEFAULT '',
            is_active INTEGER NOT NULL DEFAULT 1,
            category TEXT NOT NULL DEFAULT 'workflow',
            visibility TEXT NOT NULL DEFAULT 'private',
            creator_id TEXT NOT NULL DEFAULT '',
            lineage_origin TEXT NOT NULL DEFAULT 'imported',
            lineage_revision_id TEXT NOT NULL DEFAULT '',
            lineage_generation INTEGER NOT NULL DEFAULT 0,
            lineage_parent_revision_ids_json TEXT NOT NULL DEFAULT '[]',
            lineage_source_task_id TEXT,
            lineage_change_summary TEXT NOT NULL DEFAULT '',
            lineage_content_hash TEXT NOT NULL DEFAULT '',
            lineage_evolution_action_id TEXT,
            lineage_provenance_refs_json TEXT NOT NULL DEFAULT '[]',
            lineage_revision_metadata_json TEXT NOT NULL DEFAULT '{}',
            lineage_content_diff TEXT NOT NULL DEFAULT '',
            lineage_content_snapshot TEXT NOT NULL DEFAULT '{}',
            lineage_created_at TEXT NOT NULL,
            lineage_created_by TEXT NOT NULL DEFAULT '',
            total_selections INTEGER NOT NULL DEFAULT 0,
            total_applied INTEGER NOT NULL DEFAULT 0,
            total_completions INTEGER NOT NULL DEFAULT 0,
            total_fallbacks INTEGER NOT NULL DEFAULT 0,
            first_seen TEXT NOT NULL,
            last_updated TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        INSERT INTO skill_records (
            skill_id, name, lineage_created_at, first_seen, last_updated
        ) VALUES (?, ?, ?, ?, ?)
        """,
        ("legacy", "legacy", now, now, now),
    )
    conn.commit()
    conn.close()

    store = SkillStore(db_path)
    try:
        migrated = store.load_record("legacy")
        assert migrated is not None
        assert migrated.enabled is True
        assert migrated.trust_state == SkillTrustState.TRUSTED
    finally:
        store.close()


def test_duplicate_observation_is_idempotent(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills.db")
    try:
        record = _captured_record()
        _save(store, record)
        for _ in range(2):
            asyncio.run(
                store.record_trust_observation(
                    record.skill_id,
                    "task:task_origin",
                    "success",
                    task_id="task_origin",
                )
            )

        loaded = store.load_record(record.skill_id)
        assert loaded is not None
        assert loaded.trust_successes == 1
        assert loaded.trust_state == SkillTrustState.PROVISIONAL
    finally:
        store.close()


def test_trust_promotion_threshold_is_configurable(tmp_path: Path) -> None:
    store = SkillStore(
        tmp_path / "skills.db",
        trust_promotion_min_independent_successes=3,
    )
    try:
        record = _captured_record()
        _save(store, record)
        for task_id in ("task_origin", "task_reuse_one"):
            asyncio.run(
                store.record_trust_observation(
                    record.skill_id,
                    f"task:{task_id}",
                    "success",
                    task_id=task_id,
                )
            )
        assert store.load_record(record.skill_id).trust_state == SkillTrustState.PROVISIONAL

        asyncio.run(
            store.record_trust_observation(
                record.skill_id,
                "task:task_reuse_two",
                "success",
                task_id="task_reuse_two",
            )
        )
        assert store.load_record(record.skill_id).trust_state == SkillTrustState.TRUSTED
    finally:
        store.close()


def test_only_attributable_failure_demotes_trusted_skill(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills.db")
    try:
        record = SkillRecord(
            skill_id="trusted__v0_test",
            name="trusted",
            description="Trusted workflow",
            trust_state=SkillTrustState.TRUSTED,
        )
        _save(store, record)

        asyncio.run(
            store.record_analysis(
                _analysis("task_generic_failure", record.skill_id, completed=False)
            )
        )
        still_trusted = store.load_record(record.skill_id)
        assert still_trusted is not None
        assert still_trusted.trust_state == SkillTrustState.TRUSTED
        assert still_trusted.trust_failures == 0

        asyncio.run(
            store.record_analysis(
                _analysis(
                    "task_skill_failure",
                    record.skill_id,
                    completed=False,
                    phase_failed=True,
                )
            )
        )
        demoted = store.load_record(record.skill_id)
        assert demoted is not None
        assert demoted.trust_state == SkillTrustState.PROVISIONAL
        assert demoted.trust_failures == 1
    finally:
        store.close()


def test_repromotion_requires_fresh_successes_after_failure(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills.db")
    try:
        record = SkillRecord(
            skill_id="trusted__v0_test",
            name="trusted",
            description="Trusted workflow",
            trust_state=SkillTrustState.TRUSTED,
        )
        _save(store, record)
        asyncio.run(
            store.record_analysis(
                _analysis(
                    "task_failure",
                    record.skill_id,
                    completed=False,
                    phase_failed=True,
                )
            )
        )
        asyncio.run(
            store.record_analysis(
                _analysis("task_recovery_one", record.skill_id, completed=True)
            )
        )
        assert store.load_record(record.skill_id).trust_state == SkillTrustState.PROVISIONAL

        asyncio.run(
            store.record_analysis(
                _analysis("task_recovery_two", record.skill_id, completed=True)
            )
        )
        assert store.load_record(record.skill_id).trust_state == SkillTrustState.TRUSTED
    finally:
        store.close()


def test_enabled_is_independent_from_trust(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills.db")
    try:
        record = _captured_record()
        _save(store, record)

        assert asyncio.run(store.set_skill_enabled(record.skill_id, False)) is True
        disabled = store.load_record(record.skill_id)
        assert disabled is not None
        assert disabled.enabled is False
        assert disabled.trust_state == SkillTrustState.PROVISIONAL
        assert store.is_skill_enabled(record.skill_id) is False
        meta = SimpleNamespace(
            skill_id=record.skill_id,
            name=record.name,
            disable_model_invocation=False,
            conditional_paths=[],
        )
        registry = SimpleNamespace(list_skills=lambda: [meta])
        discovery = SkillDiscoveryService(registry, store=store)
        assert discovery._candidate_skills(None) == []
    finally:
        store.close()


def test_candidate_recurrence_counts_distinct_tasks_not_admissions(
    tmp_path: Path,
) -> None:
    store = EvolutionCandidateStore(tmp_path / "evidence.db")
    try:
        packet = _candidate_packet("task_one")
        first = store.create_or_merge(
            _candidate_decision("decision_one", "Capture the same workflow"),
            _candidate_admission("admission_one"),
            packet,
        )
        second = store.create_or_merge(
            _candidate_decision("decision_two", "Capture the same workflow"),
            _candidate_admission("admission_two"),
            packet,
        )

        assert first.candidate_id == second.candidate_id
        assert second.recurrence_count == 1
        assert second.recurrence == "single"
        assert not hasattr(store, "request_recheck")
        assert not hasattr(store, "mark_promoted")
        with pytest.raises(ValueError, match="only be rejected or superseded"):
            store.update_candidate_status(
                second.candidate_id,
                "promoted",
            )
    finally:
        store.close()


def test_candidate_recheck_trigger_is_not_supported(tmp_path: Path) -> None:
    store = TriggerStore(db_path=tmp_path / "evidence.db")
    try:
        job = store.create_job(
            TriggerJobSpec(
                trigger_type="CANDIDATE_RECHECK",
                reason="legacy_candidate_recheck",
                scope=EvidenceScope(),
                idempotency_key="legacy_candidate_recheck:test",
                evidence_profile="candidate_recheck",
                subprofile="candidate_recheck",
            ),
            manifest_watermark=0,
        )

        assert job.status == "rejected"
        assert job.error == "unknown trigger_type: CANDIDATE_RECHECK"
        assert "CANDIDATE_RECHECK" not in {
            policy.trigger_type for policy in default_policies()
        }
    finally:
        store.close()


def test_legacy_open_candidate_recheck_job_is_superseded(tmp_path: Path) -> None:
    db_path = tmp_path / "evidence.db"
    store = TriggerStore(db_path=db_path)
    try:
        job = store.create_job(
            TriggerJobSpec(
                trigger_type="CANDIDATE_RECHECK",
                reason="legacy_candidate_recheck",
                scope=EvidenceScope(),
                idempotency_key="legacy_candidate_recheck:open",
                evidence_profile="candidate_recheck",
                subprofile="candidate_recheck",
            ),
            manifest_watermark=0,
        )
    finally:
        store.close()

    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE trigger_jobs SET status='pending', completed_at=NULL, error=NULL "
        "WHERE job_id=?",
        (job.job_id,),
    )
    conn.commit()
    conn.close()

    migrated_store = TriggerStore(db_path=db_path)
    try:
        migrated = migrated_store.get_job(job.job_id)
        assert migrated is not None
        assert migrated.status == "superseded"
        assert migrated.completed_at is not None
        assert migrated.error == (
            "candidate recheck retired; evolution candidates are audit-only"
        )
    finally:
        migrated_store.close()


def test_candidate_review_items_are_inspect_only(tmp_path: Path) -> None:
    evidence_store = EvidenceStore(tmp_path / "evidence.db")
    candidate_store = EvolutionCandidateStore(evidence_store=evidence_store)
    try:
        candidate = candidate_store.create_or_merge(
            _candidate_decision("decision_one", "Capture the same workflow"),
            _candidate_admission("admission_one"),
            _candidate_packet("task_one"),
        )
        audit = EvolutionAuditService(
            evidence_store,
            candidate_store=candidate_store,
        )

        items = audit.list_review_items()

        assert len(items) == 1
        assert items[0]["candidate_id"] == candidate.candidate_id
        assert items[0]["action_kind"] == "inspect"
        assert items[0]["approval_available"] is False
        assert "never auto-promotes" in items[0]["review_note"]
    finally:
        candidate_store.close()
        evidence_store.close()


def test_single_observation_capture_is_admitted_as_provisional_by_default() -> None:
    packet = _candidate_packet("task_one")
    decision = _candidate_decision(
        "decision_one",
        "Capture this reusable multi-step workflow for future tasks",
    )

    result = EvolutionAdmission().admit(decision, packet)

    assert result.outcome == "direct"
    assert "single_observation_allowed" in result.warnings


def test_stale_repeated_metadata_does_not_override_single_source_task() -> None:
    packet = _candidate_packet("task_one")
    selected_refs = dict(packet.selected_refs)
    selected_refs["evolution_candidate_ref"] = [
        ResourceRef(
            ref_id="candidate:stale",
            ref_type="evolution_candidate_ref",
            task_id="task_one",
            preview="stale repeated candidate",
            metadata={
                "candidate_id": "stale",
                "recurrence": "repeated",
                "recurrence_count": 2,
                "source_task_ids": ["task_one"],
            },
        )
    ]
    packet = EvidencePacket(
        packet_id=packet.packet_id,
        trigger_job_id=packet.trigger_job_id,
        packet_type=packet.packet_type,
        profile_name=packet.profile_name,
        subprofile=packet.subprofile,
        manifest_watermark=packet.manifest_watermark,
        scope=packet.scope,
        selected_refs=selected_refs,
        expanded_snippets=packet.expanded_snippets,
        readable_paths=packet.readable_paths,
        instructions=packet.instructions,
        budget=packet.budget,
        redaction_status=packet.redaction_status,
        build_status=packet.build_status,
        missing_ref_types=packet.missing_ref_types,
    )

    result = EvolutionAdmission(
        allow_single_observation_capture=False
    ).admit(
        _candidate_decision(
            "decision_one",
            "Capture this reusable multi-step workflow for future tasks",
        ),
        packet,
    )

    assert result.outcome == "candidate"
    assert "provisional_evolution_disabled" in result.warnings


def test_only_structured_manual_evidence_bypasses_disabled_provisional_policy() -> None:
    packet = _candidate_packet("task_one")
    decision = _candidate_decision(
        "decision_one",
        "Capture this explicit reusable multi-step workflow for future tasks",
    )
    admission = EvolutionAdmission(allow_single_observation_capture=False)

    text_only_result = admission.admit(decision, packet)

    assert text_only_result.outcome == "candidate"
    selected_refs = dict(packet.selected_refs)
    selected_refs["manual_request_ref"] = [
        ResourceRef(
            ref_id="manual:capture",
            ref_type="manual_request_ref",
            task_id="task_one",
            preview="Capture this workflow",
        )
    ]
    manual_packet = EvidencePacket(
        packet_id="packet_manual",
        trigger_job_id=packet.trigger_job_id,
        packet_type=packet.packet_type,
        profile_name=packet.profile_name,
        subprofile=packet.subprofile,
        manifest_watermark=packet.manifest_watermark,
        scope=packet.scope,
        selected_refs=selected_refs,
        expanded_snippets=packet.expanded_snippets,
        readable_paths=packet.readable_paths,
        instructions=packet.instructions,
        budget=packet.budget,
        redaction_status=packet.redaction_status,
        build_status=packet.build_status,
        missing_ref_types=packet.missing_ref_types,
    )

    manual_result = admission.admit(decision, manual_packet)

    assert manual_result.outcome == "direct"
    assert "provisional_evolution_disabled" not in manual_result.warnings


def test_candidate_identity_keeps_unrelated_workflows_separate(tmp_path: Path) -> None:
    store = EvolutionCandidateStore(tmp_path / "evidence.db")
    try:
        packet = _candidate_packet("task_one")
        store.create_or_merge(
            _candidate_decision("decision_one", "Bound large shell output"),
            _candidate_admission("admission_one"),
            packet,
        )
        store.create_or_merge(
            _candidate_decision("decision_two", "Decode constants from an ELF"),
            _candidate_admission("admission_two"),
            packet,
        )

        candidates = store.list_candidates()
        assert len(candidates) == 2
        assert {candidate.recurrence_count for candidate in candidates} == {1}
    finally:
        store.close()


def test_candidate_store_repairs_legacy_false_recurrence(tmp_path: Path) -> None:
    db_path = tmp_path / "evidence.db"
    store = EvolutionCandidateStore(db_path)
    try:
        candidate = store.create_or_merge(
            _candidate_decision("decision_one", "Capture the same workflow"),
            _candidate_admission("admission_one"),
            _candidate_packet("task_one"),
        )
    finally:
        store.close()

    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE evolution_candidates SET recurrence='repeated', recurrence_count=2 "
        "WHERE candidate_id=?",
        (candidate.candidate_id,),
    )
    conn.commit()
    conn.close()

    repaired_store = EvolutionCandidateStore(db_path)
    try:
        repaired = repaired_store.load_candidate(candidate.candidate_id)
        assert repaired is not None
        assert repaired.recurrence == "single"
        assert repaired.recurrence_count == 1
    finally:
        repaired_store.close()
