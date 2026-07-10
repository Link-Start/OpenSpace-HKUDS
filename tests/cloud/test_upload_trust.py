import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openspace.cloud.cli import upload_skill as upload_skill_cli
from openspace.cloud.client import OpenSpaceClient
from openspace.cloud.upload_trust import (
    SkillUploadTrustError,
    require_trusted_skill_for_upload,
    require_trusted_skill_for_upload_db,
    resolve_upload_skill_store_db,
)
from openspace.skill_engine.store import SkillStore
from openspace.skill_engine.types import SkillRecord, SkillTrustState


def _skill_dir(tmp_path: Path, skill_id: str = "example__v0_test") -> Path:
    root = tmp_path / "skills" / "example"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: example\ndescription: Example workflow\n---\n",
        encoding="utf-8",
    )
    (root / ".skill_id").write_text(skill_id + "\n", encoding="utf-8")
    return root


def _record(
    root: Path,
    *,
    skill_id: str = "example__v0_test",
    trust_state: SkillTrustState = SkillTrustState.TRUSTED,
    enabled: bool = True,
) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id,
        name="example",
        description="Example workflow",
        path=str(root / "SKILL.md"),
        enabled=enabled,
        trust_state=trust_state,
    )


def test_trusted_upload_does_not_depend_on_enabled(tmp_path: Path) -> None:
    root = _skill_dir(tmp_path)
    record = _record(root, enabled=False)
    store = SimpleNamespace(load_record=lambda skill_id: record)

    assert require_trusted_skill_for_upload(root, skill_store=store) is record


def test_provisional_upload_is_blocked_with_structured_error(tmp_path: Path) -> None:
    root = _skill_dir(tmp_path)
    record = _record(root, trust_state=SkillTrustState.PROVISIONAL)
    store = SimpleNamespace(load_record=lambda skill_id: record)

    with pytest.raises(SkillUploadTrustError) as raised:
        require_trusted_skill_for_upload(root, skill_store=store)

    assert raised.value.code == "SKILL_NOT_TRUSTED"
    assert raised.value.to_payload()["actual_trust_state"] == "provisional"


def test_unknown_skill_id_fails_closed_without_path_fallback(tmp_path: Path) -> None:
    root = _skill_dir(tmp_path)
    store = SimpleNamespace(
        load_record=lambda skill_id: None,
        load_record_by_path=lambda path: _record(root),
    )

    with pytest.raises(SkillUploadTrustError) as raised:
        require_trusted_skill_for_upload(root, skill_store=store)

    assert raised.value.code == "SKILL_TRUST_UNKNOWN"


def test_copied_skill_id_cannot_reuse_another_trusted_record(tmp_path: Path) -> None:
    root = _skill_dir(tmp_path)
    other_root = tmp_path / "skills" / "other"
    record = _record(other_root)
    store = SimpleNamespace(load_record=lambda skill_id: record)

    with pytest.raises(SkillUploadTrustError) as raised:
        require_trusted_skill_for_upload(root, skill_store=store)

    assert raised.value.code == "SKILL_RECORD_PATH_MISMATCH"


def test_cli_store_resolution_prefers_explicit_then_nearest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _skill_dir(tmp_path)
    nearest = tmp_path / ".openspace" / "openspace.db"
    nearest.parent.mkdir()
    nearest.touch()
    explicit = tmp_path / "explicit.db"

    assert resolve_upload_skill_store_db(root, cwd=tmp_path) == nearest
    assert resolve_upload_skill_store_db(
        root,
        explicit_db_path=explicit,
        cwd=tmp_path,
    ) == explicit

    env_db = tmp_path / "env.db"
    monkeypatch.setenv("OPENSPACE_SKILL_STORE_DB_PATH", str(env_db))
    assert resolve_upload_skill_store_db(root, cwd=tmp_path) == env_db


def test_read_only_db_gate_allows_disabled_trusted_skill(tmp_path: Path) -> None:
    root = _skill_dir(tmp_path)
    db_path = tmp_path / "skills.db"
    store = SkillStore(db_path)
    try:
        asyncio.run(store.save_record(_record(root, enabled=False)))
    finally:
        store.close()
    modified_at = db_path.stat().st_mtime_ns

    record = require_trusted_skill_for_upload_db(root, db_path=db_path)

    assert record.trust_state == "trusted"
    assert db_path.stat().st_mtime_ns == modified_at


def test_missing_cli_store_fails_closed(tmp_path: Path) -> None:
    root = _skill_dir(tmp_path)

    with pytest.raises(SkillUploadTrustError) as raised:
        require_trusted_skill_for_upload_db(
            root,
            db_path=tmp_path / "missing.db",
        )

    assert raised.value.code == "SKILL_TRUST_UNKNOWN"


def test_cli_blocks_provisional_before_cloud_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _skill_dir(tmp_path)
    db_path = tmp_path / "skills.db"
    store = SkillStore(db_path)
    try:
        asyncio.run(
            store.save_record(
                _record(root, trust_state=SkillTrustState.PROVISIONAL)
            )
        )
    finally:
        store.close()

    def unexpected_cloud_config():
        raise AssertionError("cloud configuration must not run before trust preflight")

    monkeypatch.setattr(upload_skill_cli, "load_cloud_config", unexpected_cloud_config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "openspace-upload-skill",
            "--skill-dir",
            str(root),
            "--skill-store-db-path",
            str(db_path),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        upload_skill_cli.main()

    assert raised.value.code == 1
    assert "SKILL_NOT_TRUSTED" in capsys.readouterr().err


def test_low_level_cloud_client_cannot_bypass_local_trust_gate(
    tmp_path: Path,
) -> None:
    root = _skill_dir(tmp_path)
    db_path = tmp_path / "skills.db"
    store = SkillStore(db_path)
    try:
        asyncio.run(
            store.save_record(
                _record(root, trust_state=SkillTrustState.PROVISIONAL)
            )
        )
    finally:
        store.close()
    client = object.__new__(OpenSpaceClient)

    with pytest.raises(SkillUploadTrustError) as raised:
        client.upload_skill_v2(
            root,
            local_skill_store_db_path=db_path,
        )

    assert raised.value.code == "SKILL_NOT_TRUSTED"
