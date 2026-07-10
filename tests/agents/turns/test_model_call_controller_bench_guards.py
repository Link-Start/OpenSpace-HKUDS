import time
from types import SimpleNamespace

from openspace.agents.turns import loop
from openspace.agents.turns import model_call_controller


def _state() -> SimpleNamespace:
    return SimpleNamespace(current_iteration=4, max_iterations=80)


def test_pending_action_final_is_blocked_when_bench_guard_enabled(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_PENDING_ACTION_FINAL_GUARD", "true")

    assert model_call_controller._should_block_bench_pending_action_final(
        _state(),
        'Let me show you the commit and then merge it into master:',
        has_tool_calls=False,
    )


def test_completed_final_is_not_pending_action(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_PENDING_ACTION_FINAL_GUARD", "true")

    assert not model_call_controller._should_block_bench_pending_action_final(
        _state(),
        "Merged the lost commit into master and verified the working tree is clean.",
        has_tool_calls=False,
    )


def test_pending_action_guard_is_opt_in(monkeypatch):
    monkeypatch.delenv("OPENSPACE_BENCH_PENDING_ACTION_FINAL_GUARD", raising=False)

    assert not model_call_controller._should_block_bench_pending_action_final(
        _state(),
        "I'll run the checker next.",
        has_tool_calls=False,
    )


def test_bench_finalize_stop_after_iterations_is_hard_cap(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_FINALIZE_NUDGE_ENABLED", "true")
    monkeypatch.setenv("OPENSPACE_BENCH_FINALIZE_STOP_AFTER_ITERATIONS", "6")
    monkeypatch.setenv("OPENSPACE_BENCH_FINALIZE_STOP_AFTER_SEC", "0")
    state = SimpleNamespace(
        bench_finalize_nudge_count=1,
        bench_finalize_nudge_iteration=24,
        bench_finalize_nudge_monotonic=time.monotonic(),
        bench_finalize_last_tool_iteration=30,
        bench_finalize_last_tool_monotonic=time.monotonic(),
        current_iteration=31,
    )

    exhausted, reason = loop._bench_finalize_budget_exhausted(state)

    assert exhausted is True
    assert "after finalize nudge" in reason


def test_bench_checker_pass_stop_after_iterations(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_STOP_AFTER_CHECKER_PASS_ITERATIONS", "2")
    state = SimpleNamespace(
        bench_visible_checker_pass_iteration=8,
        bench_visible_checker_failed=False,
        current_iteration=11,
    )

    exhausted, reason = loop._bench_checker_pass_budget_exhausted(state)

    assert exhausted is True
    assert "visible checker pass" in reason


def test_bench_checker_pass_stop_ignores_unresolved_failure(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_STOP_AFTER_CHECKER_PASS_ITERATIONS", "2")
    state = SimpleNamespace(
        bench_visible_checker_pass_iteration=8,
        bench_visible_checker_failed=True,
        current_iteration=11,
    )

    exhausted, _ = loop._bench_checker_pass_budget_exhausted(state)

    assert exhausted is False
