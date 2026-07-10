import json
from types import SimpleNamespace

from openspace.agents.turns import bench_checker_guard


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        current_iteration=1,
        bench_visible_checker_failed=False,
        bench_visible_checker_failure_iteration=None,
        bench_visible_checker_failure_command=None,
        bench_visible_checker_failure_excerpt=None,
        bench_visible_checker_pass_iteration=None,
    )


def _shell_turn(command: str, *, content: str, status: str) -> tuple[list[dict], list[dict]]:
    return (
        [
            {
                "id": "call-1",
                "function": {
                    "name": "shell",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
        [
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": content,
                "_meta": {"status": status},
            }
        ],
    )


def _file_turn(path: str, *, content: str, status: str) -> tuple[list[dict], list[dict]]:
    return (
        [
            {
                "id": "call-1",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"file_path": path}),
                },
            }
        ],
        [
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": content,
                "_meta": {"status": status},
            }
        ],
    )


def test_failed_file_discovery_command_is_not_checker_failure(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_CHECKER_FAILURE_GUARD", "true")
    state = _state()
    tool_calls, result_messages = _shell_turn(
        "ls -la /app/*.py /app/*.sh /app/Makefile 2>/dev/null",
        content="Command failed with exit code 2",
        status="error",
    )

    bench_checker_guard.update_from_tool_turn(
        state,
        tool_calls=tool_calls,
        result_messages=result_messages,
    )

    assert state.bench_visible_checker_failed is False


def test_reading_checker_source_is_not_checker_execution(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_CHECKER_FAILURE_GUARD", "true")
    state = _state()
    tool_calls, result_messages = _file_turn(
        "/app/check.py",
        content="raise AssertionError('Match: False')",
        status="success",
    )

    bench_checker_guard.update_from_tool_turn(
        state,
        tool_calls=tool_calls,
        result_messages=result_messages,
    )

    assert state.bench_visible_checker_failed is False


def test_finding_checker_source_is_not_checker_pass(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_CHECKER_FAILURE_GUARD", "true")
    state = _state()
    tool_calls, result_messages = _shell_turn(
        'find / -name "test_outputs.py" 2>/dev/null',
        content="/tests/test_outputs.py",
        status="success",
    )

    bench_checker_guard.update_from_tool_turn(
        state,
        tool_calls=tool_calls,
        result_messages=result_messages,
    )

    assert state.bench_visible_checker_failed is False
    assert state.bench_visible_checker_pass_iteration is None


def test_mixed_inspection_and_unrelated_python_is_not_checker_pass(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_CHECKER_FAILURE_GUARD", "true")
    state = _state()
    tool_calls, result_messages = _shell_turn(
        (
            'ls -la /tests 2>/dev/null; echo "---"; '
            'find / -name "test_outputs.py" 2>/dev/null; '
            'echo "---python---"; python -c "import chess; print(chess.__version__)"'
        ),
        content="---\n---python---\n1.11.2\n",
        status="success",
    )

    bench_checker_guard.update_from_tool_turn(
        state,
        tool_calls=tool_calls,
        result_messages=result_messages,
    )

    assert state.bench_visible_checker_failed is False
    assert state.bench_visible_checker_pass_iteration is None


def test_grep_regex_for_test_output_name_is_not_checker_pass(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_CHECKER_FAILURE_GUARD", "true")
    state = _state()
    tool_calls, result_messages = _shell_turn(
        (
            'python3 -c "import chess; print(chess.__version__)"; '
            'grep -rl "immortal_game\\|test_outputs\\|Game of the Century" / '
            '2>/dev/null | grep -v "/proc" | head'
        ),
        content=(
            "Command exceeded the assistant-mode blocking budget (15s) and was "
            "moved to the background with ID: abc123."
        ),
        status="success",
    )

    bench_checker_guard.update_from_tool_turn(
        state,
        tool_calls=tool_calls,
        result_messages=result_messages,
    )

    assert state.bench_visible_checker_failed is False
    assert state.bench_visible_checker_pass_iteration is None


def test_executed_checker_failure_is_tracked(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_CHECKER_FAILURE_GUARD", "true")
    state = _state()
    tool_calls, result_messages = _shell_turn(
        "python /app/check.py",
        content="Command failed with exit code 1\nMatch: False",
        status="error",
    )

    bench_checker_guard.update_from_tool_turn(
        state,
        tool_calls=tool_calls,
        result_messages=result_messages,
    )

    assert state.bench_visible_checker_failed is True
    assert state.bench_visible_checker_failure_command == "python /app/check.py"


def test_checker_pipeline_success_is_tracked(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_CHECKER_FAILURE_GUARD", "true")
    state = _state()
    tool_calls, result_messages = _shell_turn(
        "timeout 120 python3 check.py 2>&1 | tail -20",
        content="20 passed",
        status="success",
    )

    bench_checker_guard.update_from_tool_turn(
        state,
        tool_calls=tool_calls,
        result_messages=result_messages,
    )

    assert state.bench_visible_checker_failed is False
    assert state.bench_visible_checker_pass_iteration == 1


def test_read_only_followup_does_not_stale_checker_pass(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_CHECKER_FAILURE_GUARD", "true")
    state = _state()
    state.bench_visible_checker_pass_iteration = 2
    state.current_iteration = 3
    tool_calls, result_messages = _shell_turn(
        'find / -name "test_outputs.py" 2>/dev/null',
        content="/tests/test_outputs.py",
        status="success",
    )

    bench_checker_guard.update_from_tool_turn(
        state,
        tool_calls=tool_calls,
        result_messages=result_messages,
    )

    assert state.bench_visible_checker_pass_iteration == 2


def test_mutating_followup_stales_checker_pass(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_CHECKER_FAILURE_GUARD", "true")
    state = _state()
    state.bench_visible_checker_pass_iteration = 2
    state.current_iteration = 3
    tool_calls, result_messages = _shell_turn(
        "cat > /app/re.json <<'EOF'\n[]\nEOF",
        content="",
        status="success",
    )

    bench_checker_guard.update_from_tool_turn(
        state,
        tool_calls=tool_calls,
        result_messages=result_messages,
    )

    assert state.bench_visible_checker_pass_iteration is None


def test_failed_debug_followup_stales_checker_pass(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_CHECKER_FAILURE_GUARD", "true")
    state = _state()
    state.bench_visible_checker_pass_iteration = 2
    state.current_iteration = 3
    tool_calls, result_messages = _shell_turn(
        "python3 dbg.py",
        content="Traceback (most recent call last):\nAssertionError: still wrong",
        status="error",
    )

    bench_checker_guard.update_from_tool_turn(
        state,
        tool_calls=tool_calls,
        result_messages=result_messages,
    )

    assert state.bench_visible_checker_pass_iteration is None


def test_later_same_checker_success_clears_failure(monkeypatch):
    monkeypatch.setenv("OPENSPACE_BENCH_CHECKER_FAILURE_GUARD", "true")
    state = _state()
    failed_calls, failed_messages = _shell_turn(
        "python /app/check.py",
        content="Command failed with exit code 1\nMatch: False",
        status="error",
    )
    bench_checker_guard.update_from_tool_turn(
        state,
        tool_calls=failed_calls,
        result_messages=failed_messages,
    )

    state.current_iteration = 2
    passed_calls, passed_messages = _shell_turn(
        "python /app/check.py",
        content="",
        status="success",
    )
    bench_checker_guard.update_from_tool_turn(
        state,
        tool_calls=passed_calls,
        result_messages=passed_messages,
    )

    assert state.bench_visible_checker_failed is False
    assert state.bench_visible_checker_pass_iteration == 2
