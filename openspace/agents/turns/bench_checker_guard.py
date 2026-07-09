"""Terminal-Bench visible-checker failure tracking."""

from __future__ import annotations

import json
import os
import re
from typing import Any


_CHECKER_COMMAND_RE = re.compile(
    r"(?ix)"
    r"("
    r"(?:^|[;&|]\s*)\s*pytest\b|"
    r"\bpython(?:3)?\s+-m\s+pytest\b|"
    r"\b(?:uv|poetry|pipenv)\s+run\s+pytest\b|"
    r"\bpython(?:3)?\s+[^;&|]*\b(?:check|test|tests|test_outputs)\.py\b|"
    r"\b(?:bash|sh)\s+[^;&|]*\b(?:check|test|tests|run[-_]?tests|test[-_]?outputs)\.sh\b|"
    r"(?:^|[;&|]\s*)\s*(?:\.?/|/)?[^;&|\s]*\b(?:check|test|tests|test_outputs)\.py\b|"
    r"(?:^|[;&|]\s*)\s*(?:\.?/|/)?[^;&|\s]*\b(?:check|test|tests|run[-_]?tests|test[-_]?outputs)\.sh\b|"
    r"\b(?:npm|pnpm|yarn)\s+(?:run\s+)?test\b|"
    r"\bcargo\s+test\b|"
    r"\bgo\s+test\b|"
    r"\bmake\s+(?:test|check)\b|"
    r"\b(?:run[-_]?tests|test[-_]?outputs)\b"
    r")"
)

_CHECKER_OUTPUT_RE = re.compile(
    r"(?is)"
    r"("
    r"test session starts|"
    r"collected\s+\d+\s+items|"
    r"=+\s*(?:failures|errors)\s*=+|"
    r"\b(?:FAILED|ERROR)\s+(?:tests?/|[^:\s]+::)|"
    r"\b\d+\s+passed\b|"
    r"\b\d+\s+failed\b|"
    r"\bMatch:\s*(?:True|False)\b"
    r")"
)

_INSPECTION_COMMAND_RE = re.compile(
    r"(?ix)"
    r"^\s*"
    r"(?:"
    r"read(?:[-_]?file)?|view(?:[-_]?file)?|list(?:[-_]?files)?|"
    r"glob|search|"
    r"ls|find|rg|grep|cat|sed|awk|head|tail|pwd|tree|stat|file|wc|du|"
    r"echo|printf"
    r")\b"
)

_FAILURE_RE = re.compile(
    r"(?is)"
    r"("
    r"=+\s*failures\s*=+|"
    r"\bFAIL(?:ED|URES?)?\b|"
    r"\b\d+\s+failed\b|"
    r"\bAssertionError\b|"
    r"\bTraceback\b|"
    r"\bModuleNotFoundError\b|"
    r"\bImportError\b|"
    r"\bCommand failed\b|"
    r"\bCheck failed\b|"
    r"\bnot found in\b|"
    r"\bMatch:\s*False\b|"
    r"\bexpected\b.*\bactual\b"
    r")"
)

_PASS_RE = re.compile(
    r"(?is)"
    r"("
    r"\b\d+\s+passed\b|"
    r"\bOK\b|"
    r"\ball tests passed\b|"
    r"\btests?\s+passed\b|"
    r"\bMatch:\s*True\b"
    r")"
)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def enabled() -> bool:
    return env_bool("OPENSPACE_BENCH_CHECKER_FAILURE_GUARD", False)


def nudge_limit() -> int:
    raw = os.environ.get("OPENSPACE_BENCH_CHECKER_FAILURE_MAX_NUDGES", "2")
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


def should_block_final(state: Any) -> bool:
    return enabled() and bool(getattr(state, "bench_visible_checker_failed", False))


def summarize_failure(state: Any) -> str:
    command = str(getattr(state, "bench_visible_checker_failure_command", "") or "")
    excerpt = str(getattr(state, "bench_visible_checker_failure_excerpt", "") or "")
    parts: list[str] = []
    if command:
        parts.append(f"last failing checker command: {command[:240]}")
    if excerpt:
        parts.append(f"failure excerpt: {excerpt[:600]}")
    return "\n".join(parts)


def build_nudge_message(state: Any) -> dict[str, Any]:
    details = summarize_failure(state)
    detail_suffix = f"\n\n{details}" if details else ""
    return {
        "role": "user",
        "content": (
            "A visible Terminal-Bench checker or test command failed earlier, "
            "and there has not been a later passing checker run. Do not finish "
            "in chat. Fix the artifact under /app, then rerun the visible "
            "checker or a close equivalent until it passes."
            f"{detail_suffix}"
        ),
        "_meta": {
            "type": "bench_checker_failure_nudge",
            "is_meta": True,
        },
    }


def update_from_tool_turn(
    state: Any,
    *,
    tool_calls: list[dict[str, Any]],
    result_messages: list[dict[str, Any]],
) -> None:
    if not enabled():
        return

    result_by_id: dict[str, dict[str, Any]] = {}
    for message in result_messages:
        if message.get("role") != "tool":
            continue
        tool_call_id = str(message.get("tool_call_id") or "")
        if tool_call_id:
            result_by_id[tool_call_id] = message

    for tool_call in tool_calls:
        tool_call_id = str(tool_call.get("id") or "")
        result_message = result_by_id.get(tool_call_id)
        if result_message is None:
            continue
        command = _tool_call_command(tool_call)
        content = _message_text(result_message)
        status = _message_status(result_message)
        verdict = _classify_checker_result(
            command=command,
            content=content,
            status=status,
        )
        if verdict is None and _looks_like_recheck_after_failure(
            state,
            command=command,
            content=content,
            status=status,
        ):
            verdict = "passed"
        if verdict == "failed":
            state.bench_visible_checker_failed = True
            state.bench_visible_checker_failure_iteration = state.current_iteration
            state.bench_visible_checker_failure_command = command
            state.bench_visible_checker_failure_excerpt = _failure_excerpt(content)
            state.bench_visible_checker_pass_iteration = None
        elif verdict == "passed":
            state.bench_visible_checker_failed = False
            state.bench_visible_checker_pass_iteration = state.current_iteration


def _message_status(message: dict[str, Any]) -> str:
    meta = message.get("_meta")
    if not isinstance(meta, dict):
        return ""
    return str(meta.get("status") or "").lower()


def _message_text(message: dict[str, Any]) -> str:
    return _stringify_content(message.get("content"))


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text is not None:
                    parts.append(str(text))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _tool_call_command(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return ""
    name = str(function.get("name") or "")
    arguments = function.get("arguments")
    data: Any = arguments
    if isinstance(arguments, str):
        try:
            data = json.loads(arguments)
        except json.JSONDecodeError:
            data = {"raw": arguments}
    if not isinstance(data, dict):
        data = {}
    command = data.get("command") or data.get("cmd") or data.get("script") or ""
    if command:
        return str(command)
    path = data.get("file_path") or data.get("path") or ""
    if path:
        return f"{name} {path}"
    return name


def _classify_checker_result(
    *,
    command: str,
    content: str,
    status: str,
) -> str | None:
    is_checker = _looks_like_checker(command, content)
    if not is_checker:
        return None
    if status in {"error", "failed", "denied", "cancelled"}:
        return "failed"
    if _FAILURE_RE.search(content):
        return "failed"
    if status == "success":
        return "passed"
    if _PASS_RE.search(content):
        return "passed"
    # Many Terminal-Bench visible checks are shell pipelines, grep commands, or
    # custom scripts that exit 0 without printing a pytest-style "passed"/"OK".
    # Once a checker-like command runs without a failure signal, clear any
    # previous checker failure instead of letting a stale failure lock the run.
    return "passed"


def _looks_like_recheck_after_failure(
    state: Any,
    *,
    command: str,
    content: str,
    status: str,
) -> bool:
    if not bool(getattr(state, "bench_visible_checker_failed", False)):
        return False
    if status in {"error", "failed", "denied", "cancelled"}:
        return False
    if _FAILURE_RE.search(content):
        return False
    failure_command = str(
        getattr(state, "bench_visible_checker_failure_command", "") or ""
    )
    if _same_checker_family(failure_command, command):
        return True
    return bool(_PASS_RE.search(content) and _looks_like_checker(command, content))


def _checker_family(command: str) -> str | None:
    lowered = command.lower()
    if "pytest" in lowered:
        return "pytest"
    if "test_outputs.py" in lowered:
        return "test_outputs.py"
    if re.search(r"\bcheck\.py\b", lowered):
        return "check.py"
    if re.search(r"\btest\.py\b", lowered):
        return "test.py"
    if re.search(r"\b(?:npm|pnpm|yarn)\s+(?:run\s+)?test\b", lowered):
        return "js-test"
    if "cargo test" in lowered:
        return "cargo-test"
    if "go test" in lowered:
        return "go-test"
    if re.search(r"\bmake\s+(?:test|check)\b", lowered):
        return "make-test"
    return None


def _same_checker_family(previous_command: str, command: str) -> bool:
    previous = _checker_family(previous_command)
    current = _checker_family(command)
    if previous is None or current is None:
        return False
    if previous == "pytest" and current in {"pytest", "test_outputs.py"}:
        return True
    if previous == "test_outputs.py" and current in {"pytest", "test_outputs.py"}:
        return True
    return previous == current


def _looks_like_checker(command: str, content: str) -> bool:
    if _has_checker_execution_command(command):
        return True
    if _looks_like_inspection_only(command):
        return False
    return bool(_CHECKER_OUTPUT_RE.search(content))


def _command_segments(command: str) -> list[str]:
    return [
        segment.strip()
        for segment in re.split(r"\s*(?:&&|\|\||;)\s*", command)
        if segment.strip()
    ]


def _has_checker_execution_command(command: str) -> bool:
    for segment in _command_segments(command):
        if _CHECKER_COMMAND_RE.search(segment) and not _INSPECTION_COMMAND_RE.search(
            segment
        ):
            return True
    return False


def _looks_like_inspection_only(command: str) -> bool:
    segments = _command_segments(command)
    if not segments:
        return False
    saw_segment = False
    for segment in segments:
        saw_segment = True
        if not _INSPECTION_COMMAND_RE.search(segment):
            return False
    return saw_segment


def _failure_excerpt(content: str) -> str:
    text = " ".join(content.split())
    if not text:
        return ""
    match = _FAILURE_RE.search(text)
    if not match:
        return text[:600]
    start = max(0, match.start() - 160)
    end = min(len(text), match.end() + 440)
    return text[start:end]


__all__ = [
    "build_nudge_message",
    "enabled",
    "env_bool",
    "nudge_limit",
    "should_block_final",
    "summarize_failure",
    "update_from_tool_turn",
]
