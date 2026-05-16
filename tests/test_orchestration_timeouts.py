import asyncio
import platform
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from openspace.grounding.backends.shell.session import RunShellTool
from openspace.grounding.backends.shell.transport.local_connector import (
    LocalShellConnector,
)
from openspace.prompts.grounding_agent_prompts import GroundingAgentPrompts
from openspace.tool_layer import OpenSpace, OpenSpaceConfig


@pytest.mark.asyncio
@pytest.mark.skipif(platform.system() == "Windows", reason="POSIX process-group check")
async def test_run_subprocess_timeout_kills_child_process(tmp_path):
    marker = tmp_path / "child-survived"
    script = tmp_path / "spawn_child.py"
    child_code = (
        "import pathlib, time; "
        "time.sleep(1.0); "
        f"pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    parent_code = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "time.sleep(30)"
    )
    script.write_text(parent_code)

    connector = LocalShellConnector()
    result = await connector._run_subprocess(
        [sys.executable, str(script)],
        timeout=0.2,
    )

    await asyncio.sleep(1.4)

    assert result["returncode"] == -1
    assert "timed out" in result["content"]
    assert "(killed)" in result["content"]
    assert not marker.exists()


@pytest.mark.asyncio
async def test_run_shell_command_timeout_reports_cleanup():
    connector = LocalShellConnector()
    code = "import time; time.sleep(10)"
    if platform.system() == "Windows":
        command = subprocess.list2cmdline([sys.executable, "-c", code])
    else:
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

    result = await connector._run_shell_command(command, timeout=0.2)

    assert result["returncode"] == -1
    assert "timed out" in result["content"]
    assert "(killed)" in result["content"] or "(kill attempted)" in result["content"]


@pytest.mark.asyncio
async def test_run_shell_timeout_cap_allows_longer_work():
    class FakeConnector:
        def __init__(self):
            self.timeouts = []

        async def run_bash_script(self, command, *, timeout):
            self.timeouts.append(timeout)
            return {"returncode": 0, "content": "ok", "error": ""}

    connector = FakeConnector()
    tool = RunShellTool(SimpleNamespace(connector=connector))

    await tool._arun("echo ok", timeout=180)
    await tool._arun("echo ok", timeout=9999)

    assert connector.timeouts == [180, 1800]


def test_grounding_prompt_warns_against_timeout_retry():
    prompt = GroundingAgentPrompts.build_system_prompt(["shell"])

    assert "# On Tool Timeouts" in prompt
    assert "Do NOT call the same tool with the same arguments twice" in prompt
    assert "UNKNOWN" in prompt
    assert "paused-pending-investigation" in prompt


@pytest.mark.asyncio
async def test_execute_bounds_post_task_analysis():
    space = OpenSpace(
        OpenSpaceConfig(
            enable_recording=False,
            execution_analysis_timeout=0.01,
        )
    )
    space._initialized = True
    space._grounding_agent = SimpleNamespace(
        process=lambda context: asyncio.sleep(
            0,
            result={
                "status": "success",
                "response": "done",
                "iterations": 1,
                "tool_executions": [],
            },
        ),
        _last_tools=[],
    )

    async def never_returns(*args, **kwargs):
        await asyncio.sleep(3600)

    space._maybe_analyze_execution = never_returns

    result = await space.execute("do a tiny task")

    assert result["status"] == "success"
    assert result["analysis_timed_out"] is True
    assert space.is_running() is False
