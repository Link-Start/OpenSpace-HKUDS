# Terminal-Bench / Harbor

This directory contains OpenSpace adapters for Terminal-Bench 2.1 via Harbor.

## Setup

Use the `openspace` conda environment:

```bash
source /home/anaconda3/etc/profile.d/conda.sh
conda activate openspace
```

Load the project `.env` with `python-dotenv` rather than `source`; the file may use
dotenv formatting that is not valid shell syntax.

The runner does this automatically.

## Provider Selection

Use an explicit provider prefix in `OPENSPACE_MODEL` or `--model`:

```bash
# OpenRouter
OPENSPACE_MODEL=openrouter/deepseek/deepseek-v4-pro
OPENROUTER_API_KEY=sk-or-v1-...

# DeepSeek official API
OPENSPACE_MODEL=deepseek/deepseek-v4-pro
DEEPSEEK_API_KEY=sk-...
OPENSPACE_LLM_API_BASE=https://api.deepseek.com
```

The runner also accepts `dpsk/...` as a shorthand for `deepseek/...` and
`or/...` as a shorthand for `openrouter/...`. Legacy two-part model names that
do not match a known direct provider, such as `qwen/...`, are still routed via
OpenRouter for compatibility.

The Terminal-Bench adapters pin provider-default API bases for direct providers
such as OpenRouter and DeepSeek, so a stale generic `OPENSPACE_LLM_API_BASE` in
your project `.env` does not accidentally send an `openrouter/...` model to the
DeepSeek endpoint. To intentionally use a custom proxy/base URL, pass it
explicitly with `--agent-env OPENSPACE_LLM_API_BASE=...` or the adapter
`base_url` kwarg.

Credential resolution is provider-aware for benchmark runs. For
`openrouter/...` models, `OPENROUTER_API_KEY` / `OR_API_KEY` or a host OpenRouter
config is preferred over a generic `OPENSPACE_LLM_API_KEY`; this prevents a
generic DeepSeek key in `.env` from being reused against OpenRouter. Passing
`api_key=...` or `--agent-env OPENSPACE_LLM_API_KEY=...` remains an explicit
override.

## Run

Smoke test one task with one OpenSpace iteration:

```bash
python -m benchmarks.terminal_bench --sample smoke
```

Run the default 20-task stratified sample:

```bash
python -m benchmarks.terminal_bench --sample sample20
```

Run a more robust 20-task pass for slower OpenRouter models:

```bash
python -m benchmarks.terminal_bench \
  --sample sample20 \
  --model openrouter/deepseek/deepseek-v4-pro \
  --max-iterations 200 \
  --run-timeout-sec 1800 \
  --llm-timeout-sec 300 \
  --workers 4 \
  --agent-timeout-multiplier 3 \
  --agent-setup-timeout-multiplier 11 \
  --agent-kwarg llm_max_tokens=8192 \
  --agent-kwarg llm_max_retries=2 \
  --agent-kwarg backend_scope=shell \
  --agent-kwarg bench_finalize_stop_after_iterations=6 \
  --agent-kwarg bench_finalize_stop_after_sec=300 \
  --agent-kwarg evolution_final_drain_limit=2 \
  --agent-kwarg evolution_final_drain_timeout_s=180 \
  --harbor-max-retries 1 \
  --retry-include AgentSetupTimeoutError \
  --retry-include AgentTimeoutError \
  --retry-include RuntimeError
```

Run the paid OpenRouter Hy3 model. The runner accepts `tencent/hy3` and
normalizes it to `openrouter/tencent/hy3`; use a request delay because this
model can return 429s under parallel benchmark traffic:

```bash
python -m benchmarks.terminal_bench \
  --sample sample20 \
  --model tencent/hy3 \
  --max-iterations 200 \
  --run-timeout-sec 1800 \
  --llm-timeout-sec 300 \
  --workers 1 \
  --agent-timeout-multiplier 3 \
  --agent-setup-timeout-multiplier 11 \
  --agent-kwarg llm_max_tokens=16384 \
  --agent-kwarg llm_max_retries=4 \
  --agent-kwarg llm_rate_limit_delay=20 \
  --agent-kwarg backend_scope=shell \
  --agent-kwarg openrouter_reasoning_effort=high \
  --agent-kwarg evolution_final_drain_limit=4 \
  --agent-kwarg evolution_final_drain_timeout_s=300 \
  --agent-kwarg evolution_allow_single_observation_capture=true \
  --agent-kwarg bench_checker_failure_guard=false \
  --harbor-max-retries 1 \
  --retry-include AgentSetupTimeoutError \
  --retry-include AgentTimeoutError \
  --retry-include RuntimeError
```

`--run-timeout-sec` controls the OpenSpace process execution inside the task
container. Harbor also applies the Terminal-Bench task's own agent-phase timeout,
so use `--agent-timeout-multiplier` for long runs. When `--run-timeout-sec` is
set, the adapter applies the timeout inside the task container with a short kill
grace period before Harbor's outer timeout fires, which preserves
`openspace-stdout.txt` and `openspace-stderr.txt` for timeout debugging.
The wrapper also passes an agent setup timeout multiplier derived from
`install_timeout_sec` by default, because OpenSpace setup can otherwise be killed
by Harbor's outer setup timeout before the adapter's own install timeout fires.
When evolution is enabled, the Harbor adapter defaults `post_execution_timeout_s`
to `evolution_final_drain_timeout_s + 60` seconds so benchmark verification is
not held up for many extra minutes after the task artifact is already produced.
Override it explicitly with `--agent-kwarg post_execution_timeout_s=...`; `0`
disables the inline timeout and is useful for diagnostic runs where complete
post-task skill evolution matters more than benchmark throughput.
Terminal-Bench does not provide an OpenSpace replay runner by default, so the
adapter sets `evolution_behavior_eval_require_replay_runner=false`; deterministic
behavior checks still run, but replay-backed changes are not blocked solely
because no external runner was configured. The adapter also sets
`evolution_routing_eval_enabled=false` by default to avoid several extra
post-task LLM selector calls per generated skill inside the benchmark container.
Use `--agent-kwarg evolution_routing_eval_enabled=true` when you want the full
routing behavior check during a diagnostic run.

The wrapper and adapter default to a lean task-solving tool surface for
Terminal-Bench: `backend_scope=shell`, `skills_disabled=true`,
`memory_mode=direct`, `policy_deferred_tool_names=[]`, and
`active_tool_names=write,read,edit,grep,glob,ls,bash`. This keeps the agent
focused on inspecting `/app`, writing requested artifacts, and running concrete
checks without exposing `tool_search` or multi-agent/meta tools by default.
Evolution, evidence capture, and quality-signal analysis remain enabled as
runtime/background paths. The adapter writes CAPTURED skills to
`/installed-agent/evolved-skills` and downloads that directory with each trial's
recordings/evidence DB. To A/B test the full OpenSpace interactive surface:

```bash
--agent-kwarg backend_scope=shell,meta \
--agent-kwarg active_tool_names=none \
--agent-kwarg skills_disabled=false
```

Terminal-Bench uses short-lived task containers, so the adapter enables a small
final evolution drain by default. This gives retryable post-execution evolution
jobs one more chance to author, validate, and commit a skill before the
container exits.

The adapter also treats OpenSpace internal non-success states such as
`MODEL_ERROR` and `INCOMPLETE` as Harbor agent errors by default
(`strict_internal_status=true`). This lets Harbor retry model timeouts instead
of running the verifier against an unfinished workspace and recording a plain
zero score. Disable only for debugging with:

```bash
--agent-kwarg strict_internal_status=false
```

Terminal-Bench runs also set OpenSpace max-output recovery to one retry by
default (`max_output_recovery_limit=1`). This avoids spending many minutes on
repeated truncated non-tool responses during benchmark tasks. Restore the
interactive default when debugging with:

```bash
--agent-kwarg max_output_recovery_limit=3
```

For OpenRouter models, the adapter does not lower reasoning effort on ordinary
task-solving turns. Hard Terminal-Bench tasks often need the model's full
provider-default reasoning behavior. The adapter only excludes returned
reasoning text by default (`openrouter_reasoning_exclude=true`) so benchmark
traces and response budgets focus on actionable assistant text and tool calls.
Tune it explicitly only when needed:

```bash
--agent-kwarg openrouter_reasoning_effort=high
```

or, for providers that support a separate reasoning-token cap:

```bash
--agent-kwarg openrouter_reasoning_max_tokens=16384
```

To include returned reasoning text in traces, disable exclusion explicitly:

```bash
--agent-kwarg openrouter_reasoning_exclude=false
```

`openrouter_reasoning_effort` and `openrouter_reasoning_max_tokens` are explicit
overrides; omit them to use the provider/model default.

When OpenSpace is recovering from a no-tool max-output response, Terminal-Bench
runs force the next call to use a tool. The adapter also defaults
`disable_reasoning_on_required_tool_choice=true` for those required-tool recovery
calls only. This keeps normal reasoning turns unchanged, but avoids providers
spending the whole recovery response budget in hidden reasoning before emitting a
tool call. Override it when debugging:

```bash
--agent-kwarg disable_reasoning_on_required_tool_choice=false
```

For the official DeepSeek API, V4 models default to thinking mode. OpenSpace
does not disable that for normal benchmark turns. However, DeepSeek V4 thinking
mode rejects API-level `tool_choice=required`, which OpenSpace uses only as a
max-output recovery mechanism after a truncated no-tool response. For
Terminal-Bench runs with `deepseek/...`, the adapter also sets:

```bash
OPENSPACE_DEEPSEEK_DISABLE_THINKING_ON_REQUIRED_TOOL_CHOICE=true
```

This keeps ordinary reasoning turns unchanged, but temporarily sends
`thinking={"type":"disabled"}` on required-tool recovery calls so the recovery
can force the model back into shell/file action. Override it when debugging:

```bash
--agent-env OPENSPACE_DEEPSEEK_DISABLE_THINKING_ON_REQUIRED_TOOL_CHOICE=false
```

You can also control official DeepSeek thinking explicitly for the whole run:

```bash
--agent-env OPENSPACE_DEEPSEEK_THINKING=enabled
--agent-env OPENSPACE_DEEPSEEK_REASONING_EFFORT=high
```

The adapter additionally enables a Terminal-Bench finalization nudge by default.
When a long run reaches either 1200 seconds or 24 model iterations, OpenSpace
asks the agent to preserve its best current answer in the requested `/app`
artifact path, run at most one quick check, and stop open-ended exploration.
After that nudge, the adapter stops the run after 6 iterations or 300 seconds
by default, even if the model keeps exploring with more tools. This gives the
external verifier a chance to score the best artifact instead of waiting for
open-ended self-checking to time out. If a visible checker has already passed,
the adapter also stops after 2 additional iterations by default. Tune or disable
these guards with:

```bash
--agent-kwarg bench_finalize_nudge_after_sec=1500 \
--agent-kwarg bench_finalize_nudge_after_iteration=40 \
--agent-kwarg bench_finalize_stop_after_iterations=0 \
--agent-kwarg bench_finalize_stop_after_sec=0 \
--agent-kwarg bench_stop_after_checker_pass_iterations=0 \
--agent-kwarg bench_finalize_nudge_enabled=false
```

The finalization guard is structural rather than phrase-based: after the
checkpoint, a no-tool final response is not accepted until the agent performs a
later shell/file action. OpenSpace first injects tool-backed finalization
nudges, then switches to the configured LLM fallback model when one is
available; if neither path produces tool-backed progress, the run stops as
`bench_no_tool_final_unresolved` instead of being marked completed.

The adapter also enables a visible-checker failure guard
(`OPENSPACE_BENCH_CHECKER_FAILURE_GUARD=true`). If a checker-like command such
as `python check.py` or `pytest` visibly fails, OpenSpace will not accept a
no-tool final answer until the agent fixes the artifact and runs a later
passing checker. After the configured nudge budget is exhausted, the run ends
with `bench_visible_checker_failed` instead of being reported as a successful
OpenSpace completion. For debugging only, override it with:

```bash
--agent-env OPENSPACE_BENCH_CHECKER_FAILURE_GUARD=false
```

Startup retryable recovery is separate and remains disabled by default. Only
enable it when intentionally reusing an evidence DB and wanting persisted
`failed_retryable` jobs to run during startup, for example:

```bash
--agent-kwarg evolution_startup_retryable_drain_limit=4 \
--agent-kwarg evolution_startup_retryable_drain_timeout_s=180
```

Run the full 89-task Terminal-Bench 2.1 suite:

```bash
python -m benchmarks.terminal_bench --sample full --workers 4
```

Run a two-pass replay experiment:

```bash
# Pass 1: cold run. Each task writes its own evidence DB and evolved skills.
python -m benchmarks.terminal_bench \
  --sample full \
  --model tencent/hy3:free \
  --max-iterations 200 \
  --run-timeout-sec 1800 \
  --llm-timeout-sec 300 \
  --workers 4 \
  --job-name tb21_hy3_full_cold

# Pass 2: per-task replay. Each task is seeded from the matching pass-1 trial.
python -m benchmarks.terminal_bench \
  --sample full \
  --model tencent/hy3:free \
  --max-iterations 200 \
  --run-timeout-sec 1800 \
  --llm-timeout-sec 300 \
  --workers 4 \
  --job-name tb21_hy3_full_replay \
  --replay-from-run "$(pwd)/benchmarks/terminal_bench/runs/tb21_hy3_full_cold" \
  --agent-kwarg evolution_recovery_stale_job_timeout_s=30 \
  --agent-kwarg evolution_startup_retryable_drain_limit=4 \
  --agent-kwarg evolution_startup_retryable_drain_timeout_s=180
```

`--replay-from-run` does not merge all 89 task databases into one global memory.
It performs a per-task replay: `regex-chess` pass 2 receives only the pass-1
`regex-chess` evidence DB, runtime `openspace.db`, and `evolved-skills/`;
`fix-git` receives only `fix-git`, and so on. In replay mode the runner enables
skills by default (`skills_disabled=false`) so evolved skills can be discovered
and invoked; pass `--agent-kwarg skills_disabled=true` to override that behavior.
The adapter also injects a short prior-verifier feedback summary directly into
the task prompt, so exact verifier failures and acceptance targets remain
visible even if the model does not proactively load the replay skill.
If the previous run ended shortly before replay, lower
`evolution_recovery_stale_job_timeout_s` so startup recovery can reset
final-drain jobs that were interrupted while still marked `running`.
Replay mode also drains copied `pending` and `failed_retryable` evolution jobs
by default, so evidence created in pass 1 can continue producing skills in pass
2. The adapter sets a finite post-execution timeout so already-complete task
runs are not held open indefinitely by best-effort analysis/evolution work; pass
`--agent-kwarg post_execution_timeout_s=0` when validating skill generation and
you prefer to rely only on the outer task timeout.

Run specific tasks:

```bash
python -m benchmarks.terminal_bench --task fix-git --task raman-fitting
```

Print the selected sample distribution without running Harbor:

```bash
python -m benchmarks.terminal_bench --sample sample20 --list-sample
```

Summarize one or more completed run directories:

```bash
python benchmarks/terminal_bench/analyze_runs.py \
  tb21_hy3_full_cold \
  tb21_hy3_full_replay \
  --show-paths
```

The analyzer reports reward, failure reason, evidence/evolution DB counts,
running trigger jobs, evolved skill artifacts, runtime DB replay, and replay
seed status for each trial.

The smoke command is a harness test. It verifies the Terminal-Bench 2.1
dataset, Docker environment, OpenSpace agent startup, OpenRouter model call, and
Harbor verifier path. It does not aim to solve the task because `max_iterations=1`.

## OpenSpace Runtime Defaults

The adapter keeps OpenSpace's runtime analysis paths enabled by default:

- recording enabled with conversation logs
- evolution evidence, triggers, and engine enabled
- quality signal detector, trigger, and reconciliation enabled
- evolution mode `autonomous`

Screenshots and video stay disabled by default, matching the current OpenSpace
CLI defaults and avoiding a local GUI recording server requirement inside
Terminal-Bench containers. They can be enabled explicitly:

```bash
python -m benchmarks.terminal_bench --task fix-git \
  --agent-kwarg enable_screenshot=true \
  --agent-kwarg enable_video=true
```

Each trial saves runtime artifacts under its own `agent/` directory:

- `recordings/` for OpenSpace trajectory files such as `metadata.json`,
  `traj.jsonl`, `conversations.jsonl`, and `summary.json`
- `openspace-evidence.db*` for evolution/evidence/quality-signal state
- `evolved-skills/` for skills committed by the evolution engine
- `workspace-state/openspace.db*` for workspace/session runtime state
- `openspace-logs/` for OpenSpace runtime log files
- `openspace-stdout.txt` and `openspace-stderr.txt` for CLI output

`conversations.jsonl` includes both normal tool iterations and model-only
iterations such as no-tool `length`/`MAX_OUTPUT_TOKENS` recoveries, empty
responses, stop-hook blocks, and final no-tool responses.

## Default Sample

The default sample is fixed for reproducibility:

- 20 tasks
- 12 categories
- difficulty mix: 1 easy, 14 medium, 5 hard

It is designed as a practical middle ground when the full 89-task suite is too
expensive: it covers the largest categories, includes singleton categories
where useful, mixes short and long timeouts, and includes both implementation
and data/science/security/system tasks.
