"""Known-answer day run: InterCode CTF (inspect_evals a68dade) in one pinned image, and a 24-hour clock canary."""
import asyncio, datetime, json, logging, os, pathlib, tempfile, time

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.solver import solver
from inspect_ai.util import SandboxEnvironmentSpec, sandbox
from inspect_evals.gdm_intercode_ctf.gdm_intercode_ctf import gdm_intercode_ctf

# Upstream compose (build: ., init: true, tail -f, network_mode: none) with build: replaced by the pushed image and
# Hawk's default sandbox size written out, so local Docker enforces the same 500m / 2Gi. Hawk drops `init` itself
# (_IGNORED_SERVICE_KEYS, hawk/hawk/runner/run_eval_set.py:137).
_COMPOSE = """\
services:
  default:
    image: {image}
    init: true
    command: tail -f /dev/null
    network_mode: none
    cpus: 0.5
    mem_limit: 2g
"""
_log = logging.getLogger("ka_clock")


def _spec(image_digest: str) -> SandboxEnvironmentSpec:
    # Hawk's runner sets DOCKER_IMAGE_REPO to the stack's task repository (hawk/hawk/api/run.py:723,
    # infra/hawk/api.py:1017); the local reference sets it by hand. No account ID lives in this repo.
    repo = os.environ["DOCKER_IMAGE_REPO"]
    compose = pathlib.Path(tempfile.mkdtemp(prefix="ka-day1-")) / "compose.yaml"
    compose.write_text(_COMPOSE.format(image=f"{repo}@{image_digest}"))
    return SandboxEnvironmentSpec("docker", str(compose))


@task
def intercode_pinned(image_digest: str) -> Task:
    return gdm_intercode_ctf(sandbox_config=_spec(image_digest))


@solver
def _clock_loop(calls: int, every_s: int):
    async def solve(state, generate):
        model, rows, t0 = get_model(), [], time.monotonic()
        for i in range(calls):
            row = {"i": i, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
            try:
                out = await model.generate(f"Reply with exactly: OK {i}")
                row["model_ok"] = f"OK {i}" in (out.completion or "")
            except Exception as e:  # a failed call is recorded and the loop goes on
                row["model_ok"], row["model_err"] = False, f"{type(e).__name__}: {str(e)[:200]}"
            try:
                r = await sandbox().exec(["sh", "-c", "date -u +%FT%TZ"], timeout=60)
                row["sbx_ok"] = r.success
            except Exception as e:
                row["sbx_ok"], row["sbx_err"] = False, f"{type(e).__name__}: {str(e)[:200]}"
            rows.append(row)
            _log.warning("ka_clock %s", json.dumps(row))       # one line per call in the runner log
            state.store.set("clock_rows", rows)
            await asyncio.sleep(max(0.0, t0 + (i + 1) * every_s - time.monotonic()))
        return state
    return solve


@scorer(metrics=[mean()])
def _clock_score():
    async def score(state, target):
        rows = state.store.get("clock_rows", [])
        ok = sum(1 for r in rows if r.get("model_ok") and r.get("sbx_ok"))
        return Score(value=ok / max(1, len(rows)), answer=f"{ok}/{len(rows)}",
                     metadata={"failed": [r for r in rows if not (r.get("model_ok") and r.get("sbx_ok"))]})
    return score


@task
def clock_canary(image_digest: str, calls: int = 97, every_s: int = 900) -> Task:
    """One sample that calls the model and its sandbox every `every_s` seconds, `calls` times (97 x 15 min = 24 h 15 min)."""
    return Task(dataset=[Sample(id="clock", input="clock canary", target=str(calls))],
                solver=_clock_loop(calls, every_s), scorer=_clock_score(), sandbox=_spec(image_digest))
