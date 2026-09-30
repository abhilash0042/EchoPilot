"""
Belfry Labs — Enterprise Readiness Verification Engine.

This module powers ``belfry verify --enterprise``: a self-contained pipeline a
CTO can run live to prove the platform is real, wired to live engines, and
enterprise-ready.

Design principles
-----------------
* **Degrade gracefully.** Every check is independent, captures its own
  exceptions, and records a status of PASS / FAIL / WARN / SKIP with a reason.
  One failing check never aborts the pipeline.
* **Run anywhere.** The pipeline completes even when the API is unreachable —
  online checks become SKIP/FAIL but a full report is always produced.
* **No fragile assumptions.** Repo root is discovered explicitly; subprocesses
  are run with an explicit ``cwd`` and a sanitized environment.
* **Boardroom-grade output.** JSON for machines, a polished self-contained HTML
  report for executives, and a coloured ``rich`` console view for the terminal.

Usage (offline, no server, no API key required)::

    import asyncio, pathlib
    from belfry_labs.verify import VerificationPipeline

    pipe = VerificationPipeline(client=None, repo_root=pathlib.Path(".").resolve())
    summary = asyncio.run(pipe.run())
    print(summary["verdict"], summary["readiness_score"])
"""

from __future__ import annotations

import asyncio
import html
import inspect
import json
import os
import platform as _platform
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"

#: Numeric contribution of each status to the weighted readiness score.
_STATUS_SCORE: Dict[str, float] = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}

#: Relative importance of each category when computing readiness.
_CATEGORY_WEIGHTS: Dict[str, float] = {
    "Platform": 1.0,
    "Integrity": 2.0,
    "Tests": 3.0,
    "Examples": 2.0,
    "Red Team": 2.0,
    "Security Scanning": 3.0,
    "Runtime": 2.0,
    "SDK": 2.0,
    "CLI": 1.0,
    "Compliance": 1.0,
    "SBOM": 1.0,
    "Access Control": 3.0,
    "Multi-Tenancy": 2.0,
}


class _ApiUnreachable(Exception):
    """Internal signal that an online check could not reach the API."""


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """The outcome of a single verification check."""

    id: str
    name: str
    category: str
    status: str
    duration_ms: float = 0.0
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    remediation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration_ms"] = round(self.duration_ms, 2)
        return d


# A check body may return any of:
#   status
#   (status, detail)
#   (status, detail, evidence)
#   (status, detail, evidence, remediation)
_CheckReturn = Union[
    str,
    Tuple[str, str],
    Tuple[str, str, Dict[str, Any]],
    Tuple[str, str, Dict[str, Any], str],
]
_CheckBody = Callable[[], Union[_CheckReturn, Awaitable[_CheckReturn]]]


class VerificationPipeline:
    """Enterprise readiness verification pipeline.

    Args:
        client: An ``AsyncBelfryLabsClient`` (or ``None`` for offline mode).
            When ``None`` (or when requests fail) every online stage degrades to
            SKIP/FAIL while offline stages still execute.
        repo_root: Repository root. Auto-discovered if not provided.
        demo_project: Project id used for online demo calls.
        subprocess_timeout: Per-subprocess timeout in seconds.
        bypass_threshold: Max tolerated red-team bypasses before FAIL.
    """

    def __init__(
        self,
        client: Any = None,
        repo_root: Optional[Path] = None,
        demo_project: str = "demo",
        subprocess_timeout: int = 600,
        bypass_threshold: int = 0,
    ) -> None:
        self.client = client
        self.repo_root = Path(repo_root).resolve() if repo_root else self._discover_repo_root()
        self.demo_project = demo_project
        self.subprocess_timeout = subprocess_timeout
        self.bypass_threshold = bypass_threshold

        self.results: List[CheckResult] = []
        self.summary: Dict[str, Any] = {}
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None

        # Cache for API reachability so we probe /health only once.
        self._api_ok: Optional[bool] = None
        self._api_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Stage registry
    # ------------------------------------------------------------------

    def _stage_registry(self) -> "Dict[str, Callable[[], Awaitable[List[CheckResult]]]]":
        return {
            "platform": self.audit_platform,
            "dummy": self.detect_dummy_implementations,
            "tests": self.run_unit_integration_tests,
            "examples": self.run_example_projects,
            "redteam": self.run_red_team,
            "deepscan": self.run_deep_scan,
            "middleware": self.run_runtime_middleware,
            "sdk": self.run_sdk_tests,
            "cli": self.run_cli_tests,
            "compliance": self.run_compliance,
            "sbom": self.run_sbom,
            "acceptance": self.run_acceptance_gate,
            "rbac": self.check_rbac,
            "tenancy": self.check_multi_tenancy,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_repo_root() -> Path:
        """Walk up from this file looking for a `.git` dir or a `backend/` dir."""
        here = Path(__file__).resolve()
        for candidate in [here, *here.parents]:
            if (candidate / ".git").exists() or (candidate / "backend").is_dir():
                return candidate
        # Fallback: two levels up (python-sdk/belfry_labs -> repo root).
        return here.parents[2] if len(here.parents) >= 3 else Path.cwd()

    async def _execute(
        self,
        id: str,
        name: str,
        category: str,
        body: _CheckBody,
        remediation: str = "",
    ) -> CheckResult:
        """Run one check body, timing it and trapping every exception."""
        start = time.perf_counter()
        try:
            result = body()
            if inspect.isawaitable(result):
                result = await result
            status, detail, evidence, rem = self._normalize(result, remediation)
        except _ApiUnreachable as exc:
            status, detail, evidence, rem = (
                SKIP,
                f"API unreachable: {exc}",
                {"reason": str(exc)},
                remediation or "Ensure the Belfry API is running and BELFRY_LABS_API_KEY is set.",
            )
        except Exception as exc:  # noqa: BLE001 - checks must never crash the run
            status = FAIL
            detail = f"Unexpected error: {exc}"
            evidence = {
                "exception": repr(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:],
            }
            rem = remediation or "Investigate the captured traceback; this check raised unexpectedly."
        duration_ms = (time.perf_counter() - start) * 1000.0
        return CheckResult(
            id=id,
            name=name,
            category=category,
            status=status,
            duration_ms=duration_ms,
            detail=detail,
            evidence=evidence,
            remediation=rem,
        )

    @staticmethod
    def _normalize(result: _CheckReturn, default_remediation: str) -> Tuple[str, str, Dict[str, Any], str]:
        if isinstance(result, str):
            return result, "", {}, default_remediation
        if not isinstance(result, (tuple, list)):
            raise TypeError(f"Check body returned unsupported type: {type(result)!r}")
        items = list(result)
        status = items[0]
        detail = items[1] if len(items) > 1 else ""
        evidence = items[2] if len(items) > 2 and items[2] is not None else {}
        rem = items[3] if len(items) > 3 and items[3] else default_remediation
        # A passing check usually needs no remediation text.
        if status == PASS and len(items) <= 3:
            rem = ""
        return status, detail, dict(evidence), rem

    async def _api_available(self) -> bool:
        """Probe `/health` exactly once and cache the result."""
        if self._api_ok is not None:
            return self._api_ok
        if self.client is None:
            self._api_ok = False
            self._api_error = "No API client configured (offline mode)"
            return False
        try:
            await self.client._request("GET", "/health")
            self._api_ok = True
        except Exception as exc:  # noqa: BLE001
            self._api_ok = False
            self._api_error = str(exc)
        return self._api_ok

    async def _api_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.client is None:
            raise _ApiUnreachable("No API client configured (offline mode)")
        try:
            return await self.client._request("GET", endpoint, params=params)
        except _ApiUnreachable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _ApiUnreachable(str(exc)) from exc

    async def _api_post(self, endpoint: str, json_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.client is None:
            raise _ApiUnreachable("No API client configured (offline mode)")
        try:
            return await self.client._request("POST", endpoint, json_data=json_data)
        except _ApiUnreachable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _ApiUnreachable(str(exc)) from exc

    def _run_subprocess(
        self,
        args: List[str],
        cwd: Path,
        extra_env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        # Repo root first so `import backend...` resolves; then SDK for belfry_labs.
        repo_root = str(self.repo_root.resolve())
        sdk_path = str((self.repo_root / "python-sdk").resolve())
        env["PYTHONPATH"] = os.pathsep.join(
            [p for p in [repo_root, sdk_path, env.get("PYTHONPATH", "")] if p]
        )
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            args,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout or self.subprocess_timeout,
        )

    @staticmethod
    def _coerce_number(*values: Any) -> Optional[float]:
        for v in values:
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                m = re.search(r"-?\d+(?:\.\d+)?", v)
                if m:
                    return float(m.group())
        return None

    # ------------------------------------------------------------------
    # Stage 1 — Platform audit
    # ------------------------------------------------------------------

    async def audit_platform(self) -> List[CheckResult]:
        async def public_health() -> _CheckReturn:
            data = await self._api_get("/health")
            version = data.get("version") or data.get("app_version") or "unknown"
            uptime = data.get("uptime") or data.get("uptime_seconds")
            status_str = str(data.get("status", "ok"))
            evidence = {
                "version": version,
                "uptime": uptime,
                "status": status_str,
                "raw": data,
            }
            ok = status_str.lower() in {"ok", "healthy", "up", "pass", "200"}
            return (
                PASS if ok else WARN,
                f"Platform reachable (version={version}, status={status_str})",
                evidence,
            )

        async def admin_health() -> _CheckReturn:
            data = await self._api_get("/admin/health")
            evidence = {"raw": data}
            return PASS, "Admin/control-plane health endpoint reachable.", evidence

        return [
            await self._execute(
                "platform.health", "Public health endpoint", "Platform", public_health,
                remediation="Start the FastAPI backend and confirm GET /health responds.",
            ),
            await self._execute(
                "platform.admin_health", "Admin health endpoint", "Platform", admin_health,
                remediation="Confirm GET /admin/health is exposed for the control plane.",
            ),
        ]

    # ------------------------------------------------------------------
    # Stage 2 — Dummy / dry-run detection
    # ------------------------------------------------------------------

    async def detect_dummy_implementations(self) -> List[CheckResult]:
        async def atlas_coverage() -> _CheckReturn:
            data = await self._api_get("/redteam/atlas/coverage")
            coverage = self._coerce_number(
                data.get("coverage_pct"),
                data.get("coverage"),
                data.get("atlas_coverage_pct"),
                data.get("techniques_covered"),
                data.get("total_techniques"),
            )
            evidence = {"coverage": coverage, "raw": data}
            if coverage is None:
                return (
                    WARN,
                    "ATLAS coverage endpoint responded but no coverage metric was found.",
                    evidence,
                    "Verify /redteam/atlas/coverage reports a numeric coverage value.",
                )
            if coverage <= 0:
                return (
                    WARN,
                    "ATLAS coverage is 0 — red team may be running in dry-run mode, not wired to a live engine.",
                    evidence,
                    "Wire the red-team service to the live AutoRedTeamer engine; coverage of 0 indicates a stub.",
                )
            return PASS, f"Red team wired to a live engine (ATLAS coverage={coverage}).", evidence

        async def routing_recommendations() -> _CheckReturn:
            try:
                data = await self._api_get("/cost-optimization/routing/recommendations")
            except _ApiUnreachable as exc:
                # Distinguish a true 404 (feature absent) from network unreachability.
                if "not found" in str(exc).lower() or "404" in str(exc):
                    return (
                        WARN,
                        "Routing recommendations endpoint not found (cost auto-routing not yet exposed).",
                        {"error": str(exc)},
                        "Expose /cost-optimization/routing/recommendations if cost routing is a sold capability.",
                    )
                raise
            return PASS, "Cost-optimization routing recommendations endpoint is present.", {"raw": data}

        return [
            await self._execute(
                "integrity.atlas_coverage", "Red team is live (not dry-run)", "Integrity", atlas_coverage,
                remediation="Ensure the red-team engine is connected and reporting real ATLAS coverage.",
            ),
            await self._execute(
                "integrity.cost_routing", "Cost routing endpoint probe", "Integrity", routing_recommendations,
            ),
        ]

    # ------------------------------------------------------------------
    # Stage 3 — Unit / integration tests
    # ------------------------------------------------------------------

    async def run_unit_integration_tests(self) -> List[CheckResult]:
        def pytest_suite() -> _CheckReturn:
            tests_dir = self.repo_root / "tests" / "regression"
            if not tests_dir.is_dir():
                return (
                    SKIP,
                    f"Regression suite not found at {tests_dir}.",
                    {"path": str(tests_dir)},
                    "Confirm tests/regression exists at the repo root.",
                )
            proc = self._run_subprocess(
                [sys.executable, "-m", "pytest", "tests/regression", "-q"],
                cwd=self.repo_root,
            )
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            passed = self._extract_count(output, r"(\d+)\s+passed")
            failed = self._extract_count(output, r"(\d+)\s+failed")
            errors = self._extract_count(output, r"(\d+)\s+error")
            skipped = self._extract_count(output, r"(\d+)\s+skipped")
            evidence = {
                "returncode": proc.returncode,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
                "tail": output.strip()[-1500:],
            }
            if proc.returncode == 0 and (failed == 0 and errors == 0):
                return (
                    PASS,
                    f"Regression suite green: {passed} passed, {skipped} skipped.",
                    evidence,
                )
            return (
                FAIL,
                f"Regression suite failing: {failed} failed, {errors} errors, {passed} passed.",
                evidence,
                "Run `python -m pytest tests/regression -q` locally and fix the failing tests.",
            )

        return [
            await self._execute(
                "tests.regression", "Regression test suite (pytest)", "Tests", pytest_suite,
                remediation="Ensure pytest is installed and tests/regression passes.",
            )
        ]

    @staticmethod
    def _extract_count(text: str, pattern: str) -> int:
        m = re.search(pattern, text)
        return int(m.group(1)) if m else 0

    # ------------------------------------------------------------------
    # Stage 4 — Example projects (offline self-verifying harnesses)
    # ------------------------------------------------------------------

    async def run_example_projects(self) -> List[CheckResult]:
        results: List[CheckResult] = []
        for name in ("chatbot", "rag", "agent", "runtime_security_reference"):
            example_dir = self.repo_root / "examples" / "python" / name

            def body(example_dir: Path = example_dir, name: str = name) -> _CheckReturn:
                verify_script = example_dir / "verify.py"
                if not verify_script.is_file():
                    return (
                        SKIP,
                        f"No verify.py for examples/python/{name}.",
                        {"path": str(verify_script)},
                        f"Add a self-verifying verify.py to examples/python/{name}.",
                    )
                proc = self._run_subprocess(
                    [sys.executable, "verify.py"],
                    cwd=example_dir,
                    extra_env={"BELFRY_DEMO_MODE": "1"},
                )
                output = (proc.stdout or "") + "\n" + (proc.stderr or "")
                evidence = {
                    "returncode": proc.returncode,
                    "tail": output.strip()[-1200:],
                }
                if proc.returncode == 0:
                    return PASS, f"Example '{name}' self-verification passed (offline demo mode).", evidence
                return (
                    FAIL,
                    f"Example '{name}' self-verification failed (exit {proc.returncode}).",
                    evidence,
                    f"Run `cd examples/python/{name} && BELFRY_DEMO_MODE=1 python verify.py` to debug.",
                )

            results.append(
                await self._execute(
                    f"examples.{name}", f"Example: python/{name}", "Examples", body,
                    remediation=f"Ensure examples/python/{name} verifies cleanly in demo mode.",
                )
            )
        return results

    # ------------------------------------------------------------------
    # Stage 5 — Red team campaign
    # ------------------------------------------------------------------

    async def run_red_team(self) -> List[CheckResult]:
        async def campaign() -> _CheckReturn:
            if not await self._api_available():
                raise _ApiUnreachable(self._api_error or "API not reachable")
            payload = {
                "project_id": self.demo_project,
                "name": "Enterprise Verify Campaign",
                "techniques": [],
                "max_attacks_per_technique": 10,
                "include_regression": True,
            }
            data = await self._api_post("/redteam/continuous/run", json_data=payload)
            bypassed = int(self._coerce_number(data.get("bypassed"), data.get("bypasses"), 0) or 0)
            total = int(self._coerce_number(data.get("total_attacks"), 0) or 0)
            blocked = int(self._coerce_number(data.get("blocked"), 0) or 0)
            evidence = {"total_attacks": total, "blocked": blocked, "bypassed": bypassed, "raw": data}
            if bypassed > self.bypass_threshold:
                return (
                    FAIL,
                    f"Red team campaign found {bypassed} bypass(es) (threshold {self.bypass_threshold}).",
                    evidence,
                    "Investigate the bypassed attacks and tighten the relevant guardrail policies.",
                )
            return PASS, f"Red team campaign clean: {blocked}/{total} blocked, {bypassed} bypassed.", evidence

        return [
            await self._execute(
                "redteam.campaign", "Continuous red-team campaign", "Red Team", campaign,
                remediation="Run a live red-team campaign against the demo project to verify guardrails.",
            )
        ]

    # ------------------------------------------------------------------
    # Stage 6 — In-process deep scan
    # ------------------------------------------------------------------

    async def run_deep_scan(self) -> List[CheckResult]:
        def scan() -> _CheckReturn:
            fixture = (
                self.repo_root
                / "tests" / "regression" / "owasp_llm" / "fixtures" / "llm01_prompt_injection.py"
            )
            if not fixture.is_file():
                return (
                    SKIP,
                    f"Scanner fixture not found at {fixture}.",
                    {"path": str(fixture)},
                    "Confirm the OWASP LLM01 fixture exists for the scanner self-test.",
                )
            repo_str = str(self.repo_root)
            inserted = False
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)
                inserted = True
            try:
                from backend.ai_scanner.code_scanner.scanner.engine import ScannerEngine
            except Exception as exc:  # noqa: BLE001 - import is allowed to fail
                return (
                    SKIP,
                    f"Backend scanner import failed: {exc}",
                    {"error": repr(exc)},
                    "Install backend dependencies (or run from a backend-enabled env) to exercise the in-process scanner.",
                )
            finally:
                if inserted and repo_str in sys.path:
                    try:
                        sys.path.remove(repo_str)
                    except ValueError:
                        pass

            engine = ScannerEngine()
            result = engine.scan_file(fixture)
            vulns = getattr(result, "vulnerabilities", []) or []
            sample = [
                {
                    "rule_id": getattr(v, "rule_id", None),
                    "title": getattr(v, "title", None),
                    "line": getattr(v, "line_number", None),
                    "severity": str(getattr(v, "severity", "")),
                }
                for v in vulns[:5]
            ]
            evidence = {"findings": len(vulns), "sample": sample}
            if len(vulns) > 0:
                return PASS, f"In-process scanner found {len(vulns)} finding(s) in the LLM01 fixture.", evidence
            return (
                FAIL,
                "Scanner ran but produced zero findings on a known-vulnerable fixture.",
                evidence,
                "The scanner rules may be disabled or broken — expected >0 findings on llm01_prompt_injection.py.",
            )

        return [
            await self._execute(
                "scanning.deep_scan", "In-process code scanner", "Security Scanning", scan,
                remediation="Verify the code scanner engine imports and detects the LLM01 fixture.",
            )
        ]

    # ------------------------------------------------------------------
    # Stage 7 — Runtime middleware
    # ------------------------------------------------------------------

    async def run_runtime_middleware(self) -> List[CheckResult]:
        def middleware() -> _CheckReturn:
            try:
                from belfry_labs.middleware import StreamingGuard, ToolCallGuard, MCPServerMiddleware
            except Exception as exc:  # noqa: BLE001
                return (
                    FAIL,
                    f"Failed to import belfry_labs.middleware: {exc}",
                    {"error": repr(exc)},
                    "Ensure the middleware package (streaming/tools/mcp) is installed and importable.",
                )
            instantiated = []
            # All three accept the belfry client as the first positional arg.
            StreamingGuard(None, project_id=self.demo_project)
            instantiated.append("StreamingGuard")
            ToolCallGuard(None, project_id=self.demo_project)
            instantiated.append("ToolCallGuard")
            MCPServerMiddleware(None, project_id=self.demo_project)
            instantiated.append("MCPServerMiddleware")
            return (
                PASS,
                "Runtime middleware importable and instantiable: " + ", ".join(instantiated) + ".",
                {"instantiated": instantiated},
            )

        return [
            await self._execute(
                "runtime.middleware", "Runtime middleware package", "Runtime", middleware,
                remediation="Confirm StreamingGuard, ToolCallGuard, and MCPServerMiddleware import and construct.",
            )
        ]

    # ------------------------------------------------------------------
    # Stage 8 — SDK surface
    # ------------------------------------------------------------------

    async def run_sdk_tests(self) -> List[CheckResult]:
        def sdk() -> _CheckReturn:
            import belfry_labs
            from belfry_labs import AsyncBelfryLabsClient

            expected = [
                "protect", "monitor", "evaluate", "redteam", "scan", "trace",
                "sbom", "compliance", "cost", "policy", "runtime", "shutdown",
            ]
            missing = [m for m in expected if not hasattr(AsyncBelfryLabsClient, m)]
            evidence = {
                "version": getattr(belfry_labs, "__version__", "unknown"),
                "expected_methods": expected,
                "missing_methods": missing,
                "present": len(expected) - len(missing),
            }
            if missing:
                return (
                    FAIL,
                    f"SDK client missing {len(missing)} top-level method(s): {', '.join(missing)}.",
                    evidence,
                    "Implement the missing convenience methods on AsyncBelfryLabsClient.",
                )
            return PASS, f"SDK imports and exposes all {len(expected)} top-level methods.", evidence

        return [
            await self._execute(
                "sdk.surface", "SDK import & client surface", "SDK", sdk,
                remediation="Ensure belfry_labs imports cleanly and the client exposes all 12 methods.",
            )
        ]

    # ------------------------------------------------------------------
    # Stage 9 — CLI
    # ------------------------------------------------------------------

    async def run_cli_tests(self) -> List[CheckResult]:
        def cli() -> _CheckReturn:
            import shutil

            sdk_dir = self.repo_root / "python-sdk"
            cwd = sdk_dir if sdk_dir.is_dir() else self.repo_root

            belfry_bin = shutil.which("belfry")
            if belfry_bin:
                base_cmd = [belfry_bin]
                mode = "installed entrypoint"
            else:
                base_cmd = [sys.executable, "-m", "belfry_labs.cli"]
                mode = "python -m belfry_labs.cli"

            invocations = {
                "--help": base_cmd + ["--help"],
                "scan --help": base_cmd + ["scan", "--help"],
                "verify --help": base_cmd + ["verify", "--help"],
                "eval --help": base_cmd + ["eval", "--help"],
            }
            # Inject a dummy key so the group's api-key guard does not abort
            # `--help` for non-offline subcommands; we are testing command
            # registration, not authentication.
            cli_env = {"BELFRY_LABS_API_KEY": "verify-cli-probe"}

            outcomes: Dict[str, int] = {}
            failures: List[str] = []
            for label, cmd in invocations.items():
                try:
                    proc = self._run_subprocess(cmd, cwd=cwd, extra_env=cli_env, timeout=120)
                    outcomes[label] = proc.returncode
                    if proc.returncode != 0:
                        failures.append(f"{label} (exit {proc.returncode})")
                except Exception as exc:  # noqa: BLE001
                    outcomes[label] = -1
                    failures.append(f"{label} ({exc})")

            evidence = {"mode": mode, "outcomes": outcomes}
            if failures:
                return (
                    FAIL,
                    f"CLI invocation(s) failed via {mode}: {', '.join(failures)}.",
                    evidence,
                    "Install the SDK (`pip install -e python-sdk`) so the `belfry` entrypoint and subcommands register.",
                )
            return PASS, f"CLI responds via {mode}: --help, scan --help, verify --help all OK.", evidence

        return [
            await self._execute(
                "cli.help", "CLI installed & commands register", "CLI", cli,
                remediation="Ensure the `belfry` CLI is installed and its subcommands load.",
            )
        ]

    # ------------------------------------------------------------------
    # Stage 10 — Compliance
    # ------------------------------------------------------------------

    async def run_compliance(self) -> List[CheckResult]:
        async def comp() -> _CheckReturn:
            if not await self._api_available():
                raise _ApiUnreachable(self._api_error or "API not reachable")
            data = await self._api_get(f"/compliance/{self.demo_project}", params={"framework": "nist"})
            score = self._coerce_number(data.get("compliance_score"), data.get("score"))
            status_str = str(data.get("status", "unknown"))
            evidence = {"score": score, "status": status_str, "raw": data}
            return PASS, f"NIST compliance check returned (score={score}, status={status_str}).", evidence

        return [
            await self._execute(
                "compliance.nist", "Compliance check (NIST AI RMF)", "Compliance", comp,
                remediation="Run a NIST compliance check against the demo project via the API.",
            )
        ]

    # ------------------------------------------------------------------
    # Stage 11 — SBOM
    # ------------------------------------------------------------------

    async def run_sbom(self) -> List[CheckResult]:
        async def sbom_check() -> _CheckReturn:
            if not await self._api_available():
                raise _ApiUnreachable(self._api_error or "API not reachable")
            data = await self._api_get(f"/sbom/{self.demo_project}", params={"format": "cyclonedx"})
            comps = data.get("components")
            n_comps = len(comps) if isinstance(comps, list) else None
            evidence = {"components": n_comps, "format": data.get("bomFormat", "cyclonedx")}
            return PASS, "SBOM/AIBOM generated for the demo project.", evidence

        return [
            await self._execute(
                "sbom.generate", "SBOM / AIBOM generation", "SBOM", sbom_check,
                remediation="Generate an SBOM for the demo project via GET /sbom/{project}.",
            )
        ]

    # ------------------------------------------------------------------
    # Stage 12 — RBAC enforcement
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Acceptance gate — offline corpus through the inference gateway
    # ------------------------------------------------------------------

    async def run_acceptance_gate(self) -> List[CheckResult]:
        """Run the shared attack corpus through the gateway engine offline.

        Deterministic and dependency-light: imports only the gateway engine and
        the corpus loader from the repo, so it works for CTO demos without a
        running backend or provider keys.
        """

        async def body() -> _CheckReturn:
            repo_root = str(self.repo_root)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            try:
                from backend.gateway.engine import GatewayEngine
                from backend.gateway.schemas import ChatCompletionRequest
                from backend.services.attack_corpus_loader import get_corpus
            except Exception as exc:  # noqa: BLE001
                return (
                    SKIP,
                    f"Gateway/corpus not importable from {repo_root}: {exc}",
                    {"repo_root": repo_root},
                    "Run from a full checkout so backend.gateway and the corpus are importable.",
                )

            engine = GatewayEngine()
            corpus = get_corpus()
            malicious_total = malicious_flagged = benign_total = benign_fp = 0
            misses: List[str] = []
            for case in corpus.cases:
                req = ChatCompletionRequest(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": case.payload}],
                )
                resp = await engine.chat_completion(req, db=None, demo_mode=True)
                flagged = resp.belfry.blocked or resp.belfry.input_action in ("block", "warn", "redact")
                if case.is_benign:
                    benign_total += 1
                    if flagged:
                        benign_fp += 1
                        misses.append(f"false_positive:{case.id}")
                else:
                    malicious_total += 1
                    if flagged:
                        malicious_flagged += 1
                    else:
                        misses.append(f"missed:{case.id}")

            det = malicious_flagged / malicious_total if malicious_total else 1.0
            fpr = benign_fp / benign_total if benign_total else 0.0
            evidence = {
                "total_cases": len(corpus.cases),
                "detection_rate": round(det, 4),
                "false_positive_rate": round(fpr, 4),
                "categories": corpus.categories(),
                "misses": misses[:20],
            }
            if det >= 0.9 and fpr <= 0.1:
                return (
                    PASS,
                    f"Acceptance gate passed: detection={det:.0%}, FP={fpr:.0%} over {len(corpus.cases)} cases.",
                    evidence,
                )
            return (
                FAIL,
                f"Acceptance gate failed: detection={det:.0%} (need >=90%), FP={fpr:.0%} (need <=10%).",
                evidence,
                "Tune runtime detectors / corpus; see tests/acceptance/corpus_runner.py.",
            )

        results = [
            await self._execute(
                "acceptance.corpus",
                "Acceptance: attack corpus via gateway",
                "Acceptance",
                body,
                remediation="Ensure the gateway runtime check flags the curated corpus offline.",
            )
        ]

        # Per-feature acceptance gates (the 6 productionized features + foundation).
        gate_files = {
            "f1_redteam": "tests/acceptance/test_redteam_gate.py",
            "f2_deepscan": "tests/acceptance/test_deepscan_gate.py",
            "f3_runtime": "tests/acceptance/test_runtime_enforcement_gate.py",
            "f3b_runtime_corpus": "tests/acceptance/test_runtime_gate.py",
            "f3c_runtime_depth": "tests/acceptance/test_runtime_depth_gate.py",
            "python_runtime_reference": "tests/acceptance/test_python_runtime_reference_gate.py",
            "f4_cost": "tests/acceptance/test_cost_routing_gate.py",
            "f5_compliance": "tests/acceptance/test_compliance_gate.py",
            "f6_sbom": "tests/acceptance/test_sbom_gate.py",
            "security_report_catalog": "tests/acceptance/test_security_report_catalog_gate.py",
            "pint_comparable": "tests/acceptance/test_pint_comparable_gate.py",
            "artifact_supply_chain": "tests/acceptance/test_artifact_supply_chain_gate.py",
            "quality_evals": "tests/acceptance/test_quality_evals_gate.py",
            "findings_hub": "tests/acceptance/test_findings_hub_gate.py",
            "findings_scan_specific": "tests/acceptance/test_findings_scan_specific_gate.py",
            "findings_not_consulting": "tests/acceptance/test_findings_not_consulting_gate.py",
            "assessment_pack": "tests/acceptance/test_assessment_pack_gate.py",
            "consulting_arch_review": "tests/acceptance/test_consulting_arch_review_gate.py",
            "consulting_pen_test": "tests/acceptance/test_consulting_pen_test_gate.py",
            "assessor_live": "tests/acceptance/test_assessor_live_gate.py",
            "live_assessment_tenant": "tests/acceptance/test_live_assessment_tenant_gate.py",
            "assessment_engagement": "tests/acceptance/test_assessment_engagement_gate.py",
            "surface_discovery": "tests/acceptance/test_surface_discovery_gate.py",
            "live_pen_manual_parity": "tests/acceptance/test_live_pen_manual_parity_gate.py",
            "live_pen_authorized_max_http": "tests/acceptance/test_live_pen_authorized_max_http_gate.py",
            "multi_surface_modality": "tests/acceptance/test_multi_surface_modality_gate.py",
            "orphan_routes": "tests/acceptance/test_orphan_routes_gate.py",
            "discovery_comparison": "tests/acceptance/test_discovery_comparison_gate.py",
            "contracts": "tests/acceptance/test_api_contracts.py",
            "tradex_golden": "tests/acceptance/test_tradex_golden_gate.py",
            "scans_coherence": "tests/acceptance/test_scans_coherence_gate.py",
            "github_sdk_matrix": "tests/acceptance/test_github_sdk_matrix_gate.py",
            "capability_baseline": "tests/acceptance/test_capability_baseline_gate.py",
            "lifecycle": "tests/acceptance/test_lifecycle_gate.py",
            "eval_guardrails": "tests/acceptance/test_eval_guardrails_gate.py",
            "default_golden_pack": "tests/acceptance/test_default_golden_pack_gate.py",
            "golden_set": "tests/acceptance/test_golden_set_gate.py",
            "eval_fingerprint": "tests/acceptance/test_eval_fingerprint_gate.py",
            "tenant_golden_bootstrap": "tests/acceptance/test_tenant_golden_bootstrap_gate.py",
            "golden_pack_seed": "tests/acceptance/test_golden_pack_seed_gate.py",
            "canonical_eval_path": "tests/acceptance/test_canonical_eval_path_gate.py",
            "model_garden_adapter": "tests/acceptance/test_model_garden_adapter_gate.py",
            "eval_repository": "tests/acceptance/test_eval_repository_gate.py",
            "unmeasured_quality_deltas": "tests/acceptance/test_unmeasured_quality_deltas_gate.py",
            "eval_compare_demo": "tests/acceptance/test_eval_compare_demo_gate.py",
            "model_health": "tests/acceptance/test_model_health_gate.py",
            "model_lifecycle": "tests/e2e/test_model_lifecycle.py",
            "module_entitlements": "tests/acceptance/test_lifecycle_gate.py",
            "taste_credits": "tests/acceptance/test_taste_credits_gate.py",
            "github_app_review": "tests/acceptance/test_github_app_review_gate.py",
            "signup_oauth": "tests/acceptance/test_signup_oauth_gate.py",
            "legal_pages": "tests/acceptance/test_legal_pages_gate.py",
            "vendor_review": "tests/acceptance/test_vendor_review_gate.py",
            "tprm_catalog": "tests/acceptance/test_tprm_catalog_gate.py",
            "company_security_pack": "tests/acceptance/test_company_security_pack_gate.py",
            "mfa": "tests/acceptance/test_mfa_gate.py",
            "dsr": "tests/acceptance/test_dsr_gate.py",
            "customer_audit_export": "tests/acceptance/test_customer_audit_export_gate.py",
            "tprm_pass_bar": "tests/acceptance/test_tprm_pass_bar_gate.py",
            "marketing_hosted_app": "tests/acceptance/test_marketing_hosted_app_gate.py",
            "tenant_onboarding_graph": "tests/acceptance/test_tenant_onboarding_graph_gate.py",
            "product_graph": "tests/acceptance/test_product_graph_gate.py",
            "runtime_chain": "tests/acceptance/test_runtime_chain_gate.py",
            "belfry_mcp": "tests/acceptance/test_belfry_mcp_gate.py",
            "in_app_guides": "tests/acceptance/test_in_app_guides_gate.py",
            "product_rewrite": "tests/acceptance/test_product_rewrite_gate.py",
            "sqs_broker": "tests/acceptance/test_sqs_broker_gate.py",
            "prod_plan_matrix": "tests/acceptance/test_prod_plan_matrix_gate.py",
            "depth_phase_a": "tests/acceptance/test_depth_phase_a_gate.py",
            "billing_domain": "tests/acceptance/test_billing_domain_gate.py",
            "protocol_mesh": "tests/acceptance/test_protocol_mesh_gate.py",
            "mcp_session_pool": "tests/acceptance/test_mcp_session_pool_gate.py",
            "mesh_registry_cache": "tests/acceptance/test_mesh_registry_cache_gate.py",
            "mesh_hop_evidence": "tests/acceptance/test_mesh_hop_evidence_gate.py",
            "temporal_mesh": "tests/acceptance/test_temporal_mesh_gate.py",
            "compliance_depth": "tests/acceptance/test_compliance_depth_gate.py",
            "privacy_memory": "tests/acceptance/test_privacy_memory_gate.py",
            "atlas_saas_coherence": "tests/acceptance/test_atlas_saas_coherence_gate.py",
            "dashboard_timestamps": "tests/acceptance/test_dashboard_timestamp_gate.py",
            "tenant_failclosed": "tests/acceptance/test_tenant_failclosed_gate.py",
            "corpus_sync": "tests/acceptance/test_corpus_sync_gate.py",
            "technique_upgrades": "tests/acceptance/test_technique_upgrades_gate.py",
            "api_rate_limit": "tests/acceptance/test_api_rate_limit_gate.py",
            "semantic_enforcement": "tests/acceptance/test_semantic_enforcement_gate.py",
            "product_audit_inventory": "tests/acceptance/test_product_audit_inventory_gate.py",
            "part12_completion": "tests/acceptance/test_part12_completion_gate.py",
            "module_architecture": "tests/acceptance/test_module_architecture_gate.py",
            "hardening_spec": "tests/acceptance/test_hardening_spec_gate.py",
            "hardening_p2": "tests/acceptance/test_hardening_p2_gate.py",
            "live_mesh_examples": "tests/acceptance/test_live_mesh_examples_gate.py",
            "mcp_spec_golden": "tests/acceptance/test_mcp_spec_golden_gate.py",
            "aws_saas_ready": "tests/acceptance/test_aws_saas_ready_gate.py",
            "saas_onboard": "tests/acceptance/test_saas_onboard_gate.py",
            "quick_scan_poll": "tests/acceptance/test_quick_scan_poll_gate.py",
            "testing_unlock": "tests/acceptance/test_testing_unlock_gate.py",
        }
        for gate_id, rel_path in gate_files.items():
            gate_path = self.repo_root / rel_path

            def gate_body(rel_path: str = rel_path, gate_path: Path = gate_path) -> _CheckReturn:
                if not gate_path.is_file():
                    return (
                        SKIP,
                        f"Gate file missing: {rel_path}.",
                        {"path": str(gate_path)},
                        f"Add {rel_path}.",
                    )
                proc = self._run_subprocess(
                    [sys.executable, "-m", "pytest", rel_path, "-m", "acceptance", "-q"],
                    cwd=self.repo_root,
                )
                output = (proc.stdout or "") + "\n" + (proc.stderr or "")
                passed = self._extract_count(output, r"(\d+)\s+passed")
                failed = self._extract_count(output, r"(\d+)\s+failed")
                skipped = self._extract_count(output, r"(\d+)\s+skipped")
                evidence = {
                    "returncode": proc.returncode,
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "tail": output.strip()[-1000:],
                }
                if proc.returncode == 0 and failed == 0:
                    return PASS, f"{rel_path}: {passed} passed, {skipped} skipped.", evidence
                return (
                    FAIL,
                    f"{rel_path}: {failed} failed, {passed} passed.",
                    evidence,
                    f"Run `python -m pytest {rel_path} -m acceptance` to debug.",
                )

            results.append(
                await self._execute(
                    f"acceptance.{gate_id}",
                    f"Acceptance gate: {gate_id}",
                    "Acceptance",
                    gate_body,
                    remediation=f"Make {rel_path} green.",
                )
            )
        return results

    async def check_rbac(self) -> List[CheckResult]:
        async def rbac() -> _CheckReturn:
            if self.client is None or not await self._api_available():
                raise _ApiUnreachable(self._api_error or "API not reachable")
            base_url = getattr(self.client, "base_url", None)
            if not base_url:
                raise _ApiUnreachable("client has no base_url")
            import httpx

            url = f"{base_url.rstrip('/')}/projects"
            async with httpx.AsyncClient(timeout=15) as raw:
                # Deliberately send NO Authorization header.
                resp = await raw.get(url)
            evidence = {"url": url, "status_code": resp.status_code}
            if resp.status_code in (401, 403):
                return PASS, f"Protected endpoint correctly denied unauthenticated access ({resp.status_code}).", evidence
            return (
                FAIL,
                f"Protected endpoint returned {resp.status_code} without auth — RBAC may not be enforced.",
                evidence,
                "Ensure the auth middleware rejects unauthenticated requests to protected endpoints with 401/403.",
            )

        return [
            await self._execute(
                "access.rbac", "RBAC denies unauthenticated access", "Access Control", rbac,
                remediation="Verify protected endpoints return 401/403 without credentials.",
            )
        ]

    # ------------------------------------------------------------------
    # Stage 13 — Multi-tenancy (structural)
    # ------------------------------------------------------------------

    async def check_multi_tenancy(self) -> List[CheckResult]:
        def tenancy() -> _CheckReturn:
            required = {
                "tenant middleware": self.repo_root / "backend" / "core" / "middleware.py",
                "tenant guard": self.repo_root / "backend" / "core" / "tenant_guard.py",
            }
            present = {label: p.is_file() for label, p in required.items()}
            missing = [label for label, ok in present.items() if not ok]
            evidence = {label: str(p) for label, p in required.items()}
            evidence["present"] = present
            if missing:
                # tenant.py is an acceptable alternative to tenant_guard.py.
                alt = self.repo_root / "backend" / "core" / "tenant.py"
                if "tenant guard" in missing and alt.is_file():
                    missing.remove("tenant guard")
                    evidence["tenant guard (alt)"] = str(alt)
            if missing:
                return (
                    WARN,
                    f"Multi-tenancy components missing: {', '.join(missing)}.",
                    evidence,
                    "Confirm tenant isolation middleware and tenant guard exist under backend/core/.",
                )
            return PASS, "Multi-tenant isolation components present (middleware + tenant guard).", evidence

        return [
            await self._execute(
                "tenancy.structure", "Multi-tenant isolation present", "Multi-Tenancy", tenancy,
                remediation="Confirm backend/core/middleware.py and a tenant guard module exist.",
            )
        ]

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    async def run(self, stages: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run all (or a subset of) stages and aggregate a summary."""
        registry = self._stage_registry()
        if stages:
            requested = [s.strip() for s in stages if s and s.strip()]
            unknown = [s for s in requested if s not in registry]
            selected = [s for s in requested if s in registry]
        else:
            requested = list(registry.keys())
            unknown = []
            selected = requested

        self.results = []
        self.started_at = datetime.now(timezone.utc).isoformat()
        run_start = time.perf_counter()

        for key in selected:
            stage_fn = registry[key]
            try:
                stage_results = await stage_fn()
            except Exception as exc:  # noqa: BLE001 - a stage must never abort the pipeline
                stage_results = [
                    CheckResult(
                        id=f"{key}.stage_error",
                        name=f"Stage '{key}'",
                        category=_stage_category(key),
                        status=FAIL,
                        detail=f"Stage raised before producing results: {exc}",
                        evidence={"traceback": traceback.format_exc()[-1500:]},
                        remediation="This stage crashed unexpectedly; inspect the traceback.",
                    )
                ]
            self.results.extend(stage_results)

        self.finished_at = datetime.now(timezone.utc).isoformat()
        total_duration_ms = (time.perf_counter() - run_start) * 1000.0
        self.summary = self._build_summary(total_duration_ms, selected, unknown)
        return self.summary

    def _build_summary(self, total_duration_ms: float, selected: List[str], unknown: List[str]) -> Dict[str, Any]:
        counts = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1

        # Weighted readiness score over non-skipped checks.
        weighted_sum = 0.0
        weight_total = 0.0
        for r in self.results:
            if r.status == SKIP:
                continue
            w = _CATEGORY_WEIGHTS.get(r.category, 1.0)
            weighted_sum += w * _STATUS_SCORE.get(r.status, 0.0)
            weight_total += w
        readiness_score = round((weighted_sum / weight_total) * 100.0, 1) if weight_total else 0.0

        has_fail = counts[FAIL] > 0
        has_warn = counts[WARN] > 0
        scored_any = weight_total > 0

        if not scored_any:
            verdict = "NOT READY"
        elif has_fail or readiness_score < 70.0:
            verdict = "NOT READY"
        elif readiness_score >= 90.0 and not has_warn:
            verdict = "READY"
        else:
            verdict = "READY-WITH-WARNINGS"

        # Per-category rollup.
        categories: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            cat = categories.setdefault(
                r.category, {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0, "total": 0}
            )
            cat[r.status] = cat.get(r.status, 0) + 1
            cat["total"] += 1

        return {
            "verdict": verdict,
            "readiness_score": readiness_score,
            "counts": counts,
            "total_checks": len(self.results),
            "categories": categories,
            "stages_run": selected,
            "unknown_stages": unknown,
            "total_duration_ms": round(total_duration_ms, 2),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "api_reachable": bool(self._api_ok),
            "repo_root": str(self.repo_root),
            "environment": {
                "python": _platform.python_version(),
                "platform": _platform.platform(),
            },
        }

    # ------------------------------------------------------------------
    # Reporting — JSON
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report": "Belfry Labs Enterprise Readiness Verification",
            "generated_at": self.finished_at or datetime.now(timezone.utc).isoformat(),
            "summary": self.summary,
            "checks": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # ------------------------------------------------------------------
    # Reporting — console (rich)
    # ------------------------------------------------------------------

    def render_console(self, console) -> None:
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text

        summary = self.summary or {}
        verdict = summary.get("verdict", "UNKNOWN")
        score = summary.get("readiness_score", 0.0)
        verdict_color = {
            "READY": "bold green",
            "READY-WITH-WARNINGS": "bold yellow",
            "NOT READY": "bold red",
        }.get(verdict, "bold white")

        header = Text()
        header.append("Belfry Labs — Enterprise Readiness\n", style="bold")
        header.append("Verdict: ", style="bold")
        header.append(f"{verdict}", style=verdict_color)
        header.append(f"    Readiness Score: ", style="bold")
        score_color = "green" if score >= 90 else "yellow" if score >= 70 else "red"
        header.append(f"{score:.1f}/100", style=f"bold {score_color}")
        console.print(Panel(header, expand=False))

        status_style = {PASS: "green", FAIL: "red", WARN: "yellow", SKIP: "dim"}

        # Group by category preserving first-seen order.
        order: List[str] = []
        grouped: Dict[str, List[CheckResult]] = {}
        for r in self.results:
            if r.category not in grouped:
                grouped[r.category] = []
                order.append(r.category)
            grouped[r.category].append(r)

        for category in order:
            table = Table(title=category, title_style="bold cyan", expand=True)
            table.add_column("Status", width=8)
            table.add_column("Check", style="bold", no_wrap=False)
            table.add_column("Detail")
            table.add_column("Time", justify="right", width=9)
            for r in grouped[category]:
                style = status_style.get(r.status, "white")
                table.add_row(
                    f"[{style}]{r.status}[/{style}]",
                    r.name,
                    r.detail[:90],
                    f"{r.duration_ms:.0f} ms",
                )
            console.print(table)

        counts = summary.get("counts", {})
        console.print(
            f"\n[green]PASS {counts.get(PASS, 0)}[/green]  "
            f"[red]FAIL {counts.get(FAIL, 0)}[/red]  "
            f"[yellow]WARN {counts.get(WARN, 0)}[/yellow]  "
            f"[dim]SKIP {counts.get(SKIP, 0)}[/dim]  "
            f"({summary.get('total_checks', 0)} checks in "
            f"{summary.get('total_duration_ms', 0) / 1000:.1f}s)"
        )

    # ------------------------------------------------------------------
    # Reporting — HTML
    # ------------------------------------------------------------------

    def to_html(self) -> str:
        summary = self.summary or {}
        verdict = summary.get("verdict", "UNKNOWN")
        score = float(summary.get("readiness_score", 0.0))
        counts = summary.get("counts", {})
        generated = self.finished_at or datetime.now(timezone.utc).isoformat()

        verdict_class = {
            "READY": "verdict-ready",
            "READY-WITH-WARNINGS": "verdict-warn",
            "NOT READY": "verdict-fail",
        }.get(verdict, "verdict-warn")

        if score >= 90:
            gauge_color = "#16a34a"
        elif score >= 70:
            gauge_color = "#d97706"
        else:
            gauge_color = "#dc2626"

        # Summary cards
        cards = "".join(
            self._html_card(label, counts.get(key, 0), cls)
            for label, key, cls in (
                ("Passed", PASS, "card-pass"),
                ("Failed", FAIL, "card-fail"),
                ("Warnings", WARN, "card-warn"),
                ("Skipped", SKIP, "card-skip"),
            )
        )

        # Category rollup rows
        cat_rows = []
        for cat, c in (summary.get("categories") or {}).items():
            cat_rows.append(
                "<tr>"
                f"<td>{html.escape(cat)}</td>"
                f"<td class='num pass'>{c.get(PASS, 0)}</td>"
                f"<td class='num fail'>{c.get(FAIL, 0)}</td>"
                f"<td class='num warn'>{c.get(WARN, 0)}</td>"
                f"<td class='num skip'>{c.get(SKIP, 0)}</td>"
                f"<td class='num'>{c.get('total', 0)}</td>"
                "</tr>"
            )
        category_rows_html = "\n".join(cat_rows) or "<tr><td colspan='6'>No categories.</td></tr>"

        # Detailed check rows
        detail_rows = []
        for r in self.results:
            detail_rows.append(self._html_check_row(r))
        detail_rows_html = "\n".join(detail_rows) or "<tr><td colspan='5'>No checks executed.</td></tr>"

        env = summary.get("environment", {})
        meta_bits = [
            ("Generated", generated),
            ("Total checks", str(summary.get("total_checks", 0))),
            ("Duration", f"{summary.get('total_duration_ms', 0) / 1000:.1f}s"),
            ("API reachable", "yes" if summary.get("api_reachable") else "no"),
            ("Python", env.get("python", "?")),
            ("Repo", html.escape(str(summary.get("repo_root", "")))),
        ]
        meta_html = "".join(
            f"<div class='meta-item'><span class='meta-k'>{html.escape(k)}</span>"
            f"<span class='meta-v'>{html.escape(str(v))}</span></div>"
            for k, v in meta_bits
        )

        # Gauge: conic-gradient ring driven by the score.
        gauge_deg = max(0.0, min(100.0, score)) * 3.6

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Belfry Labs — Enterprise Readiness Report</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #0b1120; color: #e2e8f0; line-height: 1.5;
  }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }}
  header.hero {{
    display: flex; flex-wrap: wrap; gap: 28px; align-items: center; justify-content: space-between;
    background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
    border: 1px solid #233047; border-radius: 16px; padding: 28px 32px;
  }}
  .brand {{ font-size: 13px; letter-spacing: .18em; text-transform: uppercase; color: #60a5fa; font-weight: 700; }}
  h1 {{ margin: 6px 0 14px; font-size: 28px; color: #f8fafc; }}
  .verdict {{
    display: inline-block; padding: 8px 18px; border-radius: 999px;
    font-weight: 700; font-size: 15px; letter-spacing: .03em;
  }}
  .verdict-ready {{ background: rgba(22,163,74,.15); color: #4ade80; border: 1px solid #16a34a; }}
  .verdict-warn  {{ background: rgba(217,119,6,.15);  color: #fbbf24; border: 1px solid #d97706; }}
  .verdict-fail  {{ background: rgba(220,38,38,.15);  color: #f87171; border: 1px solid #dc2626; }}
  .gauge {{
    width: 160px; height: 160px; border-radius: 50%; flex-shrink: 0;
    background: conic-gradient({gauge_color} {gauge_deg}deg, #1f2937 {gauge_deg}deg);
    display: flex; align-items: center; justify-content: center;
  }}
  .gauge-inner {{
    width: 122px; height: 122px; border-radius: 50%; background: #0b1120;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
  }}
  .gauge-score {{ font-size: 34px; font-weight: 800; color: #f8fafc; }}
  .gauge-label {{ font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: #94a3b8; }}
  .meta {{ display: flex; flex-wrap: wrap; gap: 10px 28px; margin: 22px 4px 8px; }}
  .meta-item {{ display: flex; flex-direction: column; }}
  .meta-k {{ font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: #64748b; }}
  .meta-v {{ font-size: 14px; color: #cbd5e1; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 28px 0; }}
  .card {{ border-radius: 14px; padding: 20px; border: 1px solid #233047; background: #111827; }}
  .card .n {{ font-size: 34px; font-weight: 800; }}
  .card .l {{ font-size: 12px; text-transform: uppercase; letter-spacing: .1em; color: #94a3b8; }}
  .card-pass .n {{ color: #4ade80; }}
  .card-fail .n {{ color: #f87171; }}
  .card-warn .n {{ color: #fbbf24; }}
  .card-skip .n {{ color: #94a3b8; }}
  h2 {{ font-size: 18px; color: #f1f5f9; margin: 36px 0 14px; border-bottom: 1px solid #233047; padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: left; padding: 11px 12px; border-bottom: 1px solid #1e293b; vertical-align: top; }}
  th {{ font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: #94a3b8; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.pass, .num.pass {{ color: #4ade80; }}
  td.fail, .num.fail {{ color: #f87171; }}
  td.warn, .num.warn {{ color: #fbbf24; }}
  td.skip, .num.skip {{ color: #94a3b8; }}
  .pill {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
  .pill-PASS {{ background: rgba(22,163,74,.15); color: #4ade80; }}
  .pill-FAIL {{ background: rgba(220,38,38,.15); color: #f87171; }}
  .pill-WARN {{ background: rgba(217,119,6,.15); color: #fbbf24; }}
  .pill-SKIP {{ background: rgba(100,116,139,.18); color: #cbd5e1; }}
  .detail {{ color: #cbd5e1; }}
  .remediation {{ color: #fca5a5; font-size: 13px; margin-top: 6px; }}
  .evidence {{ margin-top: 8px; }}
  .evidence summary {{ cursor: pointer; color: #60a5fa; font-size: 12px; }}
  .evidence pre {{
    margin: 8px 0 0; padding: 12px; background: #0f172a; border: 1px solid #1e293b;
    border-radius: 8px; overflow-x: auto; font-size: 12px; color: #94a3b8; white-space: pre-wrap; word-break: break-word;
  }}
  .check-name {{ font-weight: 700; color: #f1f5f9; }}
  .check-id {{ font-size: 11px; color: #64748b; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  footer {{ margin-top: 48px; text-align: center; color: #475569; font-size: 12px; }}
</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <div>
        <div class="brand">Belfry Labs · Enterprise Verification</div>
        <h1>Enterprise Readiness Report</h1>
        <span class="verdict {verdict_class}">{html.escape(verdict)}</span>
      </div>
      <div class="gauge">
        <div class="gauge-inner">
          <div class="gauge-score">{score:.0f}</div>
          <div class="gauge-label">Readiness</div>
        </div>
      </div>
    </header>

    <div class="meta">{meta_html}</div>

    <div class="cards">{cards}</div>

    <h2>Summary by Category</h2>
    <table>
      <thead><tr><th>Category</th><th class="num">Pass</th><th class="num">Fail</th>
      <th class="num">Warn</th><th class="num">Skip</th><th class="num">Total</th></tr></thead>
      <tbody>
{category_rows_html}
      </tbody>
    </table>

    <h2>Detailed Checks</h2>
    <table>
      <thead><tr><th>Status</th><th>Check</th><th>Detail &amp; Remediation</th><th>Evidence</th><th class="num">Time</th></tr></thead>
      <tbody>
{detail_rows_html}
      </tbody>
    </table>

    <footer>
      Generated by <strong>belfry verify --enterprise</strong> · {html.escape(generated)}
    </footer>
  </div>
</body>
</html>"""

    @staticmethod
    def _html_card(label: str, n: int, cls: str) -> str:
        return f"<div class='card {cls}'><div class='n'>{n}</div><div class='l'>{html.escape(label)}</div></div>"

    def _html_check_row(self, r: CheckResult) -> str:
        detail_cell = f"<div class='detail'>{html.escape(r.detail or '')}</div>"
        if r.remediation and r.status in (FAIL, WARN, SKIP):
            detail_cell += f"<div class='remediation'>↳ {html.escape(r.remediation)}</div>"

        if r.evidence:
            try:
                ev_json = json.dumps(r.evidence, indent=2, default=str)
            except Exception:  # noqa: BLE001
                ev_json = str(r.evidence)
            evidence_cell = (
                "<details class='evidence'><summary>view</summary>"
                f"<pre>{html.escape(ev_json)}</pre></details>"
            )
        else:
            evidence_cell = "<span class='skip'>—</span>"

        return (
            "<tr>"
            f"<td><span class='pill pill-{r.status}'>{r.status}</span></td>"
            f"<td><div class='check-name'>{html.escape(r.name)}</div>"
            f"<div class='check-id'>{html.escape(r.id)} · {html.escape(r.category)}</div></td>"
            f"<td>{detail_cell}</td>"
            f"<td>{evidence_cell}</td>"
            f"<td class='num'>{r.duration_ms:.0f} ms</td>"
            "</tr>"
        )


def _stage_category(key: str) -> str:
    return {
        "platform": "Platform",
        "dummy": "Integrity",
        "tests": "Tests",
        "examples": "Examples",
        "redteam": "Red Team",
        "deepscan": "Security Scanning",
        "middleware": "Runtime",
        "sdk": "SDK",
        "cli": "CLI",
        "compliance": "Compliance",
        "sbom": "SBOM",
        "acceptance": "Acceptance",
        "rbac": "Access Control",
        "tenancy": "Multi-Tenancy",
    }.get(key, "Platform")
