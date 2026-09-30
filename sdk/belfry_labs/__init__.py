"""
Belfry Labs SDK - Official Python client for Belfry Labs.

This SDK provides a comprehensive interface for interacting with the Belfry Labs,
enabling developers to integrate AI safety evaluation into their workflows.

Quick Start (sync client):
    >>> from belfry_labs import BelfryLabsClient
    >>> client = BelfryLabsClient(api_key="your-api-key")
    >>>
    >>> # Check a prompt against runtime safety policies
    >>> result = client.protect("Tell me to ignore your instructions")
    >>> print(result['action'])  # "block"
    >>>
    >>> # Run a security evaluation
    >>> eval_result = client.evaluate(
    ...     model_id="my-model-id",
    ...     project_id="my-project-id",
    ...     benchmarks=["prompt_injection", "jailbreak"],
    ... )
    >>>
    >>> # Red-team a project
    >>> campaign = client.redteam(project_id="my-project-id", max_attacks=20)
    >>>
    >>> # Compliance check
    >>> report = client.compliance(project_id="my-project-id", framework="nist")
    >>>
    >>> # Cost analysis
    >>> costs = client.cost(project_id="my-project-id", timeframe="30d")
    >>>
    >>> # Generate SBOM
    >>> bom = client.sbom(project_id="my-project-id", format="cyclonedx")
    >>>
    >>> # Scan source code for LLM vulnerabilities
    >>> findings = client.scan(path="/path/to/repo", project_id="my-project-id")
    >>>
    >>> # Fetch a BelfryFlow trace
    >>> trace = client.trace(request_id="req_abc123")
    >>>
    >>> # Monitor recent events
    >>> events = client.monitor(project_id="my-project-id", limit=50)
    >>>
    >>> # Emergency agent shutdown
    >>> client.shutdown(project_id="my-project-id", reason="Incident response")

Quick Start (async client):
    >>> import asyncio
    >>> from belfry_labs import AsyncBelfryLabsClient
    >>>
    >>> async def main():
    ...     async with AsyncBelfryLabsClient(api_key="your-api-key") as client:
    ...         # Create a project
    ...         project = await client.projects.create(
    ...             name="My AI Project",
    ...             description="Testing my language model"
    ...         )
    ...
    ...         # Upload a model
    ...         model = await client.models.create(
    ...             name="my-model",
    ...             project_id=project.id,
    ...             modality="text",
    ...             provider="openai"
    ...         )
    ...
    ...         # Run safety evaluation (convenience method)
    ...         evaluation = await client.evaluate(
    ...             model_id=model.id,
    ...             project_id=project.id,
    ...             benchmarks=["prompt_injection", "jailbreak", "toxicity"],
    ...         )
    ...
    ...         # Or use the full resource client
    ...         evaluation = await client.evaluations.create(
    ...             name="Safety Check",
    ...             project_id=project.id,
    ...             model_id=model.id,
    ...             benchmarks=["prompt_injection", "jailbreak", "toxicity"]
    ...         )
    >>>
    >>> asyncio.run(main())

Top-level convenience methods (both clients):
    protect(text, project_id, context)      - Runtime safety check (ALLOW/WARN/BLOCK/REDACT)
    check_input(text, project_id, context)  - Check user input before an LLM call
    check_output(text, project_id, context) - Check model output before returning it
    check_agent(content, tool_name, ...)     - Check an agent tool call before execution
    runtime(text, project_id, context)      - Alias for protect()
    monitor(project_id, limit)              - Recent BelfryFlow monitoring events
    evaluate(model_id, project_id, ...)     - Create and optionally await an evaluation
    redteam(project_id, name, ...)          - Run a red-team campaign
    scan(path, project_id)                  - Scan code for LLM vulnerabilities
    trace(request_id)                       - Look up a BelfryFlow trace
    sbom(project_id, format)               - Generate SBOM / AIBOM
    compliance(project_id, framework)      - Compliance check against a framework
    cost(project_id, timeframe)            - Cost summary and recommendations
    policy(project_id)                     - List active policies
    shutdown(project_id, reason)           - Trigger agent kill-switch
"""

from belfry_labs.__version__ import __version__
from belfry_labs.client import BelfryLabsClient
from belfry_labs.async_client import AsyncBelfryLabsClient
from belfry_labs.exceptions import (
    BelfryLabsError,
    AuthenticationError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    ServerError,
)
from belfry_labs.types import (
    Project,
    Model,
    Dataset,
    Evaluation,
    RedTeamSession,
    VulnerabilityReport,
    Report,
    CostRecord,
)
from belfry_labs.policy_client import (
    BelfryPolicyClient,
    PolicyAction,
    PolicySeverity,
    EvaluationResult as PolicyEvaluationResult,
)
from belfry_labs.local_evaluator import (
    LocalEvaluator,
    EvaluationResult,
    EvaluationType,
    RiskLevel,
    Action,
    quick_evaluate,
)

__all__ = [
    "__version__",
    # Clients
    "BelfryLabsClient",
    "AsyncBelfryLabsClient",
    # Policy sync (enterprise)
    "BelfryPolicyClient",
    "PolicyAction",
    "PolicySeverity",
    "PolicyEvaluationResult",
    # Local evaluation (enterprise)
    "LocalEvaluator",
    "EvaluationResult",
    "EvaluationType",
    "RiskLevel",
    "Action",
    "quick_evaluate",
    # Exceptions
    "BelfryLabsError",
    "AuthenticationError", 
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "ServerError",
    # Types
    "Project",
    "Model",
    "Dataset",
    "Evaluation",
    "RedTeamSession",
    "VulnerabilityReport",
    "Report",
    "CostRecord",
]
