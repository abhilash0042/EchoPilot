"""
Asynchronous Belfry Labs client.
"""

import asyncio
import json
import os
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from belfry_labs.__version__ import __version__
from belfry_labs.exceptions import (
    BelfryLabsError,
    AuthenticationError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    ServerError,
)
from belfry_labs.types import *
from belfry_labs.resources import (
    ComplianceResource,
    SecurityResource as SecurityResourceClient,
    AnalyticsResource,
    CostsResource,
)


class AsyncBelfryLabsClient:
    """
    Asynchronous client for Belfry Labs.
    
    This client provides full async support with automatic retries, error handling,
    and comprehensive type hints.
    
    Args:
        api_key: Your Belfry Labs API key
        base_url: Base URL for the Belfry Labs API
        tenant_id: Tenant ID for multi-tenant setups
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries for failed requests
        retry_delay: Initial delay between retries in seconds
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.belfrylabs.com/v1",
        tenant_id: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.tenant_id = tenant_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Setup HTTP client
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"belfry-labs-sdk/{__version__}",
        }
        # Access keys (bas_...) must be sent as X-API-Key. Bearer is JWT-only.
        if str(api_key).startswith("bas_"):
            headers["X-API-Key"] = api_key
        
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id
            
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            **kwargs
        )
        
        # Initialize resource clients
        self.projects = ProjectsClient(self)
        self.models = ModelsClient(self)
        self.datasets = DatasetsClient(self)
        self.evaluations = EvaluationsClient(self)
        self.security = SecurityClient(self)
        self.reports = ReportsClient(self)
        self.webhooks = WebhooksClient(self)
        self.costs = CostsClient(self)
        
        # Initialize extended resource clients
        self.compliance = ComplianceResource(self)
        self.security_extended = SecurityResourceClient(self)
        self.analytics = AnalyticsResource(self)
        self.cost_tracking = CostsResource(self)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, RateLimitError))
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an HTTP request with retry logic."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = await self._client.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                files=files,
                **kwargs
            )
            
            # Handle HTTP errors
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 404:
                raise NotFoundError("Resource not found")
            elif response.status_code == 422:
                raise ValidationError(f"Validation error: {response.text}")
            elif response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            elif response.status_code >= 500:
                raise ServerError(f"Server error: {response.status_code}")
            elif response.status_code >= 400:
                raise BelfryLabsError(f"Request failed: {response.status_code} - {response.text}")
            
            response.raise_for_status()
            
            # Parse JSON response
            if response.headers.get("content-type", "").startswith("application/json"):
                return response.json()
            else:
                return {"content": response.content, "headers": dict(response.headers)}
                
        except httpx.RequestError as e:
            raise BelfryLabsError(f"Request failed: {str(e)}")
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # ------------------------------------------------------------------
    # Top-level convenience methods (Phase 3)
    # ------------------------------------------------------------------

    async def protect(
        self,
        text: str,
        project_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check input text against the runtime safety engine.

        Returns a dict with ``action`` (ALLOW / WARN / BLOCK / REDACT)
        and ``findings`` describing any policy violations.
        """
        return await self.check_input(
            text=text,
            project_id=project_id,
            context=context,
        )

    async def check_input(
        self,
        text: str,
        project_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check user input before it is sent to an LLM."""
        payload: Dict[str, Any] = {
            "content": text,
            "tenant_id": self.tenant_id or "default",
            "project_id": project_id,
            "context": context or {},
            "persist": True,
        }
        return await self._request("POST", "/runtime-safety/check/input", json_data=payload)

    async def check_output(
        self,
        text: str,
        project_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check generated content before it is returned to a user."""
        payload: Dict[str, Any] = {
            "content": text,
            "tenant_id": self.tenant_id or "default",
            "project_id": project_id,
            "context": context or {},
            "persist": True,
        }
        return await self._request("POST", "/runtime-safety/check/output", json_data=payload)

    async def check_agent(
        self,
        content: str,
        *,
        tool_name: str,
        agent_id: str,
        project_id: Optional[str] = None,
        tool_calls_count: int = 0,
        dry_run: bool = False,
        kill_switch_active: bool = False,
        kill_switch_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check an agent tool call before executing it."""
        payload: Dict[str, Any] = {
            "content": content,
            "tool_name": tool_name,
            "agent_id": agent_id,
            "tenant_id": self.tenant_id or "default",
            "project_id": project_id,
            "tool_calls_count": tool_calls_count,
            "dry_run": dry_run,
            "kill_switch_active": kill_switch_active,
            "kill_switch_reason": kill_switch_reason,
        }
        return await self._request("POST", "/runtime-safety/check/agent", json_data=payload)

    async def monitor(
        self,
        project_id: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Fetch recent runtime monitoring events (BelfryFlow traces) for a project."""
        return await self._request(
            "GET",
            "/belfry-flow/flows",
            params={"project_id": project_id, "limit": limit},
        )

    async def evaluate(
        self,
        model_id: str,
        project_id: str,
        benchmarks: Optional[List[str]] = None,
        name: Optional[str] = None,
        wait: bool = True,
        timeout: int = 1800,
    ) -> Dict[str, Any]:
        """Create an evaluation and optionally wait for it to complete.

        Args:
            model_id: ID of the model to evaluate.
            project_id: Project the evaluation belongs to.
            benchmarks: Benchmark slugs to run. Defaults to
                ``["prompt_injection", "jailbreak", "toxicity"]``.
            name: Human-readable name for the evaluation run.
            wait: When ``True`` (default) block until the evaluation finishes.
            timeout: Maximum seconds to wait when ``wait=True``.

        Returns:
            Evaluation record dict (or ``Evaluation`` object when waiting).
        """
        payload: Dict[str, Any] = {
            "name": name or f"Evaluation - {model_id}",
            "project_id": project_id,
            "model_id": model_id,
            "benchmarks": benchmarks or ["prompt_injection", "jailbreak", "toxicity"],
        }
        result = await self._request("POST", "/evaluations", json_data=payload)
        if not wait:
            return result
        eval_id = result.get("id") or result.get("evaluation_id")
        if not eval_id:
            return result
        return await self.evaluations.wait_for_completion(eval_id, timeout=timeout)

    async def redteam(
        self,
        project_id: str,
        name: str = "SDK Campaign",
        techniques: Optional[List[str]] = None,
        max_attacks: int = 10,
        include_regression: bool = True,
    ) -> Dict[str, Any]:
        """Run a continuous red-team campaign against a project."""
        payload: Dict[str, Any] = {
            "project_id": project_id,
            "name": name,
            "techniques": techniques or [],
            "max_attacks_per_technique": max_attacks,
            "include_regression": include_regression,
        }
        return await self._request("POST", "/redteam/continuous/run", json_data=payload)

    async def redteam_replay(
        self,
        campaign_id: str,
        promote_regressions: bool = True,
    ) -> Dict[str, Any]:
        """Replay a persisted campaign's attacks to detect defense regressions."""
        return await self._request(
            "POST",
            f"/redteam/campaigns/{campaign_id}/replay",
            params={"promote_regressions": promote_regressions},
        )

    async def redteam_report(
        self,
        campaign_id: str,
        fmt: str = "json",
    ) -> Any:
        """Fetch a JSON (dict) or HTML (str) red-team report for a campaign run."""
        return await self._request(
            "GET",
            f"/redteam/campaigns/{campaign_id}/report",
            params={"format": fmt},
        )

    async def redteam_corpus(self) -> Dict[str, Any]:
        """Summarize the unified attack corpus the orchestrator executes."""
        return await self._request("GET", "/redteam/corpus")

    async def redteam_orchestrator_run(
        self,
        *,
        name: str = "SDK Orchestrator Campaign",
        categories: Optional[List[str]] = None,
        include_corpus: bool = True,
        include_templates: bool = True,
        include_vectors: bool = True,
        include_mutations: bool = False,
        offline: bool = True,
        max_records: int = 500,
    ) -> Dict[str, Any]:
        """Run the unified red-team orchestrator (offline-deterministic by default)."""
        payload: Dict[str, Any] = {
            "name": name,
            "categories": categories or [],
            "include_corpus": include_corpus,
            "include_templates": include_templates,
            "include_vectors": include_vectors,
            "include_mutations": include_mutations,
            "offline": offline,
            "max_records": max_records,
        }
        return await self._request("POST", "/redteam/orchestrator/run", json_data=payload)

    async def scan(
        self,
        path: str,
        project_id: Optional[str] = None,
        *,
        enable_semgrep: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Scan a file or directory for LLM security vulnerabilities.

        Calls the unified code-scanner endpoint
        ``/api/v1/ai-scanner/code-scanner/scan`` (``/ai-scanner/code-scanner/scan``
        relative to the ``/api/v1`` base URL). Semgrep is **opt-in**: default off,
        or set ``enable_semgrep=True`` / ``BELFRY_ENABLE_SEMGREP=1``. When on, the
        server adds js/go/java registry packs for those languages. GitHub PR
        scans do not run Semgrep (ScannerEngine only).
        """
        if enable_semgrep is None:
            enable_semgrep = os.getenv("BELFRY_ENABLE_SEMGREP", "").strip().lower() in {
                "1", "true", "yes",
            }
        payload: Dict[str, Any] = {
            "target_path": str(path),
            "enable_semgrep": bool(enable_semgrep),
        }
        if project_id is not None:
            payload["project_id"] = project_id
        return await self._request(
            "POST", "/ai-scanner/code-scanner/scan", json_data=payload
        )

    async def trace(self, request_id: str) -> Dict[str, Any]:
        """Look up a BelfryFlow trace by its request ID."""
        return await self._request("GET", f"/belfry-flow/flows/{request_id}")

    async def sbom(
        self,
        project_id: str,
        format: str = "cyclonedx",
        *,
        model_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        generate: bool = True,
    ) -> Dict[str, Any]:
        """Generate or list SBOMs / AIBOMs for a project.

        Args:
            project_id: Target project.
            format: ``"cyclonedx"`` (default), ``"spdx"``, or ``"aibom"``.
            model_id: Generate the SBOM for a specific model (optional).
            agent_id: Generate the SBOM for a specific agent (optional).
            generate: When ``True`` (default) generate a new SBOM via
                ``POST /sbom/generate``; when ``False`` list the project's
                existing SBOMs via ``GET /sbom/project/{project_id}``.
        """
        if generate:
            payload: Dict[str, Any] = {
                "project_id": project_id,
                "sbom_format": format,
            }
            if model_id:
                payload["model_id"] = model_id
            if agent_id:
                payload["agent_id"] = agent_id
            return await self._request("POST", "/sbom/generate", json_data=payload)
        return await self._request(
            "GET",
            f"/sbom/project/{project_id}",
            params={"format": format},
        )

    async def sbom_document(
        self,
        project_id: str,
        format: str = "cyclonedx",
    ) -> Dict[str, Any]:
        """Render a fresh, aggregated project AIBOM document (no persistence).

        Returns a schema-valid CycloneDX / SPDX / SARIF / AIBOM document.
        """
        return await self._request(
            "GET",
            f"/sbom/project/{project_id}/document",
            params={"format": format},
        )

    _COMPLIANCE_FRAMEWORK_ALIASES = {
        "nist": "nist_ai_rmf", "owasp": "owasp_top_10", "owasp_llm": "owasp_llm_top_10",
        "pci": "pci_dss", "iso23053": "iso_23053", "iso27001": "iso_27001",
        "iso42001": "iso_42001", "atlas": "mitre_atlas",
    }

    async def compliance(
        self,
        project_id: str,
        framework: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get the compliance posture for a project across frameworks.

        Hits the real ``/compliance/frameworks`` endpoint. When ``framework``
        is provided, the result is filtered to that single framework (legacy
        slugs like ``"nist"`` / ``"owasp"`` / ``"pci"`` are mapped to canonical
        codes such as ``"nist_ai_rmf"``).

        Args:
            project_id: Target project.
            framework: Optional framework code or legacy slug to filter to.
        """
        result = await self._request(
            "GET",
            "/compliance/frameworks",
            params={"project_id": project_id},
        )
        if framework:
            canonical = self._COMPLIANCE_FRAMEWORK_ALIASES.get(
                framework.lower(), framework.lower()
            )
            frameworks = [
                f for f in result.get("frameworks", [])
                if f.get("code") == canonical
            ]
            result = {**result, "frameworks": frameworks}
        return result

    async def compliance_issues(self, project_id: str) -> Dict[str, Any]:
        """Get compliance issues (gaps) for a project."""
        return await self._request(
            "GET",
            f"/compliance/frameworks/{project_id}/issues",
        )

    async def compliance_posture(self, project_id: str) -> Dict[str, Any]:
        """Get a compact compliance posture summary for a project."""
        return await self._request(
            "GET",
            "/compliance/posture",
            params={"project_id": project_id},
        )

    async def cost(
        self,
        project_id: Optional[str] = None,
        timeframe: str = "30d",
    ) -> Dict[str, Any]:
        """Get cost summary and optimization recommendations.

        Args:
            project_id: Scope to a specific project, or ``None`` for
                tenant-wide cost data.
            timeframe: Lookback window, e.g. ``"7d"``, ``"30d"``, ``"90d"``.
        """
        params: Dict[str, Any] = {"timeframe": timeframe}
        if project_id:
            params["project_id"] = project_id
        return await self._request("GET", "/cost-optimization/analysis", params=params)

    async def policy(
        self,
        project_id: str,
        action: str = "list",
    ) -> Dict[str, Any]:
        """List active policies for a project."""
        return await self._request(
            "GET",
            "/policies",
            params={"project_id": project_id},
        )

    async def runtime(
        self,
        text: str,
        project_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Alias for :meth:`protect`. Check text against the runtime safety engine."""
        return await self.protect(text=text, project_id=project_id, context=context)

    async def shutdown(
        self,
        project_id: str,
        reason: str = "SDK shutdown",
    ) -> Dict[str, Any]:
        """Trigger the agent kill-switch for a project.

        Sends a shutdown signal through the agent-control API, which stops
        all running agents associated with the project.
        """
        payload: Dict[str, Any] = {
            "project_id": project_id,
            "reason": reason,
            "action": "shutdown",
        }
        return await self._request("POST", "/agents/kill-switch", json_data=payload)


class BaseResourceClient:
    """Base class for resource-specific clients."""
    
    def __init__(self, client: AsyncBelfryLabsClient):
        self.client = client


class ProjectsClient(BaseResourceClient):
    """Client for project management."""
    
    async def list(
        self,
        skip: int = 0,
        limit: int = 100,
        search: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Project]:
        """List projects."""
        params = {"skip": skip, "limit": limit}
        if search:
            params["search"] = search
        if status:
            params["status"] = status
            
        response = await self.client._request("GET", "/projects", params=params)
        return [Project(**project) for project in response]
    
    async def create(
        self,
        name: str,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Project:
        """Create a new project."""
        data = {"name": name}
        if description:
            data["description"] = description
        if settings:
            data["settings"] = settings
        if tags:
            data["tags"] = tags
            
        response = await self.client._request("POST", "/projects", json_data=data)
        return Project(**response)
    
    async def get(self, project_id: str) -> Project:
        """Get a project by ID."""
        response = await self.client._request("GET", f"/projects/{project_id}")
        return Project(**response)
    
    async def update(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Project:
        """Update a project."""
        data = {}
        if name:
            data["name"] = name
        if description:
            data["description"] = description
        if settings:
            data["settings"] = settings
        if tags:
            data["tags"] = tags
            
        response = await self.client._request("PUT", f"/projects/{project_id}", json_data=data)
        return Project(**response)
    
    async def delete(self, project_id: str) -> Dict[str, str]:
        """Delete a project."""
        return await self.client._request("DELETE", f"/projects/{project_id}")
    
    async def get_stats(self, project_id: str) -> Dict[str, Any]:
        """Get project statistics."""
        return await self.client._request("GET", f"/projects/{project_id}/stats")


class ModelsClient(BaseResourceClient):
    """Client for model management."""
    
    async def list(
        self,
        project_id: Optional[str] = None,
        modality: Optional[str] = None,
        provider: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Model]:
        """List models."""
        params = {"skip": skip, "limit": limit}
        if project_id:
            params["project_id"] = project_id
        if modality:
            params["modality"] = modality
        if provider:
            params["provider"] = provider
            
        response = await self.client._request("GET", "/models", params=params)
        return [Model(**model) for model in response]
    
    async def create(
        self,
        name: str,
        project_id: str,
        modality: str,
        provider: str = "custom",
        source_type: str = "api",
        description: Optional[str] = None,
        version: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Model:
        """Create a new model."""
        data = {
            "name": name,
            "project_id": project_id,
            "modality": modality,
            "provider": provider,
            "source_type": source_type,
        }
        
        if description:
            data["description"] = description
        if version:
            data["version"] = version
        if parameters:
            data["parameters"] = parameters
            
        # Add any additional kwargs
        data.update(kwargs)
            
        response = await self.client._request("POST", "/models", json_data=data)
        return Model(**response)
    
    async def get(self, model_id: str) -> Model:
        """Get a model by ID."""
        response = await self.client._request("GET", f"/models/{model_id}")
        return Model(**response)
    
    async def update(self, model_id: str, **kwargs) -> Model:
        """Update a model."""
        response = await self.client._request("PUT", f"/models/{model_id}", json_data=kwargs)
        return Model(**response)
    
    async def delete(self, model_id: str) -> Dict[str, str]:
        """Delete a model."""
        return await self.client._request("DELETE", f"/models/{model_id}")
    
    async def upload_file(
        self,
        project_id: str,
        file_path: Union[str, Path],
        name: Optional[str] = None,
        description: Optional[str] = None,
        modality: str = "text",
        **kwargs
    ) -> Model:
        """Upload a model file."""
        file_path = Path(file_path)
        
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            data = {
                "project_id": project_id,
                "name": name or file_path.stem,
                "modality": modality,
                "source_type": "file",
            }
            
            if description:
                data["description"] = description
                
            data.update(kwargs)
            
            response = await self.client._request(
                "POST", "/models/upload", files=files, data=data
            )
            return Model(**response)


class DatasetsClient(BaseResourceClient):
    """Client for dataset management."""
    
    async def list(
        self,
        project_id: Optional[str] = None,
        dataset_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dataset]:
        """List datasets."""
        params = {"skip": skip, "limit": limit}
        if project_id:
            params["project_id"] = project_id
        if dataset_type:
            params["dataset_type"] = dataset_type
            
        response = await self.client._request("GET", "/datasets", params=params)
        return [Dataset(**dataset) for dataset in response]
    
    async def create(
        self,
        name: str,
        project_id: str,
        dataset_type: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dataset:
        """Create a new dataset."""
        data = {
            "name": name,
            "project_id": project_id,
            "dataset_type": dataset_type,
        }
        
        if description:
            data["description"] = description
        if tags:
            data["tags"] = tags
        if metadata:
            data["metadata"] = metadata
            
        response = await self.client._request("POST", "/datasets", json_data=data)
        return Dataset(**response)
    
    async def get(self, dataset_id: str) -> Dataset:
        """Get a dataset by ID."""
        response = await self.client._request("GET", f"/datasets/{dataset_id}")
        return Dataset(**response)
    
    async def update(self, dataset_id: str, **kwargs) -> Dataset:
        """Update a dataset."""
        response = await self.client._request("PUT", f"/datasets/{dataset_id}", json_data=kwargs)
        return Dataset(**response)
    
    async def delete(self, dataset_id: str) -> Dict[str, str]:
        """Delete a dataset."""
        return await self.client._request("DELETE", f"/datasets/{dataset_id}")
    
    async def upload_file(
        self,
        project_id: str,
        file_path: Union[str, Path],
        name: Optional[str] = None,
        description: Optional[str] = None,
        dataset_type: str = "text",
        **kwargs
    ) -> Dataset:
        """Upload a dataset file."""
        file_path = Path(file_path)
        
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            data = {
                "project_id": project_id,
                "name": name or file_path.stem,
                "dataset_type": dataset_type,
            }
            
            if description:
                data["description"] = description
                
            data.update(kwargs)
            
            response = await self.client._request(
                "POST", "/datasets/upload", files=files, data=data
            )
            return Dataset(**response)
    
    async def preview(
        self,
        dataset_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Preview dataset contents."""
        params = {"limit": limit, "offset": offset}
        return await self.client._request("GET", f"/datasets/{dataset_id}/preview", params=params)


class EvaluationsClient(BaseResourceClient):
    """Client for evaluation management."""
    
    async def list(
        self,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Evaluation]:
        """List evaluations."""
        params = {"skip": skip, "limit": limit}
        if project_id:
            params["project_id"] = project_id
        if model_id:
            params["model_id"] = model_id
        if status:
            params["status"] = status
            
        response = await self.client._request("GET", "/evaluations", params=params)
        return [Evaluation(**evaluation) for evaluation in response]
    
    async def create(
        self,
        name: str,
        project_id: str,
        model_id: str,
        benchmarks: List[str],
        description: Optional[str] = None,
        dataset_ids: Optional[List[str]] = None,
        evaluation_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Evaluation:
        """Create a new evaluation."""
        data = {
            "name": name,
            "project_id": project_id,
            "model_id": model_id,
            "benchmarks": benchmarks,
        }
        
        if description:
            data["description"] = description
        if dataset_ids:
            data["dataset_ids"] = dataset_ids
        if evaluation_params:
            data["evaluation_params"] = evaluation_params
            
        data.update(kwargs)
            
        response = await self.client._request("POST", "/evaluations", json_data=data)
        return Evaluation(**response)
    
    async def get(self, evaluation_id: str) -> Evaluation:
        """Get an evaluation by ID."""
        response = await self.client._request("GET", f"/evaluations/{evaluation_id}")
        return Evaluation(**response)
    
    async def cancel(self, evaluation_id: str) -> Dict[str, str]:
        """Cancel a running evaluation."""
        return await self.client._request("POST", f"/evaluations/{evaluation_id}/cancel")
    
    async def get_results(self, evaluation_id: str) -> Dict[str, Any]:
        """Get evaluation results."""
        return await self.client._request("GET", f"/evaluations/{evaluation_id}/results")
    
    async def wait_for_completion(
        self,
        evaluation_id: str,
        timeout: int = 3600,
        poll_interval: int = 10
    ) -> Evaluation:
        """Wait for evaluation to complete."""
        start_time = asyncio.get_event_loop().time()
        
        while True:
            evaluation = await self.get(evaluation_id)
            
            if evaluation.status in ["completed", "failed", "cancelled"]:
                return evaluation
            
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Evaluation {evaluation_id} did not complete within {timeout} seconds")
            
            await asyncio.sleep(poll_interval)


class SecurityClient(BaseResourceClient):
    """Client for security and red team testing."""
    
    async def list_red_team_sessions(
        self,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[RedTeamSession]:
        """List red team sessions."""
        params = {"skip": skip, "limit": limit}
        if project_id:
            params["project_id"] = project_id
        if status:
            params["status"] = status
            
        response = await self.client._request("GET", "/security/red-team/sessions", params=params)
        return [RedTeamSession(**session) for session in response]
    
    async def create_red_team_session(
        self,
        name: str,
        project_id: str,
        target_model_id: str,
        attack_types: List[str],
        description: Optional[str] = None,
        max_attempts: int = 100,
        timeout_seconds: int = 300,
        **kwargs
    ) -> RedTeamSession:
        """Create a red team session."""
        data = {
            "name": name,
            "project_id": project_id,
            "target_model_id": target_model_id,
            "attack_types": attack_types,
            "max_attempts": max_attempts,
            "timeout_seconds": timeout_seconds,
        }
        
        if description:
            data["description"] = description
            
        data.update(kwargs)
            
        response = await self.client._request("POST", "/security/red-team/sessions", json_data=data)
        return RedTeamSession(**response)
    
    async def get_red_team_session(self, session_id: str) -> RedTeamSession:
        """Get a red team session by ID."""
        response = await self.client._request("GET", f"/security/red-team/sessions/{session_id}")
        return RedTeamSession(**response)
    
    async def start_red_team_session(self, session_id: str) -> Dict[str, str]:
        """Start a red team session."""
        return await self.client._request("POST", f"/security/red-team/sessions/{session_id}/start")
    
    async def list_vulnerabilities(
        self,
        project_id: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[VulnerabilityReport]:
        """List vulnerability reports."""
        params = {"skip": skip, "limit": limit}
        if project_id:
            params["project_id"] = project_id
        if severity:
            params["severity"] = severity
        if status:
            params["status"] = status
            
        response = await self.client._request("GET", "/security/vulnerabilities", params=params)
        return [VulnerabilityReport(**vuln) for vuln in response]
    
    async def create_vulnerability(
        self,
        title: str,
        description: str,
        project_id: str,
        model_id: str,
        vulnerability_type: str,
        severity: str,
        **kwargs
    ) -> VulnerabilityReport:
        """Create a vulnerability report."""
        data = {
            "title": title,
            "description": description,
            "project_id": project_id,
            "model_id": model_id,
            "vulnerability_type": vulnerability_type,
            "severity": severity,
        }
        
        data.update(kwargs)
            
        response = await self.client._request("POST", "/security/vulnerabilities", json_data=data)
        return VulnerabilityReport(**response)


class ReportsClient(BaseResourceClient):
    """Client for report management."""
    
    async def list(
        self,
        project_id: Optional[str] = None,
        report_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Report]:
        """List reports."""
        params = {"skip": skip, "limit": limit}
        if project_id:
            params["project_id"] = project_id
        if report_type:
            params["report_type"] = report_type
            
        response = await self.client._request("GET", "/reports", params=params)
        return [Report(**report) for report in response]
    
    async def create(
        self,
        title: str,
        project_id: str,
        report_type: str,
        description: Optional[str] = None,
        format: str = "pdf",
        **kwargs
    ) -> Report:
        """Create a new report."""
        data = {
            "title": title,
            "project_id": project_id,
            "report_type": report_type,
            "format": format,
        }
        
        if description:
            data["description"] = description
            
        data.update(kwargs)
            
        response = await self.client._request("POST", "/reports", json_data=data)
        return Report(**response)
    
    async def get(self, report_id: str) -> Report:
        """Get a report by ID."""
        response = await self.client._request("GET", f"/reports/{report_id}")
        return Report(**response)
    
    async def download(self, report_id: str, file_path: Union[str, Path]) -> None:
        """Download a report to a file."""
        response = await self.client._request("GET", f"/reports/{report_id}/download")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(response["content"])


class WebhooksClient(BaseResourceClient):
    """Client for webhook management."""
    
    async def list(
        self,
        project_id: Optional[str] = None,
        provider: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List webhooks."""
        params = {"skip": skip, "limit": limit}
        if project_id:
            params["project_id"] = project_id
        if provider:
            params["provider"] = provider
            
        return await self.client._request("GET", "/webhooks", params=params)
    
    async def create(
        self,
        name: str,
        project_id: str,
        provider: str,
        events: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Create a webhook."""
        data = {
            "name": name,
            "project_id": project_id,
            "provider": provider,
            "events": events,
        }
        
        data.update(kwargs)
            
        return await self.client._request("POST", "/webhooks", json_data=data)
    
    async def get(self, webhook_id: str) -> Dict[str, Any]:
        """Get a webhook by ID."""
        return await self.client._request("GET", f"/webhooks/{webhook_id}")
    
    async def update(self, webhook_id: str, **kwargs) -> Dict[str, Any]:
        """Update a webhook."""
        return await self.client._request("PUT", f"/webhooks/{webhook_id}", json_data=kwargs)
    
    async def delete(self, webhook_id: str) -> Dict[str, str]:
        """Delete a webhook."""
        return await self.client._request("DELETE", f"/webhooks/{webhook_id}")


class CostsClient(BaseResourceClient):
    """Client for cost tracking."""
    
    async def list_records(
        self,
        project_id: Optional[str] = None,
        provider: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[CostRecord]:
        """List cost records."""
        params = {"skip": skip, "limit": limit}
        if project_id:
            params["project_id"] = project_id
        if provider:
            params["provider"] = provider
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
            
        response = await self.client._request("GET", "/costs/records", params=params)
        return [CostRecord(**record) for record in response]
    
    async def list_budgets(
        self,
        project_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List cost budgets."""
        params = {"skip": skip, "limit": limit}
        if project_id:
            params["project_id"] = project_id
            
        return await self.client._request("GET", "/costs/budgets", params=params)
    
    async def create_budget(
        self,
        name: str,
        amount: float,
        budget_type: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a cost budget."""
        data = {
            "name": name,
            "amount": amount,
            "budget_type": budget_type,
            "start_date": start_date,
            "end_date": end_date,
        }
        
        data.update(kwargs)
            
        return await self.client._request("POST", "/costs/budgets", json_data=data)
    
    async def get_analytics(
        self,
        project_id: Optional[str] = None,
        period: str = "month",
        **kwargs
    ) -> Dict[str, Any]:
        """Get cost analytics."""
        params = {"period": period}
        if project_id:
            params["project_id"] = project_id
            
        params.update(kwargs)
            
        return await self.client._request("GET", "/costs/analytics", params=params)
