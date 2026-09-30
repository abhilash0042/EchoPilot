"""
Belfry Labs Synchronous Client
Synchronous wrapper for AI safety evaluation
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import json

from .async_client import AsyncBelfryLabsClient
from .utils import logging_utils, metrics_utils

logger = logging.getLogger(__name__)


class SyncWrapper:
    """
    Synchronous wrapper for async operations
    """
    
    def __init__(self, async_client: AsyncBelfryLabsClient):
        self.async_client = async_client
        self._loop = None
        
    def _get_loop(self):
        """Get or create event loop"""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
    
    def _run_async(self, coro):
        """Run async coroutine in sync context"""
        loop = self._get_loop()
        
        if loop.is_running():
            # If we're already in an event loop, we need to use a different approach
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)


class BelfryLabsClient:
    """
    Synchronous Belfry Labs Client
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        retry_attempts: int = 3
    ):
        """
        Initialize Belfry Labs Client
    
    Args:
            api_key: API key for authentication
            base_url: Base URL for the API
        timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
        """
        
        self.async_client = AsyncBelfryLabsClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            retry_attempts=retry_attempts
        )
        
        self.sync_wrapper = SyncWrapper(self.async_client)
        
        # Setup logging
        self.logger = logging_utils.setup_logger(
            name=f"{__name__}.{self.__class__.__name__}",
            level=logging_utils.LogLevel.INFO
        )
        
        # Initialize metrics
        self.metrics = metrics_utils
        
        self.logger.info("Belfry Labs Client initialized")
    
    def evaluate_model(
        self,
        model_id: str,
        benchmark_ids: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a model synchronously
        
        Args:
            model_id: ID of the model to evaluate
            benchmark_ids: List of benchmark IDs to run
            parameters: Evaluation parameters
            
        Returns:
            Evaluation results
        """
        
        self.logger.info(f"Starting evaluation for model {model_id}")
        self.metrics.increment_counter("evaluations_started")
        
        start_time = datetime.utcnow()
        
        try:
            result = self.sync_wrapper._run_async(
                self.async_client.evaluate_model(
                    model_id=model_id,
                    benchmark_ids=benchmark_ids,
                    parameters=parameters
                )
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.record_metric("evaluation_duration", duration)
            self.metrics.increment_counter("evaluations_completed")
            
            self.logger.info(f"Evaluation completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            self.metrics.increment_counter("evaluations_failed")
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def get_evaluation_results(
        self,
        evaluation_id: str
    ) -> Dict[str, Any]:
        """
        Get evaluation results synchronously
        
        Args:
            evaluation_id: ID of the evaluation
            
        Returns:
            Evaluation results
        """
        
        self.logger.info(f"Getting results for evaluation {evaluation_id}")
        
        try:
            result = self.sync_wrapper._run_async(
                self.async_client.get_evaluation_results(evaluation_id)
            )
            
            self.logger.info("Results retrieved successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get results: {str(e)}")
            raise
    
    def list_benchmarks(
        self,
        modality: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available benchmarks synchronously
        
        Args:
            modality: Filter by modality
            category: Filter by category
            
        Returns:
            List of benchmarks
        """
        
        self.logger.info("Listing benchmarks")
        
        try:
            result = self.sync_wrapper._run_async(
                self.async_client.list_benchmarks(
                    modality=modality,
                    category=category
                )
            )
            
            self.logger.info(f"Found {len(result)} benchmarks")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to list benchmarks: {str(e)}")
            raise
    
    def get_benchmark_info(
        self,
        benchmark_id: str
    ) -> Dict[str, Any]:
        """
        Get benchmark information synchronously
        
        Args:
            benchmark_id: ID of the benchmark
            
        Returns:
            Benchmark information
        """
        
        self.logger.info(f"Getting info for benchmark {benchmark_id}")
        
        try:
            result = self.sync_wrapper._run_async(
                self.async_client.get_benchmark_info(benchmark_id)
            )
            
            self.logger.info("Benchmark info retrieved successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get benchmark info: {str(e)}")
            raise
    
    def run_benchmark(
        self,
        benchmark_id: str,
        model_id: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a single benchmark synchronously
        
        Args:
            benchmark_id: ID of the benchmark to run
            model_id: ID of the model to test
            parameters: Benchmark parameters
            
        Returns:
            Benchmark results
        """
        
        self.logger.info(f"Running benchmark {benchmark_id} on model {model_id}")
        self.metrics.increment_counter("benchmarks_started")
        
        start_time = datetime.utcnow()
        
        try:
            result = self.sync_wrapper._run_async(
                self.async_client.run_benchmark(
                    benchmark_id=benchmark_id,
                    model_id=model_id,
                    parameters=parameters
                )
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.record_metric("benchmark_duration", duration)
            self.metrics.increment_counter("benchmarks_completed")
            
            self.logger.info(f"Benchmark completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            self.metrics.increment_counter("benchmarks_failed")
            self.logger.error(f"Benchmark failed: {str(e)}")
            raise
    
    def get_model_info(
        self,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Get model information synchronously
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model information
        """
        
        self.logger.info(f"Getting info for model {model_id}")
        
        try:
            result = self.sync_wrapper._run_async(
                self.async_client.get_model_info(model_id)
            )
            
            self.logger.info("Model info retrieved successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {str(e)}")
            raise
    
    def list_models(
        self,
        modality: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available models synchronously
        
        Args:
            modality: Filter by modality
            status: Filter by status
            
        Returns:
            List of models
        """
        
        self.logger.info("Listing models")
        
        try:
            result = self.sync_wrapper._run_async(
                self.async_client.list_models(
                    modality=modality,
                    status=status
                )
            )
            
            self.logger.info(f"Found {len(result)} models")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {str(e)}")
            raise
    
    def create_project(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new project synchronously
        
        Args:
            name: Project name
            description: Project description
            metadata: Additional metadata
            
        Returns:
            Project information
        """
        
        self.logger.info(f"Creating project: {name}")
        
        try:
            result = self.sync_wrapper._run_async(
                self.async_client.create_project(
                    name=name,
                    description=description,
                    metadata=metadata
                )
            )
            
            self.logger.info("Project created successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create project: {str(e)}")
            raise
    
    def get_project(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Get project information synchronously
        
        Args:
            project_id: ID of the project
            
        Returns:
            Project information
        """
        
        self.logger.info(f"Getting project {project_id}")
        
        try:
            result = self.sync_wrapper._run_async(
                self.async_client.get_project(project_id)
            )
            
            self.logger.info("Project retrieved successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get project: {str(e)}")
            raise
    
    def list_projects(
        self,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List projects synchronously
        
        Args:
            status: Filter by status
            
        Returns:
            List of projects
        """
        
        self.logger.info("Listing projects")
        
        try:
            result = self.sync_wrapper._run_async(
                self.async_client.list_projects(status=status)
            )
            
            self.logger.info(f"Found {len(result)} projects")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to list projects: {str(e)}")
            raise
    
    def generate_report(
        self,
        evaluation_id: str,
        format: str = "json"
    ) -> Union[Dict[str, Any], str]:
        """
        Generate evaluation report synchronously
        
        Args:
            evaluation_id: ID of the evaluation
            format: Report format (json, html, pdf)
            
        Returns:
            Report data
        """
        
        self.logger.info(f"Generating report for evaluation {evaluation_id}")
        
        try:
            result = self.sync_wrapper._run_async(
                self.async_client.generate_report(
                    evaluation_id=evaluation_id,
                    format=format
                )
            )
            
            self.logger.info("Report generated successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get client metrics synchronously
        
        Returns:
            Metrics data
        """
        
        self.logger.info("Getting client metrics")
        
        try:
            result = self.sync_wrapper._run_async(
                self.async_client.get_metrics()
            )
            
            self.logger.info("Metrics retrieved successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {str(e)}")
            raise
    
    # ------------------------------------------------------------------
    # Top-level convenience methods (Phase 3)
    # ------------------------------------------------------------------

    def protect(
        self,
        text: str,
        project_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check text against the runtime safety engine.

        Returns a dict with ``action`` (ALLOW / WARN / BLOCK / REDACT) and
        ``findings`` describing any policy violations.
        """
        return self.sync_wrapper._run_async(
            self.async_client.protect(text=text, project_id=project_id, context=context)
        )

    def check_input(
        self,
        text: str,
        project_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check user input before it is sent to an LLM."""
        return self.sync_wrapper._run_async(
            self.async_client.check_input(text=text, project_id=project_id, context=context)
        )

    def check_output(
        self,
        text: str,
        project_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check generated content before it is returned to a user."""
        return self.sync_wrapper._run_async(
            self.async_client.check_output(text=text, project_id=project_id, context=context)
        )

    def check_agent(
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
        return self.sync_wrapper._run_async(
            self.async_client.check_agent(
                content,
                tool_name=tool_name,
                agent_id=agent_id,
                project_id=project_id,
                tool_calls_count=tool_calls_count,
                dry_run=dry_run,
                kill_switch_active=kill_switch_active,
                kill_switch_reason=kill_switch_reason,
            )
        )

    def monitor(
        self,
        project_id: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Fetch recent runtime monitoring events (BelfryFlow traces) for a project."""
        return self.sync_wrapper._run_async(
            self.async_client.monitor(project_id=project_id, limit=limit)
        )

    def evaluate(
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
        """
        return self.sync_wrapper._run_async(
            self.async_client.evaluate(
                model_id=model_id,
                project_id=project_id,
                benchmarks=benchmarks,
                name=name,
                wait=wait,
                timeout=timeout,
            )
        )

    def redteam(
        self,
        project_id: str,
        name: str = "SDK Campaign",
        techniques: Optional[List[str]] = None,
        max_attacks: int = 10,
        include_regression: bool = True,
    ) -> Dict[str, Any]:
        """Run a continuous red-team campaign against a project."""
        return self.sync_wrapper._run_async(
            self.async_client.redteam(
                project_id=project_id,
                name=name,
                techniques=techniques,
                max_attacks=max_attacks,
                include_regression=include_regression,
            )
        )

    def redteam_replay(
        self,
        campaign_id: str,
        promote_regressions: bool = True,
    ) -> Dict[str, Any]:
        """Replay a persisted campaign's attacks to detect defense regressions."""
        return self.sync_wrapper._run_async(
            self.async_client.redteam_replay(
                campaign_id=campaign_id,
                promote_regressions=promote_regressions,
            )
        )

    def redteam_report(self, campaign_id: str, fmt: str = "json") -> Any:
        """Fetch a JSON (dict) or HTML (str) red-team report for a campaign run."""
        return self.sync_wrapper._run_async(
            self.async_client.redteam_report(campaign_id=campaign_id, fmt=fmt)
        )

    def redteam_corpus(self) -> Dict[str, Any]:
        """Summarize the unified attack corpus the orchestrator executes."""
        return self.sync_wrapper._run_async(self.async_client.redteam_corpus())

    def redteam_orchestrator_run(
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
        return self.sync_wrapper._run_async(
            self.async_client.redteam_orchestrator_run(
                name=name,
                categories=categories,
                include_corpus=include_corpus,
                include_templates=include_templates,
                include_vectors=include_vectors,
                include_mutations=include_mutations,
                offline=offline,
                max_records=max_records,
            )
        )

    def scan(
        self,
        path: str,
        project_id: Optional[str] = None,
        *,
        enable_semgrep: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Scan a file or directory for LLM security vulnerabilities.

        Semgrep is opt-in (``enable_semgrep=True`` or ``BELFRY_ENABLE_SEMGREP=1``).
        """
        return self.sync_wrapper._run_async(
            self.async_client.scan(
                path=path, project_id=project_id, enable_semgrep=enable_semgrep
            )
        )

    def trace(self, request_id: str) -> Dict[str, Any]:
        """Look up a BelfryFlow trace by its request ID."""
        return self.sync_wrapper._run_async(
            self.async_client.trace(request_id=request_id)
        )

    def sbom(
        self,
        project_id: str,
        format: str = "cyclonedx",
    ) -> Dict[str, Any]:
        """Generate an SBOM / AIBOM for a project.

        Args:
            project_id: Target project.
            format: ``"cyclonedx"`` (default) or ``"spdx"``.
        """
        return self.sync_wrapper._run_async(
            self.async_client.sbom(project_id=project_id, format=format)
        )

    def compliance(
        self,
        project_id: str,
        framework: str = "nist",
    ) -> Dict[str, Any]:
        """Run a compliance check against a named framework.

        Args:
            project_id: Target project.
            framework: Framework slug, e.g. ``"nist"``, ``"owasp"``,
                ``"eu_ai_act"``, ``"hipaa"``, ``"pci"``.
        """
        return self.sync_wrapper._run_async(
            self.async_client.compliance(project_id=project_id, framework=framework)
        )

    def cost(
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
        return self.sync_wrapper._run_async(
            self.async_client.cost(project_id=project_id, timeframe=timeframe)
        )

    def policy(
        self,
        project_id: str,
        action: str = "list",
    ) -> Dict[str, Any]:
        """List active policies for a project."""
        return self.sync_wrapper._run_async(
            self.async_client.policy(project_id=project_id, action=action)
        )

    def runtime(
        self,
        text: str,
        project_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Alias for :meth:`protect`. Check text against the runtime safety engine."""
        return self.sync_wrapper._run_async(
            self.async_client.runtime(text=text, project_id=project_id, context=context)
        )

    def shutdown(
        self,
        project_id: str,
        reason: str = "SDK shutdown",
    ) -> Dict[str, Any]:
        """Trigger the agent kill-switch for a project."""
        return self.sync_wrapper._run_async(
            self.async_client.shutdown(project_id=project_id, reason=reason)
        )

    def close(self):
        """Close the client"""
        
        self.logger.info("Closing Belfry Labs Client")
        
        try:
            self.sync_wrapper._run_async(
                self.async_client.close()
            )
            
            self.logger.info("Client closed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to close client: {str(e)}")
            raise
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def __del__(self):
        """Destructor"""
        try:
            self.close()
        except:
            pass