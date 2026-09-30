"""
LangChain Integration for Belfry Labs
Seamless safety evaluation for LangChain applications

Supports two modes:
1. Cloud Mode: Send evaluations to Belfry Cloud API
2. Local Mode: Evaluate locally with synced policies (enterprise)
"""

from typing import Any, Dict, List, Optional, Union
import asyncio
import logging
import threading
import time
from datetime import datetime

try:
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import AgentAction, AgentFinish, LLMResult
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object

from belfry_labs.async_client import AsyncBelfryLabsClient
from belfry_labs.exceptions import BelfryLabsError
from belfry_labs.local_evaluator import LocalEvaluator, EvaluationResult as LocalResult

logger = logging.getLogger(__name__)


class BelfryLabsCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler for automatic AI safety evaluation
    
    Usage:
        from langchain.llms import OpenAI
        from belfry_labs.integrations import BelfryLabsCallbackHandler
        
        llm = OpenAI(temperature=0.7)
        handler = BelfryLabsCallbackHandler(
            api_key="your-key",
            auto_evaluate=True,
            safety_threshold=0.8
        )
        
        response = llm("Your prompt", callbacks=[handler])
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.belfrylabs.com/v1",
        tenant_id: Optional[str] = None,
        auto_evaluate: bool = True,
        safety_threshold: float = 0.8,
        benchmarks: Optional[List[str]] = None,
        block_unsafe: bool = False,
        **kwargs
    ):
        super().__init__()
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed. Install it with: pip install langchain")
        
        self.client = AsyncBelfryLabsClient(
            api_key=api_key,
            base_url=base_url,
            tenant_id=tenant_id
        )
        self.auto_evaluate = auto_evaluate
        self.safety_threshold = safety_threshold
        self.benchmarks = benchmarks or ["safety", "toxicity", "bias"]
        self.block_unsafe = block_unsafe
        
        self.evaluation_results = []
        self.interactions = []
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """Called when LLM starts running"""
        
        logger.info(f"LLM started with {len(prompts)} prompts")
        
        # Store interaction start
        self.interactions.append({
            "type": "llm_start",
            "prompts": prompts,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": serialized
        })
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM ends running"""
        
        logger.info("LLM completed")
        
        # Extract responses
        responses = []
        for generation_list in response.generations:
            for generation in generation_list:
                responses.append(generation.text)
        
        # Store interaction
        self.interactions.append({
            "type": "llm_end",
            "responses": responses,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Evaluate if auto_evaluate is enabled
        if self.auto_evaluate and responses:
            try:
                # Run evaluation asynchronously
                asyncio.create_task(self._evaluate_responses(responses))
            except Exception as e:
                logger.error(f"Evaluation failed: {str(e)}")
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Called when LLM errors"""
        
        logger.error(f"LLM error: {str(error)}")
        
        self.interactions.append({
            "type": "llm_error",
            "error": str(error),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _evaluate_responses(self, responses: List[str]):
        """Evaluate responses for safety"""
        
        try:
            for response in responses:
                # Create quick evaluation
                result = await self.client.evaluations.quick_evaluate(
                    text=response,
                    benchmarks=self.benchmarks
                )
                
                self.evaluation_results.append(result)
                
                # Check safety threshold
                if result.get("score", 100.0) < self.safety_threshold * 100:
                    logger.warning(
                        f"Safety threshold breached: {result.get('score')}% "
                        f"(threshold: {self.safety_threshold * 100}%)"
                    )
                    
                    if self.block_unsafe:
                        raise BelfryLabsError(
                            f"Response blocked due to safety concerns: "
                            f"score {result.get('score')}%"
                        )
        
        except Exception as e:
            logger.error(f"Response evaluation failed: {str(e)}")
    
    def get_evaluation_results(self) -> List[Dict[str, Any]]:
        """Get all evaluation results"""
        return self.evaluation_results
    
    def get_interactions(self) -> List[Dict[str, Any]]:
        """Get all interactions"""
        return self.interactions


class LangChainIntegration:
    """
    High-level LangChain integration
    
    Usage:
        from belfry_labs.integrations import LangChainIntegration
        
        integration = LangChainIntegration(api_key="your-key")
        
        # Wrap your LLM
        safe_llm = integration.wrap_llm(your_llm)
        
        # Use normally
        response = safe_llm("Your prompt")
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.belfrylabs.com/v1",
        tenant_id: Optional[str] = None,
        **kwargs
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.tenant_id = tenant_id
        self.kwargs = kwargs
    
    def wrap_llm(self, llm, **handler_kwargs):
        """Wrap a LangChain LLM with safety evaluation"""
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed")
        
        # Create callback handler
        handler = BelfryLabsCallbackHandler(
            api_key=self.api_key,
            base_url=self.base_url,
            tenant_id=self.tenant_id,
            **{**self.kwargs, **handler_kwargs}
        )
        
        # Add handler to LLM callbacks
        if hasattr(llm, 'callbacks'):
            if llm.callbacks is None:
                llm.callbacks = []
            llm.callbacks.append(handler)
        else:
            llm.callbacks = [handler]
        
        return llm
    
    def create_callback_handler(self, **kwargs) -> BelfryLabsCallbackHandler:
        """Create a callback handler"""
        
        return BelfryLabsCallbackHandler(
            api_key=self.api_key,
            base_url=self.base_url,
            tenant_id=self.tenant_id,
            **{**self.kwargs, **kwargs}
        )


class LocalBelfryLabsCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler for LOCAL safety evaluation.
    Enterprise feature: Evaluates locally, reports only metrics.
    
    Usage:
        from langchain.llms import OpenAI
        from belfry_labs.integrations import LocalBelfryLabsCallbackHandler
        
        # Create handler with local evaluation
        handler = LocalBelfryLabsCallbackHandler(
            block_unsafe=True,
            report_metrics=True,
            metrics_endpoint="https://api.belfry.ai/v1/interventions/record-batch"
        )
        
        llm = OpenAI(temperature=0.7, callbacks=[handler])
        response = llm("Your prompt")
    """
    
    def __init__(
        self,
        block_unsafe: bool = False,
        report_metrics: bool = False,
        metrics_endpoint: Optional[str] = None,
        metrics_token: Optional[str] = None,
        report_interval_seconds: int = 60,
        evaluator: Optional[LocalEvaluator] = None,
        **kwargs
    ):
        super().__init__()
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed. Install it with: pip install langchain")
        
        self.block_unsafe = block_unsafe
        self.report_metrics = report_metrics
        self.metrics_endpoint = metrics_endpoint
        self.metrics_token = metrics_token
        self.report_interval = report_interval_seconds
        
        self.evaluator = evaluator or LocalEvaluator()
        
        self.evaluation_results: List[LocalResult] = []
        self.interactions: List[Dict[str, Any]] = []
        self._current_prompts: List[str] = []
        
        # Background metrics reporting
        if report_metrics and metrics_endpoint:
            self._start_metrics_reporter()
    
    def _start_metrics_reporter(self):
        """Start background thread for metrics reporting"""
        def report_loop():
            while True:
                time.sleep(self.report_interval)
                self._report_metrics()
        
        thread = threading.Thread(target=report_loop, daemon=True)
        thread.start()
    
    def _report_metrics(self):
        """Report aggregated metrics to Belfry Cloud"""
        if not self.metrics_endpoint:
            return
        
        metrics = self.evaluator.get_metrics()
        if metrics["total_evaluations"] == 0:
            return
        
        try:
            import httpx
            
            headers = {}
            if self.metrics_token:
                headers["Authorization"] = f"Bearer {self.metrics_token}"
            
            # Convert to intervention format
            interventions = []
            for eval_type, count in metrics.get("by_type", {}).items():
                interventions.append({
                    "metric_type": "policy",
                    "count": count,
                    "reason": eval_type
                })
            
            if interventions:
                with httpx.Client() as client:
                    client.post(
                        self.metrics_endpoint,
                        json={"interventions": interventions},
                        headers=headers,
                        timeout=10.0
                    )
            
            self.evaluator.reset_metrics()
            logger.info(f"Reported {metrics['total_evaluations']} evaluations to Belfry")
            
        except Exception as e:
            logger.warning(f"Failed to report metrics: {e}")
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Called when LLM starts running"""
        
        logger.debug(f"LLM started with {len(prompts)} prompts")
        
        # Store prompts for evaluation when response comes back
        self._current_prompts = prompts
        
        self.interactions.append({
            "type": "llm_start",
            "prompts": prompts,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": serialized
        })
        
        # Evaluate inputs immediately
        for prompt in prompts:
            result = self.evaluator.evaluate(input=prompt)
            
            if not result.allowed and self.block_unsafe:
                raise BelfryLabsError(
                    f"Input blocked by safety policy: "
                    f"{[f.message for f in result.findings]}"
                )
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM ends running"""
        
        logger.debug("LLM completed")
        
        # Extract responses
        responses = []
        for generation_list in response.generations:
            for generation in generation_list:
                responses.append(generation.text)
        
        self.interactions.append({
            "type": "llm_end",
            "responses": responses,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Evaluate outputs
        for i, output in enumerate(responses):
            input_text = self._current_prompts[i] if i < len(self._current_prompts) else ""
            
            result = self.evaluator.evaluate(input=input_text, output=output)
            self.evaluation_results.append(result)
            
            if not result.allowed:
                logger.warning(
                    f"Safety issue detected: {[f.message for f in result.findings]}"
                )
                
                if self.block_unsafe:
                    raise BelfryLabsError(
                        f"Output blocked by safety policy: "
                        f"{[f.message for f in result.findings]}"
                    )
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Called when LLM errors"""
        
        logger.error(f"LLM error: {str(error)}")
        
        self.interactions.append({
            "type": "llm_error",
            "error": str(error),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_evaluation_results(self) -> List[LocalResult]:
        """Get all evaluation results"""
        return self.evaluation_results
    
    def get_interactions(self) -> List[Dict[str, Any]]:
        """Get all interactions"""
        return self.interactions
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics"""
        return self.evaluator.get_metrics()


class LocalLangChainIntegration:
    """
    Local-first LangChain integration for enterprise deployments.
    Evaluates locally, reports only aggregated metrics.
    
    Usage:
        from belfry_labs.integrations import LocalLangChainIntegration
        
        integration = LocalLangChainIntegration(
            report_metrics=True,
            metrics_endpoint="https://api.belfry.ai/v1/interventions/record-batch"
        )
        
        # Wrap your LLM
        safe_llm = integration.wrap_llm(your_llm)
        
        # Use normally - evaluates locally
        response = safe_llm("Your prompt")
    """
    
    def __init__(
        self,
        report_metrics: bool = False,
        metrics_endpoint: Optional[str] = None,
        metrics_token: Optional[str] = None,
        **kwargs
    ):
        self.report_metrics = report_metrics
        self.metrics_endpoint = metrics_endpoint
        self.metrics_token = metrics_token
        self.kwargs = kwargs
    
    def wrap_llm(self, llm, **handler_kwargs):
        """Wrap a LangChain LLM with local safety evaluation"""
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed")
        
        # Create callback handler
        handler = LocalBelfryLabsCallbackHandler(
            report_metrics=self.report_metrics,
            metrics_endpoint=self.metrics_endpoint,
            metrics_token=self.metrics_token,
            **{**self.kwargs, **handler_kwargs}
        )
        
        # Add handler to LLM callbacks
        if hasattr(llm, 'callbacks'):
            if llm.callbacks is None:
                llm.callbacks = []
            llm.callbacks.append(handler)
        else:
            llm.callbacks = [handler]
        
        return llm
    
    def create_callback_handler(self, **kwargs) -> LocalBelfryLabsCallbackHandler:
        """Create a local callback handler"""
        
        return LocalBelfryLabsCallbackHandler(
            report_metrics=self.report_metrics,
            metrics_endpoint=self.metrics_endpoint,
            metrics_token=self.metrics_token,
            **{**self.kwargs, **kwargs}
        )
