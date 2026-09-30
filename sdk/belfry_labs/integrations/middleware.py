"""
Universal Middleware for Belfry Labs
Framework-agnostic integration with decorators and wrappers
"""

from typing import Any, Callable, Dict, List, Optional
import asyncio
import functools
import logging
from datetime import datetime

from belfry_labs.async_client import AsyncBelfryLabsClient
from belfry_labs.exceptions import BelfryLabsError

logger = logging.getLogger(__name__)


class BelfryLabsMiddleware:
    """
    Universal middleware for any AI framework
    
    Usage:
        from belfry_labs.integrations import BelfryLabsMiddleware
        
        middleware = BelfryLabsMiddleware(api_key="your-key")
        
        # Wrap any model
        safe_model = middleware.wrap_model(your_model)
        
        # Wrap any API client
        safe_client = middleware.wrap_api_client(your_client)
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
        self.client = AsyncBelfryLabsClient(
            api_key=api_key,
            base_url=base_url,
            tenant_id=tenant_id
        )
        self.auto_evaluate = auto_evaluate
        self.safety_threshold = safety_threshold
        self.benchmarks = benchmarks or ["safety", "toxicity", "bias"]
        self.block_unsafe = block_unsafe
        self.kwargs = kwargs
        
        self.evaluation_results = []
    
    def wrap_model(self, model):
        """Wrap any AI model with safety evaluation"""
        
        return SafetyWrappedModel(
            model,
            self.client,
            self.auto_evaluate,
            self.safety_threshold,
            self.benchmarks,
            self.block_unsafe,
            self.evaluation_results
        )
    
    def wrap_api_client(self, client):
        """Wrap any API client with safety evaluation"""
        
        return SafetyWrappedClient(
            client,
            self.client,
            self.auto_evaluate,
            self.safety_threshold,
            self.benchmarks,
            self.block_unsafe,
            self.evaluation_results
        )
    
    def get_evaluation_results(self) -> List[Dict[str, Any]]:
        """Get all evaluation results"""
        return self.evaluation_results


class SafetyWrappedModel:
    """Wrapped model with automatic safety evaluation"""
    
    def __init__(
        self,
        model,
        belfrylabs_client,
        auto_evaluate,
        safety_threshold,
        benchmarks,
        block_unsafe,
        evaluation_results
    ):
        self.model = model
        self.belfrylabs_client = belfrylabs_client
        self.auto_evaluate = auto_evaluate
        self.safety_threshold = safety_threshold
        self.benchmarks = benchmarks
        self.block_unsafe = block_unsafe
        self.evaluation_results = evaluation_results
    
    def __call__(self, *args, **kwargs):
        """Call model with safety evaluation"""
        
        # Call original model
        result = self.model(*args, **kwargs)
        
        # Evaluate if enabled
        if self.auto_evaluate:
            asyncio.create_task(self._evaluate_result(result))
        
        return result
    
    async def _evaluate_result(self, result):
        """Evaluate model result"""
        
        try:
            # Extract text from result
            text = self._extract_text(result)
            
            if text:
                eval_result = await self.belfrylabs_client.evaluations.quick_evaluate(
                    text=text,
                    benchmarks=self.benchmarks
                )
                
                self.evaluation_results.append(eval_result)
                
                # Check safety threshold
                if eval_result.get("score", 100.0) < self.safety_threshold * 100:
                    logger.warning(
                        f"Safety threshold breached: {eval_result.get('score')}% "
                        f"(threshold: {self.safety_threshold * 100}%)"
                    )
                    
                    if self.block_unsafe:
                        raise BelfryLabsError(
                            f"Output blocked due to safety concerns: "
                            f"score {eval_result.get('score')}%"
                        )
        
        except Exception as e:
            logger.error(f"Result evaluation failed: {str(e)}")
    
    def _extract_text(self, result) -> Optional[str]:
        """Extract text from various result formats"""
        
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            # Try common keys
            for key in ['text', 'content', 'output', 'generated_text', 'response']:
                if key in result:
                    return str(result[key])
        elif isinstance(result, list) and result:
            return self._extract_text(result[0])
        
        return str(result) if result else None
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped model"""
        return getattr(self.model, name)


class SafetyWrappedClient:
    """Wrapped API client with automatic safety evaluation"""
    
    def __init__(
        self,
        client,
        belfrylabs_client,
        auto_evaluate,
        safety_threshold,
        benchmarks,
        block_unsafe,
        evaluation_results
    ):
        self.client = client
        self.belfrylabs_client = belfrylabs_client
        self.auto_evaluate = auto_evaluate
        self.safety_threshold = safety_threshold
        self.benchmarks = benchmarks
        self.block_unsafe = block_unsafe
        self.evaluation_results = evaluation_results
    
    def __getattr__(self, name):
        """Wrap all methods with safety evaluation"""
        
        attr = getattr(self.client, name)
        
        if callable(attr):
            @functools.wraps(attr)
            async def wrapped(*args, **kwargs):
                # Call original method
                result = await attr(*args, **kwargs)
                
                # Evaluate if enabled
                if self.auto_evaluate:
                    await self._evaluate_result(result)
                
                return result
            
            return wrapped
        
        return attr
    
    async def _evaluate_result(self, result):
        """Evaluate API result"""
        
        try:
            # Extract text from result
            text = self._extract_text(result)
            
            if text:
                eval_result = await self.belfrylabs_client.evaluations.quick_evaluate(
                    text=text,
                    benchmarks=self.benchmarks
                )
                
                self.evaluation_results.append(eval_result)
                
                # Check safety threshold
                if eval_result.get("score", 100.0) < self.safety_threshold * 100:
                    logger.warning(
                        f"Safety threshold breached: {eval_result.get('score')}% "
                        f"(threshold: {self.safety_threshold * 100}%)"
                    )
                    
                    if self.block_unsafe:
                        raise BelfryLabsError(
                            f"Output blocked due to safety concerns: "
                            f"score {eval_result.get('score')}%"
                        )
        
        except Exception as e:
            logger.error(f"Result evaluation failed: {str(e)}")
    
    def _extract_text(self, result) -> Optional[str]:
        """Extract text from various result formats"""
        
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            # Try common keys
            for key in ['text', 'content', 'output', 'generated_text', 'response']:
                if key in result:
                    return str(result[key])
        elif isinstance(result, list) and result:
            return self._extract_text(result[0])
        
        return str(result) if result else None


def safety_evaluate(
    api_key: str,
    benchmarks: Optional[List[str]] = None,
    threshold: float = 0.8,
    block_unsafe: bool = False,
    base_url: str = "https://api.belfrylabs.com/v1",
    tenant_id: Optional[str] = None
):
    """
    Decorator for automatic safety evaluation
    
    Usage:
        from belfry_labs.integrations import safety_evaluate
        
        @safety_evaluate(
            api_key="your-key",
            benchmarks=["safety", "toxicity"],
            threshold=0.8
        )
        async def your_ai_function(prompt: str) -> str:
            # Your AI code here
            return ai_response
    """
    
    def decorator(func: Callable) -> Callable:
        client = AsyncBelfryLabsClient(
            api_key=api_key,
            base_url=base_url,
            tenant_id=tenant_id
        )
        
        benchmarks_list = benchmarks or ["safety", "toxicity", "bias"]
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Call original function
            result = await func(*args, **kwargs)
            
            # Extract text
            text = result if isinstance(result, str) else str(result)
            
            # Evaluate
            try:
                eval_result = await client.evaluations.quick_evaluate(
                    text=text,
                    benchmarks=benchmarks_list,
                    metadata={
                        "function": func.__name__,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                # Check threshold
                if eval_result.get("score", 100.0) < threshold * 100:
                    logger.warning(
                        f"Safety threshold breached in {func.__name__}: "
                        f"{eval_result.get('score')}% (threshold: {threshold * 100}%)"
                    )
                    
                    if block_unsafe:
                        raise BelfryLabsError(
                            f"Output blocked due to safety concerns: "
                            f"score {eval_result.get('score')}%"
                        )
            
            except Exception as e:
                logger.error(f"Safety evaluation failed in {func.__name__}: {str(e)}")
            
            return result
        
        return wrapper
    
    return decorator
