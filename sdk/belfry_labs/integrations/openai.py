"""
OpenAI Integration for Belfry Labs
Seamless safety evaluation for OpenAI API calls
"""

from typing import Any, Dict, List, Optional, Union
import asyncio
import logging
from datetime import datetime

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from belfry_labs.async_client import AsyncBelfryLabsClient
from belfry_labs.exceptions import BelfryLabsError

logger = logging.getLogger(__name__)


class BelfryLabsWrapper:
    """
    Wrapper for OpenAI client with automatic safety evaluation
    
    Usage:
        import openai
        from belfry_labs.integrations import BelfryLabsWrapper
        
        # Wrap OpenAI client
        client = BelfryLabsWrapper(
            openai.OpenAI(api_key="your-openai-key"),
            belfrylabs_api_key="your-belfrylabs-key",
            auto_evaluate=True
        )
        
        # Use normally - all calls are automatically evaluated
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Your prompt"}]
        )
    """
    
    def __init__(
        self,
        openai_client,
        belfrylabs_api_key: str,
        base_url: str = "https://api.belfrylabs.com/v1",
        tenant_id: Optional[str] = None,
        auto_evaluate: bool = True,
        safety_threshold: float = 0.8,
        benchmarks: Optional[List[str]] = None,
        block_unsafe: bool = False,
        **kwargs
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI is not installed. Install it with: pip install openai")
        
        self.openai_client = openai_client
        self.belfrylabs_client = AsyncBelfryLabsClient(
            api_key=belfrylabs_api_key,
            base_url=base_url,
            tenant_id=tenant_id
        )
        self.auto_evaluate = auto_evaluate
        self.safety_threshold = safety_threshold
        self.benchmarks = benchmarks or ["safety", "toxicity", "bias"]
        self.block_unsafe = block_unsafe
        
        self.evaluation_results = []
    
    @property
    def chat(self):
        """Access chat completions with safety wrapper"""
        return ChatCompletionsWrapper(
            self.openai_client.chat,
            self.belfrylabs_client,
            self.auto_evaluate,
            self.safety_threshold,
            self.benchmarks,
            self.block_unsafe,
            self.evaluation_results
        )
    
    @property
    def completions(self):
        """Access completions with safety wrapper"""
        return CompletionsWrapper(
            self.openai_client.completions,
            self.belfrylabs_client,
            self.auto_evaluate,
            self.safety_threshold,
            self.benchmarks,
            self.block_unsafe,
            self.evaluation_results
        )
    
    def get_evaluation_results(self) -> List[Dict[str, Any]]:
        """Get all evaluation results"""
        return self.evaluation_results


class ChatCompletionsWrapper:
    """Wrapper for OpenAI chat completions"""
    
    def __init__(
        self,
        chat_api,
        belfrylabs_client,
        auto_evaluate,
        safety_threshold,
        benchmarks,
        block_unsafe,
        evaluation_results
    ):
        self.chat_api = chat_api
        self.belfrylabs_client = belfrylabs_client
        self.auto_evaluate = auto_evaluate
        self.safety_threshold = safety_threshold
        self.benchmarks = benchmarks
        self.block_unsafe = block_unsafe
        self.evaluation_results = evaluation_results
    
    @property
    def completions(self):
        """Access completions"""
        return self
    
    async def create(self, **kwargs):
        """Create chat completion with safety evaluation"""
        
        # Call OpenAI API
        response = await self.chat_api.completions.create(**kwargs)
        
        # Evaluate if enabled
        if self.auto_evaluate:
            await self._evaluate_response(response, kwargs)
        
        return response
    
    async def _evaluate_response(self, response, request_kwargs):
        """Evaluate chat completion response"""
        
        try:
            # Extract response text
            if hasattr(response, 'choices') and response.choices:
                response_text = response.choices[0].message.content
                
                # Evaluate
                result = await self.belfrylabs_client.evaluations.quick_evaluate(
                    text=response_text,
                    benchmarks=self.benchmarks,
                    metadata={
                        "model": request_kwargs.get("model"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
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


class CompletionsWrapper:
    """Wrapper for OpenAI completions"""
    
    def __init__(
        self,
        completions_api,
        belfrylabs_client,
        auto_evaluate,
        safety_threshold,
        benchmarks,
        block_unsafe,
        evaluation_results
    ):
        self.completions_api = completions_api
        self.belfrylabs_client = belfrylabs_client
        self.auto_evaluate = auto_evaluate
        self.safety_threshold = safety_threshold
        self.benchmarks = benchmarks
        self.block_unsafe = block_unsafe
        self.evaluation_results = evaluation_results
    
    async def create(self, **kwargs):
        """Create completion with safety evaluation"""
        
        # Call OpenAI API
        response = await self.completions_api.create(**kwargs)
        
        # Evaluate if enabled
        if self.auto_evaluate:
            await self._evaluate_response(response, kwargs)
        
        return response
    
    async def _evaluate_response(self, response, request_kwargs):
        """Evaluate completion response"""
        
        try:
            # Extract response text
            if hasattr(response, 'choices') and response.choices:
                response_text = response.choices[0].text
                
                # Evaluate
                result = await self.belfrylabs_client.evaluations.quick_evaluate(
                    text=response_text,
                    benchmarks=self.benchmarks,
                    metadata={
                        "model": request_kwargs.get("model"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
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


class OpenAIIntegration:
    """
    High-level OpenAI integration
    
    Usage:
        from belfry_labs.integrations import OpenAIIntegration
        import openai
        
        integration = OpenAIIntegration(
            belfrylabs_api_key="your-belfrylabs-key"
        )
        
        # Wrap OpenAI client
        client = integration.wrap_client(
            openai.OpenAI(api_key="your-openai-key")
        )
        
        # Use normally
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """
    
    def __init__(
        self,
        belfrylabs_api_key: str,
        base_url: str = "https://api.belfrylabs.com/v1",
        tenant_id: Optional[str] = None,
        **kwargs
    ):
        self.belfrylabs_api_key = belfrylabs_api_key
        self.base_url = base_url
        self.tenant_id = tenant_id
        self.kwargs = kwargs
    
    def wrap_client(self, openai_client, **wrapper_kwargs):
        """Wrap an OpenAI client with safety evaluation"""
        
        return BelfryLabsWrapper(
            openai_client,
            belfrylabs_api_key=self.belfrylabs_api_key,
            base_url=self.base_url,
            tenant_id=self.tenant_id,
            **{**self.kwargs, **wrapper_kwargs}
        )
