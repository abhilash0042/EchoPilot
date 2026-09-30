"""
Hugging Face Integration for Belfry Labs
Seamless safety evaluation for Hugging Face models
"""

from typing import Any, Dict, List, Optional, Union
import asyncio
import logging
from datetime import datetime

try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    PreTrainedModel = object
    PreTrainedTokenizer = object

from belfry_labs.async_client import AsyncBelfryLabsClient
from belfry_labs.exceptions import BelfryLabsError

logger = logging.getLogger(__name__)


class BelfryLabsEvaluator:
    """
    Evaluator for Hugging Face models
    
    Usage:
        from transformers import AutoModel, AutoTokenizer
        from belfry_labs.integrations import BelfryLabsEvaluator
        
        model = AutoModel.from_pretrained("microsoft/DialoGPT-medium")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        
        evaluator = BelfryLabsEvaluator(api_key="your-key")
        safety_score = await evaluator.evaluate_model(model, tokenizer)
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.belfrylabs.com/v1",
        tenant_id: Optional[str] = None,
        **kwargs
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers is not installed. "
                "Install it with: pip install transformers"
            )
        
        self.client = AsyncBelfryLabsClient(
            api_key=api_key,
            base_url=base_url,
            tenant_id=tenant_id
        )
        self.kwargs = kwargs
    
    async def evaluate_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        benchmarks: Optional[List[str]] = None,
        test_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a Hugging Face model for safety
        
        Args:
            model: Hugging Face model
            tokenizer: Hugging Face tokenizer
            benchmarks: List of benchmarks to run
            test_prompts: Optional test prompts
            
        Returns:
            Evaluation results
        """
        
        benchmarks = benchmarks or ["safety", "toxicity", "bias"]
        
        # Get model info
        model_info = {
            "model_name": model.name_or_path if hasattr(model, 'name_or_path') else "unknown",
            "model_type": model.config.model_type if hasattr(model, 'config') else "unknown",
            "num_parameters": sum(p.numel() for p in model.parameters())
        }
        
        # Create evaluation
        try:
            evaluation = await self.client.evaluations.create(
                name=f"HuggingFace Model Evaluation - {model_info['model_name']}",
                model_id=model_info['model_name'],
                benchmarks=benchmarks,
                metadata={
                    **model_info,
                    "framework": "huggingface",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            return evaluation
        
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    async def evaluate_generation(
        self,
        prompt: str,
        generated_text: str,
        benchmarks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single generation
        
        Args:
            prompt: Input prompt
            generated_text: Generated text
            benchmarks: Benchmarks to run
            
        Returns:
            Evaluation results
        """
        
        benchmarks = benchmarks or ["toxicity", "bias", "safety"]
        
        try:
            result = await self.client.evaluations.quick_evaluate(
                text=generated_text,
                benchmarks=benchmarks,
                metadata={
                    "prompt": prompt,
                    "framework": "huggingface",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Generation evaluation failed: {str(e)}")
            raise


class HuggingFaceIntegration:
    """
    High-level Hugging Face integration
    
    Usage:
        from belfry_labs.integrations import HuggingFaceIntegration
        from transformers import pipeline
        
        integration = HuggingFaceIntegration(api_key="your-key")
        
        # Wrap a pipeline
        safe_pipeline = integration.wrap_pipeline(
            pipeline("text-generation", model="gpt2")
        )
        
        # Use normally - outputs are automatically evaluated
        result = safe_pipeline("Your prompt")
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
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers is not installed. "
                "Install it with: pip install transformers"
            )
        
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
    
    def wrap_pipeline(self, pipeline):
        """Wrap a Hugging Face pipeline with safety evaluation"""
        
        original_call = pipeline.__call__
        
        async def wrapped_call(*args, **kwargs):
            # Call original pipeline
            result = original_call(*args, **kwargs)
            
            # Evaluate if enabled
            if self.auto_evaluate:
                await self._evaluate_result(args, result)
            
            return result
        
        pipeline.__call__ = wrapped_call
        return pipeline
    
    async def _evaluate_result(self, inputs, outputs):
        """Evaluate pipeline outputs"""
        
        try:
            # Extract text from outputs
            if isinstance(outputs, list):
                for output in outputs:
                    if isinstance(output, dict) and 'generated_text' in output:
                        text = output['generated_text']
                        await self._evaluate_text(text)
            elif isinstance(outputs, dict) and 'generated_text' in outputs:
                text = outputs['generated_text']
                await self._evaluate_text(text)
        
        except Exception as e:
            logger.error(f"Result evaluation failed: {str(e)}")
    
    async def _evaluate_text(self, text: str):
        """Evaluate generated text"""
        
        try:
            result = await self.client.evaluations.quick_evaluate(
                text=text,
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
                        f"Output blocked due to safety concerns: "
                        f"score {result.get('score')}%"
                    )
        
        except Exception as e:
            logger.error(f"Text evaluation failed: {str(e)}")
    
    def get_evaluation_results(self) -> List[Dict[str, Any]]:
        """Get all evaluation results"""
        return self.evaluation_results
    
    def create_evaluator(self) -> BelfryLabsEvaluator:
        """Create a model evaluator"""
        
        return BelfryLabsEvaluator(
            api_key=self.client.api_key,
            base_url=self.client.base_url,
            tenant_id=self.client.tenant_id
        )
