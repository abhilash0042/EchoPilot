"""
Unified HuggingFace Integration
Combines direct API access with safety evaluation capabilities
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import httpx
import base64

try:
    from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    PreTrainedModel = object
    PreTrainedTokenizer = object

from ..async_client import AsyncBelfryLabsClient
from ..exceptions import BelfryLabsError
from ..utils import logging_utils, metrics_utils

logger = logging.getLogger(__name__)


class UnifiedHuggingFaceClient:
    """
    Unified HuggingFace client combining API access and safety evaluation
    
    This client provides:
    1. Direct access to HuggingFace Inference API
    2. Safety evaluation for all generations
    3. Pipeline wrapping with automatic safety checks
    4. Comprehensive monitoring and metrics
    """
    
    def __init__(
        self,
        huggingface_api_key: str,
        belfrylabs_api_key: str,
        belfrylabs_base_url: str = "https://api.belfrylabs.com/v1",
        tenant_id: Optional[str] = None,
        auto_evaluate: bool = True,
        safety_threshold: float = 0.8,
        benchmarks: Optional[List[str]] = None,
        block_unsafe: bool = False,
        timeout: int = 30,
        retry_attempts: int = 3
    ):
        """
        Initialize unified HuggingFace client
        
        Args:
            huggingface_api_key: HuggingFace API key
            belfrylabs_api_key: Belfry Labs API key
            belfrylabs_base_url: Belfry Labs base URL
            tenant_id: Tenant ID for multi-tenancy
            auto_evaluate: Enable automatic safety evaluation
            safety_threshold: Safety score threshold (0.0-1.0)
            benchmarks: List of benchmarks to run
            block_unsafe: Block outputs below safety threshold
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
        """
        
        self.hf_api_key = huggingface_api_key
        self.base_url = "https://api-inference.huggingface.co/models"
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        
        # Initialize Belfry Labs client
        self.belfrylabs_client = AsyncBelfryLabsClient(
            api_key=belfrylabs_api_key,
            base_url=belfrylabs_base_url,
            tenant_id=tenant_id
        )
        
        # Safety configuration
        self.auto_evaluate = auto_evaluate
        self.safety_threshold = safety_threshold
        self.benchmarks = benchmarks or ["safety", "toxicity", "bias"]
        self.block_unsafe = block_unsafe
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        # Setup logging and metrics
        self.logger = logging_utils.setup_logger(
            name=f"{__name__}.{self.__class__.__name__}",
            level=logging_utils.LogLevel.INFO
        )
        self.metrics = metrics_utils
        
        # Storage for evaluation results
        self.evaluation_results = []
        
        self.logger.info("Unified HuggingFace client initialized")
    
    async def generate_text(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with automatic safety evaluation
        
        Args:
            model: HuggingFace model name
            prompt: Input prompt
            system_prompt: Optional system prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generation result with safety evaluation
        """
        
        self.logger.info(f"Generating text with model {model}")
        self.metrics.increment_counter("text_generations_started")
        
        start_time = datetime.utcnow()
        
        try:
            # Generate text using HuggingFace API
            text_result = await self._call_huggingface_api(
                model=model,
                inputs=prompt,
                system_prompt=system_prompt,
                parameters={
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "return_full_text": False
                }
            )
            
            # Extract generated text
            generated_text = self._extract_text_from_result(text_result)
            
            # Safety evaluation
            safety_result = None
            if self.auto_evaluate:
                safety_result = await self._evaluate_text_safety(
                    prompt=prompt,
                    generated_text=generated_text
                )
                
                # Check safety threshold
                if self._is_unsafe(safety_result):
                    if self.block_unsafe:
                        raise BelfryLabsError(
                            f"Generated text blocked due to safety concerns: "
                            f"score {safety_result.get('score', 0)}%"
                        )
                    else:
                        self.logger.warning(
                            f"Generated text below safety threshold: "
                            f"score {safety_result.get('score', 0)}%"
                        )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.record_metric("text_generation_duration", duration)
            self.metrics.increment_counter("text_generations_completed")
            
            return {
                "generated_text": generated_text,
                "model": model,
                "prompt": prompt,
                "safety_evaluation": safety_result,
                "generation_time": duration,
                "metadata": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            self.metrics.increment_counter("text_generations_failed")
            self.logger.error(f"Text generation failed: {str(e)}")
            raise
    
    async def generate_image(
        self,
        model: str,
        prompt: str,
        width: int = 512,
        height: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image with safety evaluation
        
        Args:
            model: HuggingFace model name
            prompt: Image generation prompt
            width: Image width
            height: Image height
            
        Returns:
            Generation result with safety evaluation
        """
        
        self.logger.info(f"Generating image with model {model}")
        self.metrics.increment_counter("image_generations_started")
        
        start_time = datetime.utcnow()
        
        try:
            # Generate image using HuggingFace API
            image_result = await self._call_huggingface_api(
                model=model,
                inputs=prompt,
                parameters={
                    "width": width,
                    "height": height
                }
            )
            
            # Convert to base64
            image_data = image_result if isinstance(image_result, bytes) else image_result
            image_b64 = base64.b64encode(image_data).decode()
            image_url = f"data:image/png;base64,{image_b64}"
            
            # Safety evaluation for image prompt
            safety_result = None
            if self.auto_evaluate:
                safety_result = await self._evaluate_text_safety(
                    prompt=prompt,
                    generated_text=prompt  # Evaluate the prompt itself
                )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.record_metric("image_generation_duration", duration)
            self.metrics.increment_counter("image_generations_completed")
            
            return {
                "image_url": image_url,
                "image_data": image_data,
                "model": model,
                "prompt": prompt,
                "safety_evaluation": safety_result,
                "generation_time": duration,
                "metadata": {
                    "width": width,
                    "height": height,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            self.metrics.increment_counter("image_generations_failed")
            self.logger.error(f"Image generation failed: {str(e)}")
            raise
    
    async def process_audio(
        self,
        model: str,
        audio_data: bytes,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process audio with safety evaluation
        
        Args:
            model: HuggingFace model name
            audio_data: Audio data bytes
            task: Processing task (transcribe, classify, etc.)
            
        Returns:
            Processing result with safety evaluation
        """
        
        self.logger.info(f"Processing audio with model {model}")
        self.metrics.increment_counter("audio_processing_started")
        
        start_time = datetime.utcnow()
        
        try:
            # Process audio using HuggingFace API
            audio_result = await self._call_huggingface_api(
                model=model,
                inputs=audio_data,
                task=task
            )
            
            # Extract text from result
            processed_text = self._extract_text_from_result(audio_result)
            
            # Safety evaluation
            safety_result = None
            if self.auto_evaluate and processed_text:
                safety_result = await self._evaluate_text_safety(
                    prompt=f"Audio {task}",
                    generated_text=processed_text
                )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.record_metric("audio_processing_duration", duration)
            self.metrics.increment_counter("audio_processing_completed")
            
            return {
                "processed_text": processed_text,
                "model": model,
                "task": task,
                "safety_evaluation": safety_result,
                "processing_time": duration,
                "metadata": {
                    "audio_size": len(audio_data),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            self.metrics.increment_counter("audio_processing_failed")
            self.logger.error(f"Audio processing failed: {str(e)}")
            raise
    
    async def evaluate_model(
        self,
        model: Union[str, PreTrainedModel],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        test_prompts: Optional[List[str]] = None,
        benchmarks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a HuggingFace model for safety
        
        Args:
            model: Model name or HuggingFace model object
            tokenizer: Optional tokenizer
            test_prompts: Test prompts for evaluation
            benchmarks: Benchmarks to run
            
        Returns:
            Model evaluation results
        """
        
        self.logger.info(f"Evaluating model: {model}")
        self.metrics.increment_counter("model_evaluations_started")
        
        start_time = datetime.utcnow()
        
        try:
            # Get model info
            if isinstance(model, str):
                model_name = model
                model_info = {"model_name": model_name}
            else:
                model_name = getattr(model, 'name_or_path', 'unknown')
                model_info = {
                    "model_name": model_name,
                    "model_type": getattr(model.config, 'model_type', 'unknown'),
                    "num_parameters": sum(p.numel() for p in model.parameters())
                }
            
            # Create evaluation
            evaluation = await self.belfrylabs_client.evaluations.create(
                name=f"HuggingFace Model Evaluation - {model_name}",
                model_id=model_name,
                benchmarks=benchmarks or self.benchmarks,
                metadata={
                    **model_info,
                    "framework": "huggingface",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.record_metric("model_evaluation_duration", duration)
            self.metrics.increment_counter("model_evaluations_completed")
            
            return evaluation
            
        except Exception as e:
            self.metrics.increment_counter("model_evaluations_failed")
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def wrap_pipeline(self, pipeline_func: Callable) -> Callable:
        """
        Wrap a HuggingFace pipeline with safety evaluation
        
        Args:
            pipeline_func: HuggingFace pipeline function
            
        Returns:
            Wrapped pipeline function
        """
        
        async def wrapped_pipeline(*args, **kwargs):
            # Call original pipeline
            result = pipeline_func(*args, **kwargs)
            
            # Evaluate if enabled
            if self.auto_evaluate:
                await self._evaluate_pipeline_result(args, result)
            
            return result
        
        return wrapped_pipeline
    
    async def _call_huggingface_api(
        self,
        model: str,
        inputs: Union[str, bytes],
        parameters: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None
    ) -> Any:
        """Call HuggingFace Inference API"""
        
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json"
        }
        
        if isinstance(inputs, str):
            payload = {
                "inputs": inputs,
                "parameters": parameters or {}
            }
        else:
            # For binary data (audio, images)
            headers["Content-Type"] = "application/octet-stream"
            payload = inputs
        
        url = f"{self.base_url}/{model}"
        if task:
            url += f"?task={task}"
        
        for attempt in range(self.retry_attempts):
            try:
                if isinstance(inputs, str):
                    response = await self.client.post(
                        url,
                        json=payload,
                        headers=headers
                    )
                else:
                    response = await self.client.post(
                        url,
                        content=payload,
                        headers=headers
                    )
                
                response.raise_for_status()
                return response.json() if isinstance(inputs, str) else response.content
                
            except httpx.HTTPStatusError as e:
                if attempt == self.retry_attempts - 1:
                    self.logger.error(f"HuggingFace API error: {e.response.status_code}")
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    self.logger.error(f"Error calling HuggingFace API: {str(e)}")
                    raise
                await asyncio.sleep(2 ** attempt)
    
    def _extract_text_from_result(self, result: Any) -> str:
        """Extract text from HuggingFace API result"""
        
        if isinstance(result, list) and len(result) > 0:
            if "generated_text" in result[0]:
                return result[0]["generated_text"]
            elif "text" in result[0]:
                return result[0]["text"]
        elif isinstance(result, dict):
            if "generated_text" in result:
                return result["generated_text"]
            elif "text" in result:
                return result["text"]
        
        return str(result)
    
    async def _evaluate_text_safety(
        self,
        prompt: str,
        generated_text: str
    ) -> Dict[str, Any]:
        """Evaluate text for safety"""
        
        try:
            result = await self.belfrylabs_client.evaluations.quick_evaluate(
                text=generated_text,
                benchmarks=self.benchmarks,
                metadata={
                    "prompt": prompt,
                    "framework": "huggingface",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            self.evaluation_results.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Safety evaluation failed: {str(e)}")
            return None
    
    def _is_unsafe(self, safety_result: Optional[Dict[str, Any]]) -> bool:
        """Check if result is unsafe based on safety threshold"""
        
        if not safety_result:
            return False
        
        score = safety_result.get("score", 100.0)
        return score < (self.safety_threshold * 100)
    
    async def _evaluate_pipeline_result(self, inputs, outputs):
        """Evaluate pipeline outputs"""
        
        try:
            # Extract text from outputs
            if isinstance(outputs, list):
                for output in outputs:
                    if isinstance(output, dict) and 'generated_text' in output:
                        text = output['generated_text']
                        await self._evaluate_text_safety("Pipeline output", text)
            elif isinstance(outputs, dict) and 'generated_text' in outputs:
                text = outputs['generated_text']
                await self._evaluate_text_safety("Pipeline output", text)
        
        except Exception as e:
            self.logger.error(f"Pipeline evaluation failed: {str(e)}")
    
    def get_evaluation_results(self) -> List[Dict[str, Any]]:
        """Get all evaluation results"""
        return self.evaluation_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics"""
        return self.metrics.get_all_metrics()
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()
        await self.belfrylabs_client.close()
        self.logger.info("Unified HuggingFace client closed")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
