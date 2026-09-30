"""
Belfry Labs Integrations
Comprehensive integrations for AI frameworks and safety evaluation

Supports:
- HuggingFace: Model evaluation and safety checks
- LangChain: Callback handlers for LLM safety
- LangGraph: Agent workflow monitoring
- Local-first evaluation for enterprise deployments
"""

from .huggingface import (
    BelfryLabsEvaluator,
    HuggingFaceIntegration
)
from .unified_huggingface import UnifiedHuggingFaceClient
from .langchain import (
    BelfryLabsCallbackHandler,
    LangChainIntegration,
    LocalBelfryLabsCallbackHandler,
    LocalLangChainIntegration
)
from .langgraph import (
    BelfryGraphMonitor,
    SafeAgentExecutor,
    wrap_langgraph,
    NodeEvaluationResult,
    WorkflowEvaluationResult
)

# Re-export main classes
__all__ = [
    # HuggingFace
    'BelfryLabsEvaluator',
    'HuggingFaceIntegration',
    'UnifiedHuggingFaceClient',
    # LangChain (cloud mode)
    'BelfryLabsCallbackHandler',
    'LangChainIntegration',
    # LangChain (local mode)
    'LocalBelfryLabsCallbackHandler',
    'LocalLangChainIntegration',
    # LangGraph
    'BelfryGraphMonitor',
    'SafeAgentExecutor',
    'wrap_langgraph',
    'NodeEvaluationResult',
    'WorkflowEvaluationResult',
]

# Version information
__version__ = "2.1.0"
__description__ = "AI framework integrations for Belfry Labs"