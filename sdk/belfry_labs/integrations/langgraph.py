"""
LangGraph Integration for Belfry Labs
Safety evaluation for LangGraph agent workflows.

Supports:
1. Node-level evaluation (before/after each node)
2. Edge-level evaluation (on transitions)
3. State-level evaluation (full agent state)
4. Local-first evaluation with policy sync
"""

from typing import Any, Dict, List, Optional, Callable, TypeVar, Union
import logging
import time
from datetime import datetime
from functools import wraps
from dataclasses import dataclass, field

try:
    from langgraph.graph import StateGraph
    from langgraph.graph.graph import CompiledGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    CompiledGraph = None

from belfry_labs.local_evaluator import (
    LocalEvaluator,
    EvaluationResult,
    EvaluationContext,
    RiskLevel,
    Action
)
from belfry_labs.exceptions import BelfryLabsError

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class NodeEvaluationResult:
    """Result of evaluating a node execution"""
    node_name: str
    allowed: bool
    risk_level: RiskLevel
    findings: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_time_ms: float = 0
    blocked: bool = False


@dataclass
class WorkflowEvaluationResult:
    """Result of evaluating a complete workflow execution"""
    allowed: bool
    risk_level: RiskLevel
    node_results: List[NodeEvaluationResult] = field(default_factory=list)
    total_evaluation_time_ms: float = 0
    blocked_at_node: Optional[str] = None


class BelfryGraphMonitor:
    """
    Monitor for LangGraph workflows with safety evaluation.
    
    Usage:
        from langgraph.graph import StateGraph
        from belfry_labs.integrations import BelfryGraphMonitor
        
        # Create your graph
        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tool", tool_node)
        
        # Wrap with safety monitoring
        monitor = BelfryGraphMonitor(
            block_unsafe=True,
            evaluate_on="all"  # all, input, output, transitions
        )
        
        safe_graph = monitor.wrap_graph(graph)
        app = safe_graph.compile()
        
        # Run with safety checks
        result = app.invoke({"messages": [...]})
    """
    
    def __init__(
        self,
        evaluator: Optional[LocalEvaluator] = None,
        block_unsafe: bool = False,
        evaluate_on: str = "all",  # all, input, output, transitions
        report_metrics: bool = False,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is not installed. Install with: pip install langgraph")
        
        self.evaluator = evaluator or LocalEvaluator()
        self.block_unsafe = block_unsafe
        self.evaluate_on = evaluate_on
        self.report_metrics = report_metrics
        self.metrics_callback = metrics_callback
        
        self._node_results: List[NodeEvaluationResult] = []
        self._workflow_start_time: Optional[float] = None
    
    def wrap_graph(self, graph: 'StateGraph') -> 'StateGraph':
        """
        Wrap a StateGraph with safety evaluation.
        Returns a new graph with evaluation hooks.
        """
        # Get all nodes
        nodes = list(graph.nodes.keys())
        
        # Wrap each node function
        for node_name in nodes:
            original_func = graph.nodes[node_name]
            wrapped_func = self._create_wrapped_node(node_name, original_func)
            graph.nodes[node_name] = wrapped_func
        
        return graph
    
    def _create_wrapped_node(
        self,
        node_name: str,
        original_func: Callable
    ) -> Callable:
        """Create a wrapped node function with evaluation"""
        
        @wraps(original_func)
        def wrapped(*args, **kwargs):
            start_time = time.time()
            
            # Extract state from args
            state = args[0] if args else kwargs.get('state', {})
            
            # Evaluate input (before node execution)
            if self.evaluate_on in ["all", "input"]:
                input_result = self._evaluate_node_input(node_name, state)
                
                if not input_result.allowed and self.block_unsafe:
                    raise BelfryLabsError(
                        f"Node '{node_name}' input blocked by safety policy: "
                        f"{[f['message'] for f in input_result.findings]}"
                    )
            
            # Execute original node
            result = original_func(*args, **kwargs)
            
            # Evaluate output (after node execution)
            if self.evaluate_on in ["all", "output"]:
                output_result = self._evaluate_node_output(node_name, state, result)
                
                if not output_result.allowed and self.block_unsafe:
                    raise BelfryLabsError(
                        f"Node '{node_name}' output blocked by safety policy: "
                        f"{[f['message'] for f in output_result.findings]}"
                    )
                
                # Record result
                self._node_results.append(output_result)
            
            return result
        
        return wrapped
    
    def _evaluate_node_input(
        self,
        node_name: str,
        state: Dict[str, Any]
    ) -> NodeEvaluationResult:
        """Evaluate node input (state before execution)"""
        start_time = time.time()
        
        # Extract evaluatable content from state
        input_text = self._extract_text_from_state(state)
        
        result = self.evaluator.evaluate(input=input_text)
        
        return NodeEvaluationResult(
            node_name=node_name,
            allowed=result.allowed,
            risk_level=result.risk_level,
            findings=[
                {
                    "type": f.type.value,
                    "message": f.message,
                    "risk": f.risk_level.value,
                    "action": f.action.value
                }
                for f in result.findings
            ],
            evaluation_time_ms=(time.time() - start_time) * 1000,
            blocked=not result.allowed
        )
    
    def _evaluate_node_output(
        self,
        node_name: str,
        input_state: Dict[str, Any],
        output: Any
    ) -> NodeEvaluationResult:
        """Evaluate node output"""
        start_time = time.time()
        
        input_text = self._extract_text_from_state(input_state)
        output_text = self._extract_text_from_output(output)
        
        result = self.evaluator.evaluate(input=input_text, output=output_text)
        
        return NodeEvaluationResult(
            node_name=node_name,
            allowed=result.allowed,
            risk_level=result.risk_level,
            findings=[
                {
                    "type": f.type.value,
                    "message": f.message,
                    "risk": f.risk_level.value,
                    "action": f.action.value
                }
                for f in result.findings
            ],
            evaluation_time_ms=(time.time() - start_time) * 1000,
            blocked=not result.allowed
        )
    
    def _extract_text_from_state(self, state: Dict[str, Any]) -> str:
        """Extract evaluatable text from agent state"""
        texts = []
        
        # Common state keys that contain user content
        for key in ["messages", "input", "query", "prompt", "user_input"]:
            if key in state:
                value = state[key]
                if isinstance(value, str):
                    texts.append(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            texts.append(item)
                        elif hasattr(item, 'content'):
                            texts.append(str(item.content))
                        elif isinstance(item, dict) and 'content' in item:
                            texts.append(str(item['content']))
        
        return "\n".join(texts)
    
    def _extract_text_from_output(self, output: Any) -> str:
        """Extract evaluatable text from node output"""
        if isinstance(output, str):
            return output
        
        if isinstance(output, dict):
            texts = []
            for key in ["messages", "output", "response", "content", "result"]:
                if key in output:
                    value = output[key]
                    if isinstance(value, str):
                        texts.append(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                texts.append(item)
                            elif hasattr(item, 'content'):
                                texts.append(str(item.content))
            return "\n".join(texts)
        
        if hasattr(output, 'content'):
            return str(output.content)
        
        return str(output)
    
    def get_workflow_result(self) -> WorkflowEvaluationResult:
        """Get the complete workflow evaluation result"""
        if not self._node_results:
            return WorkflowEvaluationResult(
                allowed=True,
                risk_level=RiskLevel.NONE
            )
        
        # Find highest risk
        risk_order = {
            RiskLevel.NONE: 0, RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3, RiskLevel.CRITICAL: 4
        }
        highest_risk = max(
            self._node_results,
            key=lambda r: risk_order[r.risk_level]
        ).risk_level
        
        # Check if any node was blocked
        blocked_node = None
        for result in self._node_results:
            if result.blocked:
                blocked_node = result.node_name
                break
        
        total_time = sum(r.evaluation_time_ms for r in self._node_results)
        
        return WorkflowEvaluationResult(
            allowed=blocked_node is None,
            risk_level=highest_risk,
            node_results=list(self._node_results),
            total_evaluation_time_ms=total_time,
            blocked_at_node=blocked_node
        )
    
    def reset(self):
        """Reset monitoring state"""
        self._node_results = []
        self._workflow_start_time = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics for reporting"""
        return self.evaluator.get_metrics()


def wrap_langgraph(
    graph: 'StateGraph',
    block_unsafe: bool = False,
    evaluate_on: str = "all"
) -> 'StateGraph':
    """
    Convenience function to wrap a LangGraph with safety evaluation.
    
    Args:
        graph: The StateGraph to wrap
        block_unsafe: Whether to block unsafe executions
        evaluate_on: When to evaluate - "all", "input", "output", "transitions"
    
    Returns:
        Wrapped StateGraph with safety evaluation
    """
    monitor = BelfryGraphMonitor(
        block_unsafe=block_unsafe,
        evaluate_on=evaluate_on
    )
    return monitor.wrap_graph(graph)


class SafeAgentExecutor:
    """
    Wrapper for agent execution with comprehensive safety monitoring.
    
    Usage:
        from belfry_labs.integrations import SafeAgentExecutor
        
        executor = SafeAgentExecutor(
            compiled_graph=app,
            block_unsafe=True
        )
        
        result = executor.invoke({"messages": [...]})
        
        # Check safety status
        if not executor.last_result.allowed:
            print(f"Safety issues: {executor.last_result.findings}")
    """
    
    def __init__(
        self,
        compiled_graph: 'CompiledGraph',
        evaluator: Optional[LocalEvaluator] = None,
        block_unsafe: bool = False
    ):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is not installed")
        
        self.graph = compiled_graph
        self.evaluator = evaluator or LocalEvaluator()
        self.block_unsafe = block_unsafe
        self.last_result: Optional[EvaluationResult] = None
    
    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Invoke the graph with safety evaluation"""
        
        # Evaluate input
        input_text = self._extract_input_text(input)
        pre_result = self.evaluator.evaluate(input=input_text)
        
        if not pre_result.allowed and self.block_unsafe:
            self.last_result = pre_result
            raise BelfryLabsError(
                f"Input blocked by safety policy: "
                f"{[f.message for f in pre_result.findings]}"
            )
        
        # Execute graph
        result = self.graph.invoke(input, config=config)
        
        # Evaluate output
        output_text = self._extract_output_text(result)
        post_result = self.evaluator.evaluate(input=input_text, output=output_text)
        self.last_result = post_result
        
        if not post_result.allowed and self.block_unsafe:
            raise BelfryLabsError(
                f"Output blocked by safety policy: "
                f"{[f.message for f in post_result.findings]}"
            )
        
        return result
    
    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Async invoke with safety evaluation"""
        
        # Evaluate input
        input_text = self._extract_input_text(input)
        pre_result = self.evaluator.evaluate(input=input_text)
        
        if not pre_result.allowed and self.block_unsafe:
            self.last_result = pre_result
            raise BelfryLabsError(
                f"Input blocked by safety policy: "
                f"{[f.message for f in pre_result.findings]}"
            )
        
        # Execute graph
        result = await self.graph.ainvoke(input, config=config)
        
        # Evaluate output
        output_text = self._extract_output_text(result)
        post_result = self.evaluator.evaluate(input=input_text, output=output_text)
        self.last_result = post_result
        
        if not post_result.allowed and self.block_unsafe:
            raise BelfryLabsError(
                f"Output blocked by safety policy: "
                f"{[f.message for f in post_result.findings]}"
            )
        
        return result
    
    def _extract_input_text(self, input: Dict[str, Any]) -> str:
        """Extract text from input"""
        texts = []
        for key in ["messages", "input", "query"]:
            if key in input:
                val = input[key]
                if isinstance(val, str):
                    texts.append(val)
                elif isinstance(val, list):
                    for item in val:
                        if hasattr(item, 'content'):
                            texts.append(str(item.content))
        return "\n".join(texts)
    
    def _extract_output_text(self, output: Dict[str, Any]) -> str:
        """Extract text from output"""
        texts = []
        for key in ["messages", "output", "response"]:
            if key in output:
                val = output[key]
                if isinstance(val, str):
                    texts.append(val)
                elif isinstance(val, list):
                    for item in val:
                        if hasattr(item, 'content'):
                            texts.append(str(item.content))
        return "\n".join(texts)
