"""
Belfry Labs Local Evaluator
Performs safety evaluations locally with policies synced from Belfry Cloud.
Enterprise feature: Evaluate locally, report only aggregated metrics.
"""

import re
import hashlib
import json
import time
import threading
from typing import List, Optional, Dict, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# ============== Enums ==============

class EvaluationType(str, Enum):
    """Types of safety evaluations"""
    PII_DETECTION = "pii_detection"
    TOXICITY = "toxicity"
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    CONTENT_POLICY = "content_policy"
    OUTPUT_VALIDATION = "output_validation"
    HALLUCINATION = "hallucination"
    BIAS = "bias"


class RiskLevel(str, Enum):
    """Risk levels for findings"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(str, Enum):
    """Actions to take based on evaluation"""
    ALLOW = "allow"
    LOG = "log"
    WARN = "warn"
    BLOCK = "block"


# ============== Data Classes ==============

@dataclass
class Finding:
    """A single finding from evaluation"""
    type: EvaluationType
    risk_level: RiskLevel
    message: str
    action: Action
    location: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of a local evaluation"""
    allowed: bool
    risk_level: RiskLevel
    findings: List[Finding] = field(default_factory=list)
    evaluation_time_ms: float = 0
    evaluators_run: List[str] = field(default_factory=list)
    
    @property
    def should_block(self) -> bool:
        return any(f.action == Action.BLOCK for f in self.findings)
    
    @property
    def highest_risk(self) -> RiskLevel:
        if not self.findings:
            return RiskLevel.NONE
        risk_order = {
            RiskLevel.NONE: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }
        return max(self.findings, key=lambda f: risk_order[f.risk_level]).risk_level


@dataclass
class EvaluationContext:
    """Context for evaluation"""
    input: str = ""
    output: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============== Evaluator Base ==============

class BaseEvaluator(ABC):
    """Base class for local evaluators"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Evaluator name"""
        pass
    
    @property
    @abstractmethod
    def evaluation_type(self) -> EvaluationType:
        """Type of evaluation performed"""
        pass
    
    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> List[Finding]:
        """Perform evaluation and return findings"""
        pass


# ============== Built-in Evaluators ==============

class PIIDetector(BaseEvaluator):
    """Detects PII in input/output"""
    
    name = "pii_detector"
    evaluation_type = EvaluationType.PII_DETECTION
    
    PATTERNS = {
        "email": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", RiskLevel.MEDIUM),
        "phone": (r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b", RiskLevel.MEDIUM),
        "ssn": (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", RiskLevel.HIGH),
        "credit_card": (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", RiskLevel.CRITICAL),
        "ip_address": (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", RiskLevel.LOW),
    }
    
    def __init__(self, action: Action = Action.WARN):
        self.default_action = action
    
    def evaluate(self, context: EvaluationContext) -> List[Finding]:
        findings = []
        
        for text, location in [
            (context.input, "input"),
            (context.output, "output")
        ]:
            for pii_type, (pattern, risk) in self.PATTERNS.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    findings.append(Finding(
                        type=self.evaluation_type,
                        risk_level=risk,
                        message=f"Detected {pii_type} in {location}",
                        action=self.default_action if risk != RiskLevel.CRITICAL else Action.BLOCK,
                        location=location,
                        metadata={"pii_type": pii_type, "count": len(matches)}
                    ))
        
        return findings


class PromptInjectionDetector(BaseEvaluator):
    """Detects prompt injection attempts"""
    
    name = "prompt_injection_detector"
    evaluation_type = EvaluationType.PROMPT_INJECTION
    
    INJECTION_PHRASES = [
        "ignore previous instructions",
        "ignore all previous",
        "disregard your instructions",
        "forget your instructions",
        "new instructions:",
        "system prompt:",
        "you are now",
        "pretend you are",
        "act as if",
        "jailbreak",
        "dan mode",
        "developer mode",
        "ignore your training",
        "override your",
        "bypass your",
    ]
    
    def __init__(self, action: Action = Action.BLOCK):
        self.default_action = action
    
    def evaluate(self, context: EvaluationContext) -> List[Finding]:
        findings = []
        input_lower = context.input.lower()
        
        for phrase in self.INJECTION_PHRASES:
            if phrase in input_lower:
                findings.append(Finding(
                    type=self.evaluation_type,
                    risk_level=RiskLevel.HIGH,
                    message=f"Potential prompt injection detected",
                    action=self.default_action,
                    location="input",
                    metadata={"matched_phrase": phrase}
                ))
                break  # One finding is enough
        
        return findings


class JailbreakDetector(BaseEvaluator):
    """Detects jailbreak attempts"""
    
    name = "jailbreak_detector"
    evaluation_type = EvaluationType.JAILBREAK
    
    JAILBREAK_PATTERNS = [
        r"(?:do\s+)?anything\s+now",  # DAN
        r"maximum\s*(?:virtual)?\s*machine",  # MVM
        r"evil\s+(?:mode|assistant)",
        r"unfiltered\s+(?:mode|response)",
        r"no\s+restrictions",
        r"bypass\s+(?:safety|filter|moderation)",
        r"hypothetical\s+scenario",
        r"roleplay\s+as\s+(?:an?\s+)?(?:evil|malicious)",
    ]
    
    def __init__(self, action: Action = Action.BLOCK):
        self.default_action = action
    
    def evaluate(self, context: EvaluationContext) -> List[Finding]:
        findings = []
        input_lower = context.input.lower()
        
        for pattern in self.JAILBREAK_PATTERNS:
            if re.search(pattern, input_lower):
                findings.append(Finding(
                    type=self.evaluation_type,
                    risk_level=RiskLevel.HIGH,
                    message="Potential jailbreak attempt detected",
                    action=self.default_action,
                    location="input",
                    metadata={"pattern": pattern}
                ))
                break
        
        return findings


class ToxicityDetector(BaseEvaluator):
    """Basic toxicity detection using keyword matching"""
    
    name = "toxicity_detector"
    evaluation_type = EvaluationType.TOXICITY
    
    # Note: In production, use a proper ML-based toxicity classifier
    TOXIC_PATTERNS = [
        r"\b(?:kill|murder|harm)\s+(?:yourself|myself|people)\b",
        r"\b(?:hate|despise)\s+(?:all|every)\b",
        r"\b(?:how\s+to|instructions\s+for)\s+(?:make|build)\s+(?:a\s+)?(?:bomb|weapon)\b",
    ]
    
    def __init__(self, action: Action = Action.BLOCK):
        self.default_action = action
    
    def evaluate(self, context: EvaluationContext) -> List[Finding]:
        findings = []
        
        for text, location in [
            (context.input, "input"),
            (context.output, "output")
        ]:
            text_lower = text.lower()
            for pattern in self.TOXIC_PATTERNS:
                if re.search(pattern, text_lower):
                    findings.append(Finding(
                        type=self.evaluation_type,
                        risk_level=RiskLevel.CRITICAL,
                        message=f"Potentially harmful content detected in {location}",
                        action=self.default_action,
                        location=location
                    ))
                    break
        
        return findings


class OutputLengthValidator(BaseEvaluator):
    """Validates output length"""
    
    name = "output_length_validator"
    evaluation_type = EvaluationType.OUTPUT_VALIDATION
    
    def __init__(
        self,
        max_length: int = 10000,
        action: Action = Action.WARN
    ):
        self.max_length = max_length
        self.default_action = action
    
    def evaluate(self, context: EvaluationContext) -> List[Finding]:
        findings = []
        
        if len(context.output) > self.max_length:
            findings.append(Finding(
                type=self.evaluation_type,
                risk_level=RiskLevel.LOW,
                message=f"Output exceeds maximum length ({len(context.output)} > {self.max_length})",
                action=self.default_action,
                location="output",
                metadata={"length": len(context.output), "max": self.max_length}
            ))
        
        return findings


# ============== Local Evaluator Engine ==============

class LocalEvaluator:
    """
    Local evaluation engine for enterprise deployments.
    Evaluates content locally using synced policies and built-in evaluators.
    Only reports aggregated metrics to Belfry Cloud.
    
    Usage:
        evaluator = LocalEvaluator()
        
        # Add evaluators
        evaluator.add_evaluator(PIIDetector())
        evaluator.add_evaluator(PromptInjectionDetector())
        
        # Evaluate
        result = evaluator.evaluate(
            input="user message",
            output="model response"
        )
        
        if not result.allowed:
            print(f"Blocked: {result.findings}")
    """
    
    def __init__(self):
        self._evaluators: Dict[str, BaseEvaluator] = {}
        self._enabled_evaluators: Set[str] = set()
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "total_evaluations": 0,
            "blocked": 0,
            "warned": 0,
            "allowed": 0,
            "by_type": {},
            "by_risk": {}
        }
        
        # Add default evaluators
        self._setup_default_evaluators()
    
    def _setup_default_evaluators(self):
        """Setup default evaluators"""
        self.add_evaluator(PIIDetector())
        self.add_evaluator(PromptInjectionDetector())
        self.add_evaluator(JailbreakDetector())
        self.add_evaluator(ToxicityDetector())
        self.add_evaluator(OutputLengthValidator())
    
    def add_evaluator(self, evaluator: BaseEvaluator, enabled: bool = True):
        """Add an evaluator"""
        self._evaluators[evaluator.name] = evaluator
        if enabled:
            self._enabled_evaluators.add(evaluator.name)
    
    def remove_evaluator(self, name: str):
        """Remove an evaluator"""
        if name in self._evaluators:
            del self._evaluators[name]
            self._enabled_evaluators.discard(name)
    
    def enable_evaluator(self, name: str):
        """Enable an evaluator"""
        if name in self._evaluators:
            self._enabled_evaluators.add(name)
    
    def disable_evaluator(self, name: str):
        """Disable an evaluator"""
        self._enabled_evaluators.discard(name)
    
    def evaluate(
        self,
        input: str = "",
        output: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate content against all enabled evaluators.
        
        Args:
            input: User input / prompt
            output: Model output / response
            conversation_history: Previous conversation turns
            model_name: Name of the model being evaluated
            metadata: Additional metadata
        
        Returns:
            EvaluationResult with findings and recommendation
        """
        start_time = time.time()
        
        context = EvaluationContext(
            input=input,
            output=output,
            conversation_history=conversation_history or [],
            model_name=model_name,
            metadata=metadata or {}
        )
        
        all_findings: List[Finding] = []
        evaluators_run: List[str] = []
        
        for name in self._enabled_evaluators:
            evaluator = self._evaluators.get(name)
            if not evaluator:
                continue
            
            try:
                findings = evaluator.evaluate(context)
                all_findings.extend(findings)
                evaluators_run.append(name)
            except Exception as e:
                logger.warning(f"Evaluator {name} failed: {e}")
        
        evaluation_time_ms = (time.time() - start_time) * 1000
        
        # Determine overall result
        should_block = any(f.action == Action.BLOCK for f in all_findings)
        has_warnings = any(f.action == Action.WARN for f in all_findings)
        
        if all_findings:
            highest_risk = max(
                all_findings,
                key=lambda f: {
                    RiskLevel.NONE: 0, RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2,
                    RiskLevel.HIGH: 3, RiskLevel.CRITICAL: 4
                }[f.risk_level]
            ).risk_level
        else:
            highest_risk = RiskLevel.NONE
        
        result = EvaluationResult(
            allowed=not should_block,
            risk_level=highest_risk,
            findings=all_findings,
            evaluation_time_ms=evaluation_time_ms,
            evaluators_run=evaluators_run
        )
        
        # Update metrics
        self._update_metrics(result)
        
        return result
    
    def _update_metrics(self, result: EvaluationResult):
        """Update aggregated metrics"""
        with self._metrics_lock:
            self._metrics["total_evaluations"] += 1
            
            if not result.allowed:
                self._metrics["blocked"] += 1
            elif result.findings:
                self._metrics["warned"] += 1
            else:
                self._metrics["allowed"] += 1
            
            for finding in result.findings:
                # By type
                ftype = finding.type.value
                self._metrics["by_type"][ftype] = self._metrics["by_type"].get(ftype, 0) + 1
                
                # By risk
                risk = finding.risk_level.value
                self._metrics["by_risk"][risk] = self._metrics["by_risk"].get(risk, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics (for reporting to Belfry Cloud)"""
        with self._metrics_lock:
            return dict(self._metrics)
    
    def reset_metrics(self):
        """Reset metrics after reporting"""
        with self._metrics_lock:
            self._metrics = {
                "total_evaluations": 0,
                "blocked": 0,
                "warned": 0,
                "allowed": 0,
                "by_type": {},
                "by_risk": {}
            }
    
    @property
    def evaluators(self) -> List[str]:
        """Get list of all evaluator names"""
        return list(self._evaluators.keys())
    
    @property
    def enabled_evaluators(self) -> List[str]:
        """Get list of enabled evaluator names"""
        return list(self._enabled_evaluators)


# ============== Convenience Functions ==============

def create_default_evaluator() -> LocalEvaluator:
    """Create a local evaluator with default settings"""
    return LocalEvaluator()


def quick_evaluate(
    input: str = "",
    output: str = ""
) -> EvaluationResult:
    """Quick evaluation with default settings"""
    evaluator = LocalEvaluator()
    return evaluator.evaluate(input=input, output=output)
