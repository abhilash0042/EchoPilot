"""
Belfry Labs Policy Client
Client-side policy evaluation for enterprise deployments.
Syncs policies from Belfry Cloud, evaluates locally, reports only aggregated metrics.
"""

import re
import hashlib
import threading
import time
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger(__name__)


class PolicyAction(str, Enum):
    """Actions a policy can take"""
    BLOCK = "block"
    WARN = "warn"
    LOG = "log"
    ALLOW = "allow"


class PolicySeverity(str, Enum):
    """Severity levels for policy violations"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PolicyRule:
    """A rule within a policy"""
    id: str
    name: str
    condition: str
    action: PolicyAction
    severity: PolicySeverity = PolicySeverity.MEDIUM
    message: Optional[str] = None


@dataclass
class Policy:
    """A safety policy for local evaluation"""
    id: str
    name: str
    policy_type: str
    rules: List[PolicyRule]
    version: int
    priority: int
    checksum: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result of evaluating content against policies"""
    allowed: bool
    action: PolicyAction
    triggered_rules: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    highest_severity: Optional[PolicySeverity] = None
    evaluation_time_ms: float = 0


@dataclass
class InterventionMetrics:
    """Aggregated intervention metrics for reporting"""
    policy_id: str
    project_id: Optional[str]
    period_start: datetime
    period_end: datetime
    total_evaluations: int = 0
    interventions_count: int = 0
    blocks_count: int = 0
    warnings_count: int = 0
    avg_evaluation_ms: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "project_id": self.project_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_evaluations": self.total_evaluations,
            "interventions_count": self.interventions_count,
            "blocks_count": self.blocks_count,
            "warnings_count": self.warnings_count,
            "avg_evaluation_ms": self.avg_evaluation_ms
        }


class PatternMatcher:
    """Built-in pattern matchers for policy conditions"""
    
    PATTERNS = {
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "api_key": r"\b(?:sk|pk)[-_][a-zA-Z0-9]{20,}\b"
    }
    
    @classmethod
    def contains_pattern(cls, text: str, pattern_name: str) -> bool:
        """Check if text contains a known pattern"""
        if pattern_name not in cls.PATTERNS:
            return False
        return bool(re.search(cls.PATTERNS[pattern_name], text, re.IGNORECASE))
    
    @classmethod
    def contains_phrase(cls, text: str, phrases: List[str]) -> bool:
        """Check if text contains any of the phrases"""
        text_lower = text.lower()
        return any(phrase.lower() in text_lower for phrase in phrases)


class LocalPolicyEvaluator:
    """Evaluates content against policies locally"""
    
    def __init__(self):
        self._custom_evaluators: Dict[str, Callable] = {}
        self._setup_default_evaluators()
    
    def _setup_default_evaluators(self):
        """Setup default evaluation functions"""
        self._custom_evaluators["contains_pattern"] = PatternMatcher.contains_pattern
        self._custom_evaluators["contains_phrase"] = PatternMatcher.contains_phrase
        self._custom_evaluators["len"] = len
    
    def register_evaluator(self, name: str, func: Callable):
        """Register a custom evaluation function"""
        self._custom_evaluators[name] = func
    
    def evaluate_condition(
        self,
        condition: str,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate a condition expression"""
        # Build safe evaluation context
        safe_context = {
            "input": context.get("input", ""),
            "output": context.get("output", ""),
            **self._custom_evaluators
        }
        
        try:
            # Parse and evaluate the condition
            # Using a simplified expression evaluator for safety
            return self._safe_eval(condition, safe_context)
        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False
    
    def _safe_eval(self, expr: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate a simple expression"""
        # Handle common patterns
        
        # Pattern: contains_pattern(input, 'pattern_name')
        match = re.match(r"contains_pattern\((\w+),\s*['\"](\w+)['\"]\)", expr)
        if match:
            var_name, pattern = match.groups()
            text = context.get(var_name, "")
            return PatternMatcher.contains_pattern(text, pattern)
        
        # Pattern: contains_phrase(input, ['phrase1', 'phrase2'])
        match = re.match(r"contains_phrase\((\w+),\s*\[(.*?)\]\)", expr)
        if match:
            var_name, phrases_str = match.groups()
            text = context.get(var_name, "")
            phrases = [p.strip().strip("'\"") for p in phrases_str.split(",")]
            return PatternMatcher.contains_phrase(text, phrases)
        
        # Pattern: len(output) > N
        match = re.match(r"len\((\w+)\)\s*([<>=!]+)\s*(\d+)", expr)
        if match:
            var_name, operator, threshold = match.groups()
            text = context.get(var_name, "")
            length = len(text)
            threshold = int(threshold)
            
            if operator == ">":
                return length > threshold
            elif operator == "<":
                return length < threshold
            elif operator == ">=":
                return length >= threshold
            elif operator == "<=":
                return length <= threshold
            elif operator == "==":
                return length == threshold
        
        return False


class BelfryPolicyClient:
    """
    Client for syncing and evaluating Belfry safety policies.
    
    Usage:
        client = BelfryPolicyClient(
            sync_token="belfry_sync_...",
            base_url="https://api.belfry.ai"
        )
        
        # Evaluate input
        result = client.evaluate(input="user message", output="model response")
        
        if not result.allowed:
            print(f"Blocked: {result.messages}")
    """
    
    def __init__(
        self,
        sync_token: str,
        base_url: str = "https://api.belfry.ai",
        environment: str = "production",
        project_id: Optional[str] = None,
        auto_sync: bool = True,
        sync_interval_seconds: int = 300
    ):
        if httpx is None:
            raise ImportError("httpx is required for BelfryPolicyClient. Install with: pip install httpx")
        
        self.sync_token = sync_token
        self.base_url = base_url.rstrip("/")
        self.environment = environment
        self.project_id = project_id
        
        self._policies: List[Policy] = []
        self._sync_version: Optional[str] = None
        self._last_sync: Optional[datetime] = None
        self._sync_interval = sync_interval_seconds
        
        self._evaluator = LocalPolicyEvaluator()
        self._metrics: Dict[str, InterventionMetrics] = {}
        self._metrics_lock = threading.Lock()
        self._metrics_period_start = datetime.utcnow()
        
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_sync = threading.Event()
        
        self._client = httpx.Client(
            timeout=30.0,
            headers={"Authorization": f"Bearer {sync_token}"}
        )
        
        # Initial sync
        self.sync_policies()
        
        # Start background sync
        if auto_sync:
            self._start_background_sync()
    
    def _start_background_sync(self):
        """Start background policy sync thread"""
        def sync_loop():
            while not self._stop_sync.wait(self._sync_interval):
                try:
                    self.sync_policies()
                except Exception as e:
                    logger.error(f"Background sync failed: {e}")
        
        self._sync_thread = threading.Thread(target=sync_loop, daemon=True)
        self._sync_thread.start()
    
    def sync_policies(self) -> bool:
        """Sync policies from Belfry Cloud"""
        try:
            response = self._client.post(
                f"{self.base_url}/api/v1/policy/sdk/sync",
                json={
                    "environment": self.environment,
                    "current_version": self._sync_version,
                    "client_version": "1.0.0"
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Update policies
            self._policies = [
                Policy(
                    id=p["id"],
                    name=p["name"],
                    policy_type=p["policy_type"],
                    rules=[
                        PolicyRule(
                            id=r["id"],
                            name=r["name"],
                            condition=r["condition"],
                            action=PolicyAction(r["action"]),
                            severity=PolicySeverity(r.get("severity", "medium")),
                            message=r.get("message")
                        )
                        for r in p["rules"]
                    ],
                    version=p["version"],
                    priority=p["priority"],
                    checksum=p.get("checksum")
                )
                for p in data["policies"]
            ]
            
            self._sync_version = data["sync_version"]
            self._sync_interval = data.get("next_sync_in_seconds", 300)
            self._last_sync = datetime.utcnow()
            
            logger.info(f"Synced {len(self._policies)} policies (version: {self._sync_version})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync policies: {e}")
            return False
    
    def evaluate(
        self,
        input: str = "",
        output: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate content against all policies.
        Returns result with action to take.
        """
        start_time = time.time()
        
        eval_context = {
            "input": input,
            "output": output,
            **(context or {})
        }
        
        triggered_rules = []
        messages = []
        highest_severity: Optional[PolicySeverity] = None
        final_action = PolicyAction.ALLOW
        
        severity_order = {
            PolicySeverity.LOW: 1,
            PolicySeverity.MEDIUM: 2,
            PolicySeverity.HIGH: 3,
            PolicySeverity.CRITICAL: 4
        }
        
        action_priority = {
            PolicyAction.ALLOW: 0,
            PolicyAction.LOG: 1,
            PolicyAction.WARN: 2,
            PolicyAction.BLOCK: 3
        }
        
        # Evaluate policies in priority order
        for policy in sorted(self._policies, key=lambda p: p.priority):
            for rule in policy.rules:
                try:
                    if self._evaluator.evaluate_condition(rule.condition, eval_context):
                        triggered_rules.append({
                            "policy_id": policy.id,
                            "policy_name": policy.name,
                            "rule_id": rule.id,
                            "rule_name": rule.name,
                            "action": rule.action.value,
                            "severity": rule.severity.value
                        })
                        
                        if rule.message:
                            messages.append(rule.message)
                        
                        # Update highest severity
                        if highest_severity is None or severity_order[rule.severity] > severity_order[highest_severity]:
                            highest_severity = rule.severity
                        
                        # Update action (more restrictive wins)
                        if action_priority[rule.action] > action_priority[final_action]:
                            final_action = rule.action
                        
                        # Record metric
                        self._record_intervention(policy.id, rule.action)
                        
                except Exception as e:
                    logger.warning(f"Error evaluating rule {rule.id}: {e}")
        
        evaluation_time_ms = (time.time() - start_time) * 1000
        
        # Record total evaluation
        for policy in self._policies:
            self._record_evaluation(policy.id, evaluation_time_ms)
        
        return EvaluationResult(
            allowed=final_action != PolicyAction.BLOCK,
            action=final_action,
            triggered_rules=triggered_rules,
            messages=messages,
            highest_severity=highest_severity,
            evaluation_time_ms=evaluation_time_ms
        )
    
    def _record_evaluation(self, policy_id: str, evaluation_time_ms: float):
        """Record an evaluation for metrics"""
        with self._metrics_lock:
            if policy_id not in self._metrics:
                self._metrics[policy_id] = InterventionMetrics(
                    policy_id=policy_id,
                    project_id=self.project_id,
                    period_start=self._metrics_period_start,
                    period_end=datetime.utcnow()
                )
            
            metric = self._metrics[policy_id]
            metric.total_evaluations += 1
            metric.period_end = datetime.utcnow()
            
            # Update rolling average
            old_avg = metric.avg_evaluation_ms
            old_count = metric.total_evaluations - 1
            if old_count > 0:
                metric.avg_evaluation_ms = (old_avg * old_count + evaluation_time_ms) / metric.total_evaluations
            else:
                metric.avg_evaluation_ms = evaluation_time_ms
    
    def _record_intervention(self, policy_id: str, action: PolicyAction):
        """Record an intervention for metrics"""
        with self._metrics_lock:
            if policy_id not in self._metrics:
                self._metrics[policy_id] = InterventionMetrics(
                    policy_id=policy_id,
                    project_id=self.project_id,
                    period_start=self._metrics_period_start,
                    period_end=datetime.utcnow()
                )
            
            metric = self._metrics[policy_id]
            metric.interventions_count += 1
            
            if action == PolicyAction.BLOCK:
                metric.blocks_count += 1
            elif action == PolicyAction.WARN:
                metric.warnings_count += 1
    
    def report_metrics(self) -> bool:
        """Report aggregated metrics to Belfry (no content, only counts)"""
        with self._metrics_lock:
            if not self._metrics:
                return True
            
            reports = [m.to_dict() for m in self._metrics.values()]
            
            # Reset metrics for next period
            self._metrics = {}
            self._metrics_period_start = datetime.utcnow()
        
        try:
            response = self._client.post(
                f"{self.base_url}/api/v1/policy/sdk/interventions",
                json={"reports": reports}
            )
            response.raise_for_status()
            logger.info(f"Reported metrics for {len(reports)} policies")
            return True
        except Exception as e:
            logger.error(f"Failed to report metrics: {e}")
            return False
    
    @property
    def policies(self) -> List[Policy]:
        """Get currently loaded policies"""
        return self._policies
    
    @property
    def last_sync(self) -> Optional[datetime]:
        """Get last sync timestamp"""
        return self._last_sync
    
    def close(self):
        """Close the client and stop background sync"""
        self._stop_sync.set()
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
        
        # Report final metrics
        self.report_metrics()
        
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions for quick evaluation
def create_client(
    sync_token: str,
    base_url: str = "https://api.belfry.ai",
    **kwargs
) -> BelfryPolicyClient:
    """Create a new policy client"""
    return BelfryPolicyClient(sync_token=sync_token, base_url=base_url, **kwargs)
