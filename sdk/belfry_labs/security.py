"""
Belfry Labs Security Module
Enterprise-grade security utilities for AI safety evaluation
"""

import asyncio
import logging
import json
import hashlib
import hmac
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Threat types"""
    PROMPT_INJECTION = "prompt_injection"
    DATA_LEAKAGE = "data_leakage"
    PRIVACY_VIOLATION = "privacy_violation"
    BIAS_AMPLIFICATION = "bias_amplification"
    TOXICITY_INDUCTION = "toxicity_induction"
    ADVERSARIAL_ATTACK = "adversarial_attack"
    BACKDOOR_ACTIVATION = "backdoor_activation"


@dataclass
class SecurityThreat:
    """Security threat definition"""
    threat_type: ThreatType
    severity: SecurityLevel
    confidence: float
    description: str
    mitigation: str
    metadata: Dict[str, Any]


@dataclass
class SecurityScan:
    """Security scan result"""
    scan_id: str
    timestamp: datetime
    threats_found: List[SecurityThreat]
    risk_score: float
    recommendations: List[str]


class SecurityAnalyzer:
    """
    Enterprise-grade security analyzer for AI systems
    """
    
    def __init__(self):
        self.threat_patterns = self._initialize_threat_patterns()
        self.scan_history = []
        self.threat_statistics = defaultdict(int)
        
    def _initialize_threat_patterns(self) -> List[Dict[str, Any]]:
        """Initialize threat detection patterns"""
        
        return [
            {
                "threat_type": ThreatType.PROMPT_INJECTION,
                "pattern": r"(?i)(ignore|forget|disregard).*?(previous|system|instructions)",
                "severity": SecurityLevel.HIGH,
                "description": "Prompt injection attempt detected",
                "mitigation": "Implement input validation and sanitization"
            },
            {
                "threat_type": ThreatType.DATA_LEAKAGE,
                "pattern": r"(?i)(training data|dataset|examples|memorized)",
                "severity": SecurityLevel.CRITICAL,
                "description": "Potential data leakage attempt",
                "mitigation": "Implement data extraction detection"
            },
            {
                "threat_type": ThreatType.PRIVACY_VIOLATION,
                "pattern": r"(?i)(ssn|social security|credit card|password|secret)",
                "severity": SecurityLevel.CRITICAL,
                "description": "Personal information extraction attempt",
                "mitigation": "Implement PII detection and blocking"
            },
            {
                "threat_type": ThreatType.TOXICITY_INDUCTION,
                "pattern": r"(?i)(hate|discriminate|offensive|racist|sexist)",
                "severity": SecurityLevel.HIGH,
                "description": "Toxicity induction attempt",
                "mitigation": "Implement toxicity filtering"
            },
            {
                "threat_type": ThreatType.BIAS_AMPLIFICATION,
                "pattern": r"(?i)(stereotype|bias|prejudice|discrimination)",
                "severity": SecurityLevel.MEDIUM,
                "description": "Bias amplification attempt",
                "mitigation": "Implement bias detection and mitigation"
            }
        ]
    
    async def analyze_text(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SecurityScan:
        """
        Analyze text for security threats
        
        Args:
            text: Text to analyze
            context: Additional context
            
        Returns:
            Security scan results
        """
        
        try:
            context = context or {}
            scan_id = self._generate_scan_id()
            timestamp = datetime.utcnow()
            
            threats_found = []
            
            # Analyze against threat patterns
            for pattern in self.threat_patterns:
                import re
                matches = re.finditer(pattern["pattern"], text, re.IGNORECASE)
                
                for match in matches:
                    confidence = self._calculate_confidence(match, pattern, context)
                    
                    if confidence >= 0.7:  # High confidence threshold
                        threat = SecurityThreat(
                            threat_type=pattern["threat_type"],
                            severity=pattern["severity"],
                            confidence=confidence,
                            description=pattern["description"],
                            mitigation=pattern["mitigation"],
                            metadata={
                                "matched_text": match.group(),
                                "position": (match.start(), match.end()),
                                "pattern": pattern["pattern"]
                            }
                        )
                        
                        threats_found.append(threat)
                        self.threat_statistics[pattern["threat_type"].value] += 1
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(threats_found)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(threats_found)
            
            # Create scan result
            scan = SecurityScan(
                scan_id=scan_id,
                timestamp=timestamp,
                threats_found=threats_found,
                risk_score=risk_score,
                recommendations=recommendations
            )
            
            # Store scan history
            self.scan_history.append(scan)
            
            # Limit history size
            if len(self.scan_history) > 1000:
                self.scan_history = self.scan_history[-500:]
            
            logger.info(f"Security scan completed: {scan_id} - Found {len(threats_found)} threats")
            
            return scan
            
        except Exception as e:
            logger.error(f"Security analysis failed: {str(e)}")
            raise
    
    def _calculate_confidence(
        self,
        match: re.Match,
        pattern: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Calculate threat confidence score"""
        
        base_confidence = 0.5
        
        # Adjust based on match quality
        match_length = len(match.group())
        if match_length > 10:
            base_confidence += 0.2
        elif match_length > 5:
            base_confidence += 0.1
        
        # Adjust based on context
        if context.get("is_system_prompt", False):
            base_confidence += 0.3
        
        if context.get("user_role") == "admin":
            base_confidence -= 0.2
        
        # Add some randomness to simulate real detection
        confidence = base_confidence + np.random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_risk_score(self, threats: List[SecurityThreat]) -> float:
        """Calculate overall risk score"""
        
        if not threats:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for threat in threats:
            # Weight by severity
            if threat.severity == SecurityLevel.CRITICAL:
                weight = 4.0
            elif threat.severity == SecurityLevel.HIGH:
                weight = 3.0
            elif threat.severity == SecurityLevel.MEDIUM:
                weight = 2.0
            else:
                weight = 1.0
            
            # Score based on confidence
            score = threat.confidence * 100
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, threats: List[SecurityThreat]) -> List[str]:
        """Generate security recommendations"""
        
        recommendations = []
        
        if not threats:
            recommendations.append("No security threats detected. Continue monitoring.")
            return recommendations
        
        # Threat-specific recommendations
        threat_types = set(threat.threat_type for threat in threats)
        
        if ThreatType.PROMPT_INJECTION in threat_types:
            recommendations.append("Implement prompt injection detection and prevention mechanisms")
        
        if ThreatType.DATA_LEAKAGE in threat_types:
            recommendations.append("Implement data extraction detection and blocking")
        
        if ThreatType.PRIVACY_VIOLATION in threat_types:
            recommendations.append("Implement privacy protection and PII detection")
        
        if ThreatType.TOXICITY_INDUCTION in threat_types:
            recommendations.append("Implement toxicity filtering and content moderation")
        
        if ThreatType.BIAS_AMPLIFICATION in threat_types:
            recommendations.append("Implement bias detection and mitigation strategies")
        
        # General recommendations
        if len(threats) > 3:
            recommendations.append("Multiple security threats detected. Implement comprehensive security monitoring")
        
        if any(threat.severity == SecurityLevel.CRITICAL for threat in threats):
            recommendations.append("CRITICAL: Immediate security review required")
        
        return recommendations
    
    def _generate_scan_id(self) -> str:
        """Generate unique scan ID"""
        
        timestamp = datetime.utcnow().isoformat()
        content = f"security_scan_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    async def get_security_statistics(self) -> Dict[str, Any]:
        """Get security statistics"""
        
        total_scans = len(self.scan_history)
        
        if total_scans == 0:
            return {
                "total_scans": 0,
                "threat_types": {},
                "average_risk_score": 0.0,
                "threat_frequency": {}
            }
        
        # Calculate statistics
        threat_types = dict(self.threat_statistics)
        
        # Average risk score
        risk_scores = [scan.risk_score for scan in self.scan_history]
        average_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        
        # Threat frequency
        threat_frequency = defaultdict(int)
        for scan in self.scan_history:
            for threat in scan.threats_found:
                threat_frequency[threat.threat_type.value] += 1
        
        return {
            "total_scans": total_scans,
            "threat_types": threat_types,
            "average_risk_score": average_risk_score,
            "threat_frequency": dict(threat_frequency)
        }
    
    async def get_scan_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get scan history"""
        
        scans = self.scan_history[-limit:]
        
        return [
            {
                "scan_id": scan.scan_id,
                "timestamp": scan.timestamp.isoformat(),
                "threats_count": len(scan.threats_found),
                "risk_score": scan.risk_score,
                "threats": [
                    {
                        "threat_type": threat.threat_type.value,
                        "severity": threat.severity.value,
                        "confidence": threat.confidence,
                        "description": threat.description
                    }
                    for threat in scan.threats_found
                ]
            }
            for scan in scans
        ]


class SecurityValidator:
    """
    Security validation utilities
    """
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format"""
        
        if not api_key or len(api_key) < 32:
            return False
        
        # Check if it's a valid format (alphanumeric with some special chars)
        import re
        return bool(re.match(r'^[A-Za-z0-9_-]+$', api_key))
    
    @staticmethod
    def validate_webhook_signature(
        payload: str,
        signature: str,
        secret: str
    ) -> bool:
        """Validate webhook signature"""
        
        try:
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize input text"""
        
        # Remove control characters
        import re
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Limit length
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000]
        
        return sanitized.strip()
    
    @staticmethod
    def detect_pii(text: str) -> List[str]:
        """Detect personally identifiable information"""
        
        pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 'Credit Card'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email'),
            (r'\b\d{3}-\d{3}-\d{4}\b', 'Phone Number')
        ]
        
        detected_pii = []
        
        for pattern, pii_type in pii_patterns:
            if re.search(pattern, text):
                detected_pii.append(pii_type)
        
        return detected_pii


class SecurityMonitor:
    """
    Real-time security monitoring
    """
    
    def __init__(self):
        self.analyzer = SecurityAnalyzer()
        self.alert_threshold = 0.8
        self.alert_history = []
        
    async def monitor_text(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Monitor text for security threats"""
        
        try:
            # Analyze text
            scan = await self.analyzer.analyze_text(text, context)
            
            # Check for alerts
            alerts = []
            if scan.risk_score >= self.alert_threshold * 100:
                alert = {
                    "type": "high_risk",
                    "message": f"High risk detected: {scan.risk_score:.1f}%",
                    "timestamp": datetime.utcnow().isoformat(),
                    "threats": len(scan.threats_found)
                }
                alerts.append(alert)
                self.alert_history.append(alert)
            
            # Check for critical threats
            critical_threats = [
                threat for threat in scan.threats_found
                if threat.severity == SecurityLevel.CRITICAL
            ]
            
            if critical_threats:
                alert = {
                    "type": "critical_threat",
                    "message": f"Critical threat detected: {len(critical_threats)} threats",
                    "timestamp": datetime.utcnow().isoformat(),
                    "threats": [threat.threat_type.value for threat in critical_threats]
                }
                alerts.append(alert)
                self.alert_history.append(alert)
            
            return {
                "scan_id": scan.scan_id,
                "risk_score": scan.risk_score,
                "threats_found": len(scan.threats_found),
                "alerts": alerts,
                "recommendations": scan.recommendations
            }
            
        except Exception as e:
            logger.error(f"Security monitoring failed: {str(e)}")
            raise
    
    async def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history"""
        
        return self.alert_history[-limit:]
    
    async def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        
        total_alerts = len(self.alert_history)
        
        if total_alerts == 0:
            return {
                "total_alerts": 0,
                "alert_types": {},
                "average_risk_score": 0.0
            }
        
        # Alert types
        alert_types = defaultdict(int)
        for alert in self.alert_history:
            alert_types[alert["type"]] += 1
        
        # Average risk score from scans
        risk_scores = [scan.risk_score for scan in self.analyzer.scan_history]
        average_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        
        return {
            "total_alerts": total_alerts,
            "alert_types": dict(alert_types),
            "average_risk_score": average_risk_score
        }


# Global security instances
security_analyzer = SecurityAnalyzer()
security_validator = SecurityValidator()
security_monitor = SecurityMonitor()


# Helper functions
async def analyze_security_async(
    text: str,
    context: Optional[Dict[str, Any]] = None
) -> SecurityScan:
    """Analyze security asynchronously"""
    return await security_analyzer.analyze_text(text, context)


async def monitor_security_async(
    text: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Monitor security asynchronously"""
    return await security_monitor.monitor_text(text, context)


def validate_api_key(api_key: str) -> bool:
    """Validate API key"""
    return SecurityValidator.validate_api_key(api_key)


def sanitize_input(text: str) -> str:
    """Sanitize input text"""
    return SecurityValidator.sanitize_input(text)


def detect_pii(text: str) -> List[str]:
    """Detect PII in text"""
    return SecurityValidator.detect_pii(text)


# Backwards-compatible alias: `belfry_labs.client` imports ``SecurityClient`` as
# the public security entry point. ``SecurityAnalyzer`` is that implementation.
SecurityClient = SecurityAnalyzer
