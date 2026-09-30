"""
Security Resource
API client for security and firewall endpoints
"""

from typing import Dict, List, Any, Optional


class SecurityResource:
    """Security and firewall API client"""
    
    def __init__(self, client):
        self._client = client
    
    async def scan_content(
        self,
        content: str,
        content_type: str = "query",
        scan_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Scan content for security threats.
        
        Args:
            content: Content to scan
            content_type: Type of content (query, response)
            scan_types: Types of scans to run (prompt_injection, pii, toxicity)
            
        Returns:
            Scan results with detected threats
        """
        return await self._client._request(
            "POST",
            "/security/scan",
            json={
                "content": content,
                "content_type": content_type,
                "scan_types": scan_types or ["prompt_injection", "pii", "toxicity"]
            }
        )
    
    async def create_firewall_policy(
        self,
        name: str,
        policy_type: str,
        rules: List[Dict[str, Any]],
        action_on_violation: str = "block",
        project_id: Optional[str] = None,
        priority: int = 100
    ) -> Dict[str, Any]:
        """
        Create a firewall policy.
        
        Args:
            name: Policy name
            policy_type: Type of policy (prompt_injection, jailbreak, pii, custom)
            rules: List of rules to enforce
            action_on_violation: Action on violation (block, warn, log)
            project_id: Optional project scope
            priority: Policy priority (lower = higher priority)
            
        Returns:
            Created policy
        """
        return await self._client._request(
            "POST",
            "/firewall/policies",
            json={
                "name": name,
                "policy_type": policy_type,
                "rules": rules,
                "action_on_violation": action_on_violation,
                "project_id": project_id,
                "priority": priority
            }
        )
    
    async def get_firewall_policies(
        self,
        project_id: Optional[str] = None,
        policy_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get firewall policies.
        
        Args:
            project_id: Filter by project
            policy_type: Filter by type
            
        Returns:
            List of policies
        """
        params = {}
        if project_id:
            params["project_id"] = project_id
        if policy_type:
            params["policy_type"] = policy_type
            
        return await self._client._request(
            "GET",
            "/firewall/policies",
            params=params
        )
    
    async def evaluate_prompt(
        self,
        prompt: str,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt against firewall policies.
        
        Args:
            prompt: Prompt to evaluate
            project_id: Project context
            
        Returns:
            Evaluation result with action and violations
        """
        return await self._client._request(
            "POST",
            "/firewall/check",
            json={
                "user_query": prompt,
                "project_id": project_id
            }
        )
    
    async def get_violations(
        self,
        project_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get firewall violations.
        
        Args:
            project_id: Filter by project
            severity: Filter by severity
            limit: Maximum results
            
        Returns:
            List of violations
        """
        params = {"limit": limit}
        if project_id:
            params["project_id"] = project_id
        if severity:
            params["severity"] = severity
            
        return await self._client._request(
            "GET",
            "/firewall/violations",
            params=params
        )
    
    async def get_security_metrics(
        self,
        project_id: Optional[str] = None,
        time_period: str = "7d"
    ) -> Dict[str, Any]:
        """
        Get security metrics.
        
        Args:
            project_id: Filter by project
            time_period: Time period (1d, 7d, 30d, 90d)
            
        Returns:
            Security metrics summary
        """
        params = {"time_period": time_period}
        if project_id:
            params["project_id"] = project_id
            
        return await self._client._request(
            "GET",
            "/security/metrics",
            params=params
        )
    
    async def run_vulnerability_scan(
        self,
        project_id: str,
        scan_type: str = "full"
    ) -> Dict[str, Any]:
        """
        Run vulnerability scan on project.
        
        Args:
            project_id: Project to scan
            scan_type: Type of scan (quick, full, focused)
            
        Returns:
            Scan job ID and status
        """
        return await self._client._request(
            "POST",
            "/security/scan/project",
            json={
                "project_id": project_id,
                "scan_type": scan_type
            }
        )
    
    async def get_dlp_policies(
        self,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get DLP (Data Loss Prevention) policies.
        
        Args:
            project_id: Filter by project
            
        Returns:
            List of DLP policies
        """
        params = {}
        if project_id:
            params["project_id"] = project_id
            
        return await self._client._request(
            "GET",
            "/dlp/policies",
            params=params
        )
    
    async def scan_for_pii(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Scan content for PII.
        
        Args:
            content: Content to scan
            
        Returns:
            PII detection results
        """
        return await self._client._request(
            "POST",
            "/dlp/scan",
            json={"content": content}
        )
