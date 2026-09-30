"""
Compliance Resource
API client for compliance framework endpoints
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


class ComplianceResource:
    """Compliance framework API client"""
    
    def __init__(self, client):
        self._client = client
    
    async def get_frameworks(self, project_id: str) -> Dict[str, Any]:
        """
        Get compliance status across all frameworks.
        
        Args:
            project_id: Project ID to get compliance status for
            
        Returns:
            Compliance status with overall score and per-framework breakdown
        """
        return await self._client._request(
            "GET",
            f"/compliance/frameworks",
            params={"project_id": project_id}
        )
    
    async def get_framework_status(
        self, 
        project_id: str, 
        framework: str
    ) -> Dict[str, Any]:
        """
        Get detailed status for a specific framework.
        
        Args:
            project_id: Project ID
            framework: Framework name (owasp, nist_ai_rmf, eu_ai_act, iso_23053)
            
        Returns:
            Detailed framework compliance status
        """
        return await self._client._request(
            "GET",
            f"/compliance/frameworks/{framework}",
            params={"project_id": project_id}
        )
    
    async def get_issues(
        self, 
        project_id: str,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get compliance issues for a project.
        
        Args:
            project_id: Project ID
            severity: Filter by severity (critical, high, medium, low)
            status: Filter by status (open, in_progress, resolved)
            limit: Maximum number of issues to return
            
        Returns:
            List of compliance issues
        """
        params = {"project_id": project_id, "limit": limit}
        if severity:
            params["severity"] = severity
        if status:
            params["status"] = status
            
        return await self._client._request(
            "GET",
            f"/compliance/frameworks/{project_id}/issues",
            params=params
        )
    
    async def export_gdpr_data(
        self,
        user_id: str,
        format: str = "json",
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Export user data for GDPR compliance.
        
        Args:
            user_id: User ID to export data for
            format: Export format (json, csv, pdf)
            categories: Data categories to include
            
        Returns:
            Export status and download URL
        """
        return await self._client._request(
            "POST",
            "/compliance/gdpr/export",
            json={
                "user_id": user_id,
                "format": format,
                "categories": categories or ["all"]
            }
        )
    
    async def process_erasure_request(
        self,
        user_id: str,
        confirm: bool = False
    ) -> Dict[str, Any]:
        """
        Process GDPR erasure request (right to be forgotten).
        
        Args:
            user_id: User ID to erase
            confirm: Confirmation flag (required)
            
        Returns:
            Erasure status and affected records
        """
        return await self._client._request(
            "POST",
            "/compliance/gdpr/delete",
            json={"user_id": user_id, "confirm": confirm}
        )
    
    async def get_consent_status(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get consent status for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Consent records and current status
        """
        return await self._client._request(
            "GET",
            "/compliance/consent/check",
            params={"user_id": user_id}
        )
    
    async def record_consent(
        self,
        user_id: str,
        consent_type: str,
        granted: bool,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Record user consent.
        
        Args:
            user_id: User ID
            consent_type: Type of consent (data_processing, marketing, etc.)
            granted: Whether consent was granted
            metadata: Additional metadata
            
        Returns:
            Consent record
        """
        return await self._client._request(
            "POST",
            "/compliance/consent/collect",
            json={
                "user_id": user_id,
                "consent_type": consent_type,
                "granted": granted,
                "metadata": metadata or {}
            }
        )
    
    async def get_eu_ai_act_assessment(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Get EU AI Act compliance assessment.
        
        Args:
            project_id: Project ID
            
        Returns:
            EU AI Act assessment with risk level and requirements
        """
        return await self._client._request(
            "POST",
            "/compliance/ai-act/risk-assessment",
            json={"project_id": project_id}
        )
    
    async def generate_compliance_report(
        self,
        project_id: str,
        framework: str = "eu_ai_act",
        format: str = "pdf"
    ) -> Dict[str, Any]:
        """
        Generate compliance report.
        
        Args:
            project_id: Project ID
            framework: Framework to report on (default: eu_ai_act)
            format: Report format (pdf, html, json)
            
        Returns:
            Report generation status and download URL
        """
        return await self._client._request(
            "GET",
            "/compliance/ai-act/report",
            params={
                "project_id": project_id,
                "format": format
            }
        )
