"""
Analytics Resource
API client for analytics and dashboard endpoints
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


class AnalyticsResource:
    """Analytics and dashboard API client"""
    
    def __init__(self, client):
        self._client = client
    
    async def get_dashboard(
        self,
        project_id: Optional[str] = None,
        time_period: str = "7d"
    ) -> Dict[str, Any]:
        """
        Get dashboard metrics.
        
        Args:
            project_id: Filter by project
            time_period: Time period (1d, 7d, 30d, 90d)
            
        Returns:
            Dashboard metrics and KPIs
        """
        params = {"time_period": time_period}
        if project_id:
            params["project_id"] = project_id
            
        return await self._client._request(
            "GET",
            "/analytics/dashboard",
            params=params
        )
    
    async def get_evaluation_metrics(
        self,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
        time_period: str = "30d"
    ) -> Dict[str, Any]:
        """
        Get evaluation metrics.
        
        Args:
            project_id: Filter by project
            model_id: Filter by model
            time_period: Time period
            
        Returns:
            Evaluation metrics (count, success rate, scores)
        """
        params = {"time_period": time_period}
        if project_id:
            params["project_id"] = project_id
        if model_id:
            params["model_id"] = model_id
            
        return await self._client._request(
            "GET",
            "/analytics/evaluations",
            params=params
        )
    
    async def get_safety_trends(
        self,
        project_id: Optional[str] = None,
        time_period: str = "30d",
        granularity: str = "daily"
    ) -> Dict[str, Any]:
        """
        Get safety score trends over time.
        
        Args:
            project_id: Filter by project
            time_period: Time period
            granularity: Data granularity (hourly, daily, weekly)
            
        Returns:
            Time series of safety scores
        """
        params = {"time_period": time_period, "granularity": granularity}
        if project_id:
            params["project_id"] = project_id
            
        return await self._client._request(
            "GET",
            "/analytics/trends",
            params=params
        )
    
    async def get_model_comparison(
        self,
        model_ids: List[str],
        benchmark_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models.
        
        Args:
            model_ids: List of model IDs to compare
            benchmark_ids: Optional benchmark filter
            
        Returns:
            Model comparison data
        """
        return await self._client._request(
            "POST",
            "/analytics/model-comparison",
            json={
                "model_ids": model_ids,
                "benchmark_ids": benchmark_ids
            }
        )
    
    async def get_vulnerability_breakdown(
        self,
        project_id: Optional[str] = None,
        time_period: str = "30d"
    ) -> Dict[str, Any]:
        """
        Get vulnerability breakdown by category.
        
        Args:
            project_id: Filter by project
            time_period: Time period
            
        Returns:
            Vulnerability counts by category and severity
        """
        params = {"time_period": time_period}
        if project_id:
            params["project_id"] = project_id
            
        return await self._client._request(
            "GET",
            "/analytics/security-summary",
            params=params
        )
    
    async def get_usage_stats(
        self,
        time_period: str = "30d"
    ) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Args:
            time_period: Time period
            
        Returns:
            Usage statistics (API calls, scans, evaluations)
        """
        return await self._client._request(
            "GET",
            "/analytics/usage",
            params={"time_period": time_period}
        )
    
    async def export_analytics(
        self,
        report_type: str,
        time_period: str = "30d",
        format: str = "csv",
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export analytics data.
        
        Args:
            report_type: Type of report (evaluations, vulnerabilities, usage)
            time_period: Time period
            format: Export format (csv, json, xlsx)
            project_id: Optional project filter
            
        Returns:
            Export status and download URL
        """
        return await self._client._request(
            "POST",
            "/analytics/export",
            json={
                "report_type": report_type,
                "time_period": time_period,
                "format": format,
                "project_id": project_id
            }
        )
    
    async def get_chat_analytics(
        self,
        project_id: Optional[str] = None,
        time_period: str = "7d"
    ) -> Dict[str, Any]:
        """
        Get chat/conversation analytics.
        
        Args:
            project_id: Filter by project
            time_period: Time period
            
        Returns:
            Chat analytics (session count, avg duration, issues detected)
        """
        params = {"time_period": time_period}
        if project_id:
            params["project_id"] = project_id
            
        return await self._client._request(
            "GET",
            "/chat/analytics/summary",
            params=params
        )
