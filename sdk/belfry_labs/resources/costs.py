"""
Costs Resource
API client for cost tracking and billing endpoints
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


class CostsResource:
    """Cost tracking and billing API client"""
    
    def __init__(self, client):
        self._client = client

    async def track(
        self,
        *,
        operation_type: str,
        cost_usd: float,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
    ) -> Dict[str, Any]:
        """
        Record an inference / evaluation cost event.

        This feeds the Cost Optimization dashboard and SLM recommendations.
        Call this (or use gateway tracking) whenever your app completes an LLM call.
        """
        return await self._client._request(
            "POST",
            "/cost-tracking/track",
            json_data={
                "operation_type": operation_type,
                "cost_usd": cost_usd,
                "project_id": project_id,
                "model_id": model_id,
                "provider": provider,
                "model_name": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens or (input_tokens + output_tokens),
            },
        )

    async def get_optimization_analysis(
        self,
        project_id: Optional[str] = None,
        timeframe: str = "30d",
    ) -> Dict[str, Any]:
        """Full cost-optimization analysis (recommendations + metrics)."""
        params: Dict[str, Any] = {"timeframe": timeframe}
        if project_id:
            params["project_id"] = project_id
        return await self._client._request(
            "GET",
            "/cost-optimization/analysis",
            params=params,
        )

    async def get_slm_recommendations(
        self,
        current_model: str,
        project_id: Optional[str] = None,
        task_complexity: str = "medium",
    ) -> Dict[str, Any]:
        """SLMOptimizer suggestions for a current model."""
        params: Dict[str, Any] = {
            "current_model": current_model,
            "task_complexity": task_complexity,
        }
        if project_id:
            params["project_id"] = project_id
        return await self._client._request(
            "GET",
            "/cost-optimization/routing/recommendations",
            params=params,
        )

    async def get_project_recommendations(
        self,
        project_id: str,
    ) -> Dict[str, Any]:
        """Project Cost dashboard payload (ops/cost recommendations)."""
        return await self._client._request(
            "GET",
            f"/ops/cost/{project_id}/recommendations",
        )
    
    async def get_usage_summary(
        self,
        time_period: str = "30d",
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get usage summary.
        
        Args:
            time_period: Time period (7d, 30d, 90d, ytd)
            project_id: Filter by project
            
        Returns:
            Usage summary with totals
        """
        params = {"time_period": time_period}
        if project_id:
            params["project_id"] = project_id
            
        return await self._client._request(
            "GET",
            "/costs/summary",
            params=params
        )
    
    async def get_cost_breakdown(
        self,
        time_period: str = "30d",
        group_by: str = "model"
    ) -> Dict[str, Any]:
        """
        Get cost breakdown.
        
        Args:
            time_period: Time period
            group_by: Grouping (model, project, user)
            
        Returns:
            Cost breakdown by group
        """
        return await self._client._request(
            "GET",
            "/costs/breakdown",
            params={
                "time_period": time_period,
                "group_by": group_by
            }
        )
    
    async def get_cost_trend(
        self,
        time_period: str = "30d",
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get cost trend over time.
        
        Args:
            time_period: Time period
            project_id: Filter by project
            
        Returns:
            Time series of costs
        """
        params = {"time_period": time_period}
        if project_id:
            params["project_id"] = project_id
            
        return await self._client._request(
            "GET",
            "/costs/trends",
            params=params
        )
    
    async def set_budget_alert(
        self,
        threshold_usd: float,
        alert_type: str = "monthly",
        project_id: Optional[str] = None,
        notify_emails: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Set budget alert threshold.
        
        Args:
            threshold_usd: Alert threshold in USD
            alert_type: Alert period (daily, weekly, monthly)
            project_id: Optional project scope
            notify_emails: Email addresses to notify
            
        Returns:
            Created budget alert
        """
        return await self._client._request(
            "POST",
            "/costs/budget-alerts",
            json_data={
                "threshold_usd": threshold_usd,
                "alert_type": alert_type,
                "project_id": project_id,
                "notify_emails": notify_emails or []
            }
        )
    
    async def get_budget_alerts(self) -> Dict[str, Any]:
        """
        Get configured budget alerts.
        
        Returns:
            List of budget alerts
        """
        return await self._client._request(
            "GET",
            "/costs/budget-alerts"
        )
    
    async def get_provider_costs(
        self,
        time_period: str = "30d"
    ) -> Dict[str, Any]:
        """
        Get costs by provider.
        
        Args:
            time_period: Time period
            
        Returns:
            Cost breakdown by provider
        """
        return await self._client._request(
            "GET",
            "/costs/by-provider",
            params={"time_period": time_period}
        )
    
    async def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, Any]:
        """
        Estimate cost for API usage.
        
        Args:
            model: Model name
            input_tokens: Expected input tokens
            output_tokens: Expected output tokens
            
        Returns:
            Estimated cost
        """
        return await self._client._request(
            "POST",
            "/costs/estimate",
            json_data={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        )
    
    async def get_billing_history(
        self,
        limit: int = 12
    ) -> Dict[str, Any]:
        """
        Get billing history.
        
        Args:
            limit: Number of billing periods to return
            
        Returns:
            Billing history
        """
        return await self._client._request(
            "GET",
            "/billing/history",
            params={"limit": limit}
        )
    
    async def get_current_invoice(self) -> Dict[str, Any]:
        """
        Get current billing period invoice.
        
        Returns:
            Current invoice details
        """
        return await self._client._request(
            "GET",
            "/billing/current"
        )
    
    async def export_usage_report(
        self,
        time_period: str = "30d",
        format: str = "csv"
    ) -> Dict[str, Any]:
        """
        Export usage report.
        
        Args:
            time_period: Time period
            format: Export format (csv, json, pdf)
            
        Returns:
            Export status and download URL
        """
        return await self._client._request(
            "POST",
            "/costs/export",
            json_data={
                "time_period": time_period,
                "format": format
            }
        )
