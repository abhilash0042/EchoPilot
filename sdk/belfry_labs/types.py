"""
Type definitions for Belfry Labs SDK.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class Project(BaseModel):
    """Project model."""
    id: str
    name: str
    description: Optional[str] = None
    slug: Optional[str] = None
    tenant_id: str
    owner_id: str
    owner_name: Optional[str] = None
    settings: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    status: str = "active"
    visibility: str = "private"
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None


class Model(BaseModel):
    """AI Model model."""
    id: str
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    project_id: str
    tenant_id: str
    provider: Optional[str] = None
    modality: str
    model_type: Optional[str] = None
    architecture: Optional[str] = None
    source_type: str = "api"
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    safety_score: Optional[float] = None
    safety_status: str = "pending"
    status: str = "active"
    visibility: str = "private"
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None


class Dataset(BaseModel):
    """Dataset model."""
    id: str
    name: str
    description: Optional[str] = None
    project_id: str
    tenant_id: str
    dataset_type: str
    format: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    record_count: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: str = "uploaded"
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None


class Evaluation(BaseModel):
    """Evaluation model."""
    id: str
    name: str
    description: Optional[str] = None
    project_id: str
    model_id: str
    user_id: str
    tenant_id: str
    benchmarks: List[str]
    evaluation_type: str = "safety"
    modality: str
    status: str = "pending"
    progress: float = 0.0
    overall_score: Optional[float] = None
    risk_level: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    tokens_used: int = 0
    api_calls_made: int = 0
    cost_total: float = 0.0
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None


class RedTeamSession(BaseModel):
    """Red team session model."""
    id: str
    name: str
    description: Optional[str] = None
    project_id: str
    target_model_id: str
    user_id: str
    tenant_id: str
    attack_types: List[str]
    status: str = "pending"
    progress: float = 0.0
    total_attacks: int = 0
    successful_attacks: int = 0
    success_rate: float = 0.0
    risk_score: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None


class VulnerabilityReport(BaseModel):
    """Vulnerability report model."""
    id: str
    title: str
    description: str
    project_id: str
    model_id: str
    tenant_id: str
    vulnerability_type: str
    severity: str
    status: str = "open"
    poc_prompt: Optional[str] = None
    poc_response: Optional[str] = None
    recommended_actions: List[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None


class Report(BaseModel):
    """Report model."""
    id: str
    title: str
    description: Optional[str] = None
    project_id: str
    user_id: str
    tenant_id: str
    report_type: str
    format: str = "pdf"
    status: str = "pending"
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None


class CostRecord(BaseModel):
    """Cost record model."""
    id: str
    project_id: str
    user_id: str
    tenant_id: str
    provider: str
    model_name: Optional[str] = None
    tokens_total: int = 0
    cost_total: float
    cost_currency: str = "USD"
    operation_type: Optional[str] = None
    usage_date: datetime
    is_active: bool = True
    created_at: datetime


class BenchmarkResult(BaseModel):
    """Benchmark result model."""
    benchmark_name: str
    score: float
    pass_rate: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    status: str = "completed"


class EvaluationResults(BaseModel):
    """Evaluation results model."""
    evaluation_id: str
    overall_score: float
    risk_level: str
    benchmarks: List[BenchmarkResult]
    summary: Dict[str, Any]
    detailed_results: Dict[str, Any]
