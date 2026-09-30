"""
Evaluations management for Belfry Labs SDK
"""

from typing import List, Dict, Optional, Any
import time

from .exceptions import APIError

class Evaluation:
    """Evaluation representation"""

    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id")
        self.name = data.get("name")
        self.description = data.get("description")
        self.project_id = data.get("project_id")
        self.project_name = data.get("project_name")
        self.model_id = data.get("model_id")
        self.model_name = data.get("model_name")
        self.benchmarks = data.get("benchmarks", [])
        self.evaluation_type = data.get("evaluation_type")
        self.modality = data.get("modality")
        self.status = data.get("status")
        self.progress = data.get("progress", 0.0)
        self.overall_score = data.get("overall_score")
        self.risk_level = data.get("risk_level")
        self.started_at = data.get("started_at")
        self.completed_at = data.get("completed_at")
        self.duration_seconds = data.get("duration_seconds")
        self.cost_total = data.get("cost_total", 0.0)
        self.created_at = data.get("created_at")

    def __repr__(self):
        return f"Evaluation(id='{self.id}', name='{self.name}', status='{self.status}')"

    @property
    def is_running(self) -> bool:
        """Check if evaluation is currently running"""
        return self.status in ["pending", "running"]

    @property
    def is_completed(self) -> bool:
        """Check if evaluation completed successfully"""
        return self.status == "completed"

class EvaluationManager:
    """Manager for evaluation operations"""

    def __init__(self, client):
        self.client = client

    def list(
        self,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Evaluation]:
        """
        List evaluations

        Args:
            project_id: Filter by project ID
            model_id: Filter by model ID
            status: Filter by status
            limit: Maximum results

        Returns:
            List of Evaluation objects
        """
        params = {"limit": limit}
        if project_id:
            params["project_id"] = project_id
        if model_id:
            params["model_id"] = model_id
        if status:
            params["status"] = status

        response = self.client.get("/api/v1/evaluations", params=params)
        return [Evaluation(item) for item in response.json()]

    def get(self, evaluation_id: str) -> Evaluation:
        """
        Get evaluation by ID

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Evaluation object
        """
        response = self.client.get(f"/api/v1/evaluations/{evaluation_id}")
        return Evaluation(response.json())

    def create(
        self,
        project_id: str,
        model_id: str,
        benchmarks: List[str],
        name: str,
        description: Optional[str] = None,
        dataset_ids: Optional[List[str]] = None,
        evaluation_params: Optional[Dict] = None,
        comparison_mode: bool = False,
        baseline_model_id: Optional[str] = None
    ) -> Evaluation:
        """
        Create new evaluation

        Args:
            project_id: Project ID
            model_id: Model ID to evaluate
            benchmarks: List of benchmark names
            name: Evaluation name
            description: Evaluation description
            dataset_ids: Specific datasets to use
            evaluation_params: Benchmark-specific parameters
            comparison_mode: Enable comparison mode
            baseline_model_id: Baseline model for comparison

        Returns:
            Evaluation object
        """
        data = {
            "name": name,
            "description": description,
            "project_id": project_id,
            "model_id": model_id,
            "benchmarks": benchmarks,
            "dataset_ids": dataset_ids or [],
            "evaluation_params": evaluation_params or {},
            "comparison_mode": comparison_mode,
            "baseline_model_id": baseline_model_id
        }

        response = self.client.post("/api/v1/evaluations", data=data)
        return Evaluation(response.json())

    def get_results(self, evaluation_id: str) -> Dict[str, Any]:
        """
        Get detailed evaluation results

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Detailed results including benchmark scores
        """
        response = self.client.get(f"/api/v1/evaluations/{evaluation_id}/results")
        return response.json()

    def wait_for_completion(
        self,
        evaluation_id: str,
        timeout: int = 3600,
        poll_interval: int = 10
    ) -> Evaluation:
        """
        Wait for evaluation to complete

        Args:
            evaluation_id: Evaluation ID
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            Completed Evaluation object

        Raises:
            APIError: If evaluation fails or times out
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            evaluation = self.get(evaluation_id)

            if evaluation.is_completed:
                return evaluation
            elif evaluation.status == "failed":
                raise APIError(f"Evaluation failed: {evaluation_id}")
            elif evaluation.status == "cancelled":
                raise APIError(f"Evaluation was cancelled: {evaluation_id}")

            print(f"Evaluation {evaluation_id}: {evaluation.status} ({evaluation.progress:.1f}%)")
            time.sleep(poll_interval)

        raise APIError(f"Evaluation timed out after {timeout} seconds")

    def cancel(self, evaluation_id: str) -> Dict[str, str]:
        """
        Cancel running evaluation

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Cancellation confirmation
        """
        response = self.client.post(f"/api/v1/evaluations/{evaluation_id}/cancel")
        return response.json()

    def get_logs(self, evaluation_id: str) -> Dict[str, Any]:
        """
        Get evaluation logs

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Evaluation logs
        """
        response = self.client.get(f"/api/v1/evaluations/{evaluation_id}/logs")
        return response.json()

    def list_benchmarks(self, modality: Optional[str] = None) -> Dict[str, Any]:
        """
        List available benchmarks

        Args:
            modality: Filter by modality

        Returns:
            Available benchmarks by modality
        """
        params = {}
        if modality:
            params["modality"] = modality

        response = self.client.get("/api/v1/evaluations/templates/benchmarks", params=params)
        return response.json()

    def register_custom(
        self,
        name: str,
        dimensions: Dict[str, Any],
        judge_prompt: Optional[str] = None,
        description: Optional[str] = None,
        modality: str = "text",
        tags: Optional[List[str]] = None,
    ) -> "EvaluationProfile":
        """
        Register a custom evaluation profile (Bring Your Own Eval).

        Custom profiles run alongside built-in benchmarks. Each dimension
        defines a scoring axis with a weight and optional prompt template.

        Args:
            name: Profile name (e.g., "hallucination_check")
            dimensions: Dict of dimension configs, each with:
                - weight (float): Relative weight for scoring
                - description (str): What this dimension measures
                - prompt_template (str, optional): Judge prompt for this dimension
            judge_prompt: Global judge prompt applied to all dimensions
            description: Human-readable description
            modality: Target modality (text, image, audio, video, multimodal)
            tags: Optional tags for filtering

        Returns:
            EvaluationProfile with the registered profile details

        Example:
            >>> profile = client.evaluations.register_custom(
            ...     name="hallucination_check",
            ...     dimensions={
            ...         "factual_accuracy": {"weight": 0.5, "description": "Are claims grounded?"},
            ...         "citation_quality": {"weight": 0.3, "description": "Are citations valid?"},
            ...         "uncertainty":      {"weight": 0.2, "description": "Does it express doubt?"},
            ...     },
            ...     judge_prompt="Evaluate the response for factual grounding...",
            ... )
        """
        data = {
            "name": name,
            "description": description or f"Custom evaluation: {name}",
            "modality": modality,
            "dimensions": dimensions,
            "judge_prompt": judge_prompt,
            "tags": tags or [],
        }

        response = self.client.post("/api/v1/evaluation-profiles", data=data)
        return EvaluationProfile(response.json())

    def list_profiles(self, modality: Optional[str] = None) -> List["EvaluationProfile"]:
        """
        List registered evaluation profiles.

        Args:
            modality: Optional modality filter

        Returns:
            List of EvaluationProfile objects
        """
        params = {}
        if modality:
            params["modality"] = modality

        response = self.client.get("/api/v1/evaluation-profiles", params=params)
        items = response.json() if isinstance(response.json(), list) else response.json().get("items", [])
        return [EvaluationProfile(item) for item in items]


class EvaluationProfile:
    """Evaluation profile (custom rubric) representation."""

    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id")
        self.name = data.get("name")
        self.description = data.get("description")
        self.modality = data.get("modality")
        self.dimensions = data.get("dimensions", {})
        self.judge_prompt = data.get("judge_prompt")
        self.tags = data.get("tags", [])
        self.created_at = data.get("created_at")

    def __repr__(self):
        return f"EvaluationProfile(id='{self.id}', name='{self.name}')"
