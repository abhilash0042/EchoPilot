"""
Models management for Belfry Labs SDK
"""

from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import json

from .exceptions import APIError

class Model:
    """Model representation"""

    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id")
        self.name = data.get("name")
        self.display_name = data.get("display_name")
        self.description = data.get("description")
        self.version = data.get("version")
        self.project_id = data.get("project_id")
        self.project_name = data.get("project_name")
        self.modality = data.get("modality")
        self.model_type = data.get("model_type")
        self.provider = data.get("provider")
        self.source_type = data.get("source_type")
        self.source_url = data.get("source_url")
        self.parameters = data.get("parameters", {})
        self.capabilities = data.get("capabilities", [])
        self.tags = data.get("tags", [])
        self.safety_score = data.get("safety_score")
        self.safety_status = data.get("safety_status")
        self.status = data.get("status")
        self.file_size = data.get("file_size")
        self.created_at = data.get("created_at")
        self.last_used = data.get("last_used")

    def __repr__(self):
        return f"Model(id='{self.id}', name='{self.name}', modality='{self.modality}')"

class ModelComparison:
    """Model comparison representation"""

    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id")
        self.name = data.get("name")
        self.project_id = data.get("project_id")
        self.model_ids = data.get("model_ids", [])
        self.benchmarks = data.get("benchmarks", [])
        self.status = data.get("status")
        self.progress = data.get("progress", 0.0)
        self.results = data.get("results")
        self.created_at = data.get("created_at")

class ModelManager:
    """Manager for model operations"""

    def __init__(self, client):
        self.client = client

    def list(
        self,
        project_id: Optional[str] = None,
        modality: Optional[str] = None,
        provider: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100
    ) -> List[Model]:
        """
        List models

        Args:
            project_id: Filter by project ID
            modality: Filter by modality (text, image, audio, video, multimodal, agents)
            provider: Filter by provider
            search: Search query
            limit: Maximum results

        Returns:
            List of Model objects
        """
        params = {"limit": limit}
        if project_id:
            params["project_id"] = project_id
        if modality:
            params["modality"] = modality
        if provider:
            params["provider"] = provider
        if search:
            params["search"] = search

        response = self.client.get("/api/v1/models", params=params)
        return [Model(item) for item in response.json()]

    def get(self, model_id: str) -> Model:
        """
        Get model by ID

        Args:
            model_id: Model ID

        Returns:
            Model object
        """
        response = self.client.get(f"/api/v1/models/{model_id}")
        return Model(response.json())

    def upload(
        self,
        project_id: str,
        file_path: Union[str, Path],
        name: str,
        modality: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        version: str = "1.0",
        model_type: Optional[str] = None,
        parameters: Optional[Dict] = None,
        capabilities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> Model:
        """
        Upload model file

        Args:
            project_id: Project ID
            file_path: Path to model file
            name: Model name
            modality: Model modality
            display_name: Display name
            description: Model description
            version: Model version
            model_type: Model type
            parameters: Model parameters
            capabilities: Model capabilities
            tags: Model tags

        Returns:
            Model object
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise APIError(f"File not found: {file_path}")

        data = {
            "project_id": project_id,
            "name": name,
            "modality": modality,
            "version": version
        }

        if display_name:
            data["display_name"] = display_name
        if description:
            data["description"] = description
        if model_type:
            data["model_type"] = model_type
        if parameters:
            data["parameters"] = json.dumps(parameters)
        if capabilities:
            data["capabilities"] = json.dumps(capabilities)
        if tags:
            data["tags"] = json.dumps(tags)

        files = {"file": open(file_path, "rb")}

        try:
            response = self.client.post("/api/v1/models/upload", data=data, files=files)
            return Model(response.json())
        finally:
            files["file"].close()

    def add_api_model(
        self,
        project_id: str,
        name: str,
        modality: str,
        api_endpoint: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        version: str = "1.0",
        provider: str = "custom",
        api_key_required: bool = True,
        headers: Optional[Dict] = None,
        rate_limits: Optional[Dict] = None,
        cost_per_token: Optional[float] = None,
        parameters: Optional[Dict] = None,
        capabilities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> Model:
        """
        Add API-based model

        Args:
            project_id: Project ID
            name: Model name
            modality: Model modality
            api_endpoint: API endpoint URL
            display_name: Display name
            description: Model description
            version: Model version
            provider: Model provider
            api_key_required: Whether API key is required
            headers: Custom headers
            rate_limits: Rate limit configuration
            cost_per_token: Cost per token
            parameters: Model parameters
            capabilities: Model capabilities
            tags: Model tags

        Returns:
            Model object
        """
        model_data = {
            "name": name,
            "display_name": display_name,
            "description": description,
            "version": version,
            "project_id": project_id,
            "modality": modality,
            "provider": provider,
            "source_type": "api",
            "parameters": parameters or {},
            "capabilities": capabilities or [],
            "tags": tags or []
        }

        api_config = {
            "api_endpoint": api_endpoint,
            "api_key_required": api_key_required,
            "headers": headers or {},
            "rate_limits": rate_limits or {},
            "cost_per_token": cost_per_token
        }

        data = {
            "model_data": model_data,
            "api_config": api_config
        }

        response = self.client.post("/api/v1/models/api", data=data)
        return Model(response.json())

    def import_from_registry(
        self,
        project_id: str,
        registry_type: str,
        model_path: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        modality: str = "text"
    ) -> Model:
        """
        Import model from external registry

        Args:
            project_id: Project ID
            registry_type: Registry type (huggingface, github, artifactory)
            model_path: Model path in registry
            name: Model name (defaults to model_path basename)
            description: Model description
            modality: Model modality

        Returns:
            Model object
        """
        data = {
            "registry_type": registry_type,
            "model_path": model_path,
            "project_id": project_id,
            "name": name,
            "description": description,
            "modality": modality
        }

        response = self.client.post("/api/v1/registry/import", data=data)
        return {"message": response.json().get("message"), "model_id": response.json().get("model_id")}

    def compare(
        self,
        project_id: str,
        model_ids: List[str],
        benchmarks: List[str],
        name: str,
        description: Optional[str] = None,
        datasets: Optional[List[str]] = None
    ) -> ModelComparison:
        """
        Create model comparison

        Args:
            project_id: Project ID
            model_ids: List of model IDs to compare
            benchmarks: List of benchmarks to run
            name: Comparison name
            description: Comparison description
            datasets: Specific datasets to use

        Returns:
            ModelComparison object
        """
        data = {
            "name": name,
            "description": description,
            "project_id": project_id,
            "model_ids": model_ids,
            "benchmarks": benchmarks,
            "datasets": datasets or []
        }

        response = self.client.post("/api/v1/models/compare", data=data)
        return ModelComparison(response.json())

    def get_comparison(self, comparison_id: str) -> Dict[str, Any]:
        """
        Get comparison results

        Args:
            comparison_id: Comparison ID

        Returns:
            Comparison results
        """
        response = self.client.get(f"/api/v1/models/compare/{comparison_id}")
        return response.json()

    def delete(self, model_id: str) -> Dict[str, str]:
        """
        Delete model

        Args:
            model_id: Model ID

        Returns:
            Deletion confirmation
        """
        response = self.client.delete(f"/api/v1/models/{model_id}")
        return response.json()
