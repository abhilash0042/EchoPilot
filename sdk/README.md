# Belfry Labs Python SDK

Official Python SDK for the Belfry Labs Enterprise AI Safety Evaluation Platform.

## Installation

The SDK is currently consumed directly from this repository; it does not need
to be available on PyPI. To use it without installing the SDK package, add its
source directory to `PYTHONPATH`:

```bash
cd /path/to/belfry-labs
export PYTHONPATH="$PWD/python-sdk:$PYTHONPATH"
python3 -c "from belfry_labs import AsyncBelfryLabsClient; print('SDK loaded')"
```

This only exposes the local `belfry_labs` source package. Its third-party
dependencies, such as `httpx`, `pydantic`, and `tenacity`, must already be
available in the Python environment.

Alternatively, install the same local source in editable mode (this does not
download `belfry-labs-sdk` from PyPI):

```bash
python3 -m pip install -e ./python-sdk
```

## Quick Start

```python
from belfry_labs import BelfryLabsClient

client = BelfryLabsClient(
    base_url="https://app.belfrylabs.ai",
    api_key="your-api-key",
)

# List available benchmarks
benchmarks = client.evaluations.list_benchmarks(modality="text")

# Run an evaluation
evaluation = client.evaluations.create(
    project_id="proj_abc123",
    model_id="model_xyz",
    benchmarks=["toxicity", "bias_detection", "prompt_injection"],
    name="Safety check v1",
)

# Wait for results
result = client.evaluations.wait_for_completion(evaluation.id)
print(f"Score: {result.overall_score}")
```

## Authentication

The SDK supports two authentication methods:

| Method | Use Case | Header |
|--------|----------|--------|
| API Key | CI/CD pipelines, scripts | `X-API-Key` |
| JWT Token | Interactive sessions | `Authorization: Bearer <token>` |

```python
# API Key (recommended for automation)
client = BelfryLabsClient(api_key="bk_live_...")

# JWT Token
client = BelfryLabsClient(token="eyJhbG...")
```

## Core Operations

### Projects

```python
# List projects
projects = client.projects.list()

# Get project details
project = client.projects.get("proj_abc123")
```

### Evaluations

```python
# Create and run evaluation
eval = client.evaluations.create(
    project_id="proj_abc123",
    model_id="model_xyz",
    benchmarks=["toxicity", "bias_detection"],
    name="Nightly safety check",
)

# Poll for completion
completed = client.evaluations.wait_for_completion(eval.id, timeout=3600)

# Get detailed results
results = client.evaluations.get_results(eval.id)
```

### Custom Evaluations (BYOE)

Register your own evaluation rubrics alongside built-in benchmarks:

```python
# Register a custom evaluation profile
profile = client.evaluations.register_custom(
    name="hallucination_check",
    dimensions={
        "factual_accuracy": {"weight": 0.5, "description": "Are claims grounded in source material?"},
        "citation_quality": {"weight": 0.3, "description": "Are citations correct and complete?"},
        "confidence_calibration": {"weight": 0.2, "description": "Does the model express appropriate uncertainty?"},
    },
    judge_prompt="Evaluate the response for factual accuracy...",
)

# Run custom eval alongside built-in benchmarks
eval = client.evaluations.create(
    project_id="proj_abc123",
    model_id="model_xyz",
    benchmarks=["toxicity", "bias_detection"],
    name="Custom + built-in",
    evaluation_params={"profile_id": profile.id},
)
```

### Security Scanning

```python
from belfry_labs import SecurityClient

security = SecurityClient(api_key="your-api-key")

# Scan a repository
scan = security.scan_repository(
    repo_url="https://github.com/org/repo",
    branch="main",
)
```

### Runtime Protection

Check every boundary around an LLM call:

```python
from belfry_labs import AsyncBelfryLabsClient

belfry = AsyncBelfryLabsClient(api_key="bak_...", tenant_id="tenant_...")

input_decision = await belfry.check_input(user_message, project_id="proj_abc")
if input_decision["action"].lower() in {"block", "terminate", "require_approval"}:
    return "Request blocked"

safe_input = input_decision.get("processed_content") or user_message
response = await llm.complete(safe_input)

output_decision = await belfry.check_output(response, project_id="proj_abc")
if output_decision["action"].lower() in {"block", "terminate", "require_approval"}:
    return "Response blocked"
return output_decision.get("processed_content") or response
```

Agent applications should call `check_agent()` before executing each tool.
See `examples/python/runtime_security_reference` for a runnable, tested example.

## Framework Integrations

The SDK ships with first-class integrations:

```python
# LangChain
from belfry_labs.integrations.langchain import BelfryCallbackHandler
handler = BelfryCallbackHandler(api_key="...")

# OpenAI
from belfry_labs.integrations.openai import BelfryOpenAIWrapper
wrapped = BelfryOpenAIWrapper(api_key="...", project_id="proj_abc")
```

## CLI

```bash
# Validate connection
belfry-labs validate --api-key bk_live_...

# Run evaluation from CLI
belfry-labs eval run --project proj_abc --benchmarks toxicity,bias

# List benchmarks
belfry-labs eval benchmarks --modality text
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `BELFRY_API_KEY` | API key for authentication | — |
| `BELFRY_BASE_URL` | Platform API URL | `https://app.belfrylabs.ai` |
| `BELFRY_TIMEOUT` | Request timeout (seconds) | `30` |

## Data Privacy

Belfry Labs processes your code only to generate safety findings. We do not store
your source code beyond the scan session. We do not use your code, models, or
evaluation results to train any AI system. All data is tenant-isolated and
encrypted at rest.

## License

MIT — see [LICENSE](LICENSE) for details.
