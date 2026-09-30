"""
Belfry Labs Utilities
Enterprise-grade utility functions for AI safety evaluation
"""

import asyncio
import logging
import json
import hashlib
import hmac
import base64
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict
import aiohttp
from pathlib import Path

try:
    import aiofiles
except ImportError:  # optional; AsyncBelfryLabsClient does not need file IO
    aiofiles = None

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataFormat(Enum):
    """Data formats"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    YAML = "yaml"
    TEXT = "text"


@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


@dataclass
class ProcessingResult:
    """Processing result"""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    processing_time: float
    errors: List[str]


class DataProcessor:
    """
    Enterprise-grade data processing utilities
    """
    
    def __init__(self):
        self.processing_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    async def process_json(
        self,
        data: Union[str, Dict[str, Any]],
        validate: bool = True
    ) -> ProcessingResult:
        """Process JSON data"""
        
        start_time = datetime.utcnow()
        
        try:
            # Parse JSON if string
            if isinstance(data, str):
                parsed_data = json.loads(data)
            else:
                parsed_data = data
            
            # Validate if requested
            errors = []
            if validate:
                validation_result = self.validate_json_structure(parsed_data)
                if not validation_result.is_valid:
                    errors.extend(validation_result.errors)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ProcessingResult(
                success=len(errors) == 0,
                data=parsed_data,
                metadata={"format": "json", "size": len(str(parsed_data))},
                processing_time=processing_time,
                errors=errors
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            return ProcessingResult(
                success=False,
                data=None,
                metadata={"format": "json"},
                processing_time=processing_time,
                errors=[str(e)]
            )
    
    async def process_csv(
        self,
        data: str,
        delimiter: str = ",",
        has_header: bool = True
    ) -> ProcessingResult:
        """Process CSV data"""
        
        start_time = datetime.utcnow()
        
        try:
            import csv
            from io import StringIO
            
            # Parse CSV
            csv_reader = csv.DictReader(StringIO(data), delimiter=delimiter)
            rows = list(csv_reader)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                data=rows,
                metadata={"format": "csv", "rows": len(rows), "columns": len(rows[0]) if rows else 0},
                processing_time=processing_time,
                errors=[]
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            return ProcessingResult(
                success=False,
                data=None,
                metadata={"format": "csv"},
                processing_time=processing_time,
                errors=[str(e)]
            )
    
    def validate_json_structure(self, data: Any) -> ValidationResult:
        """Validate JSON structure"""
        
        errors = []
        warnings = []
        
        if not isinstance(data, dict):
            errors.append("Root element must be a dictionary")
            return ValidationResult(False, errors, warnings, {})
        
        # Check for required fields
        required_fields = ["id", "timestamp"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check data types
        if "timestamp" in data:
            try:
                datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
            except ValueError:
                errors.append("Invalid timestamp format")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={"validated_at": datetime.utcnow().isoformat()}
        )
    
    async def convert_format(
        self,
        data: Any,
        from_format: DataFormat,
        to_format: DataFormat
    ) -> ProcessingResult:
        """Convert data between formats"""
        
        start_time = datetime.utcnow()
        
        try:
            # Convert to intermediate format
            if from_format == DataFormat.JSON:
                if isinstance(data, str):
                    intermediate = json.loads(data)
                else:
                    intermediate = data
            elif from_format == DataFormat.CSV:
                # Convert CSV to list of dicts
                import csv
                from io import StringIO
                csv_reader = csv.DictReader(StringIO(data))
                intermediate = list(csv_reader)
            else:
                intermediate = data
            
            # Convert to target format
            if to_format == DataFormat.JSON:
                result_data = json.dumps(intermediate, indent=2)
            elif to_format == DataFormat.CSV:
                if isinstance(intermediate, list) and intermediate:
                    import csv
                    from io import StringIO
                    output = StringIO()
                    writer = csv.DictWriter(output, fieldnames=intermediate[0].keys())
                    writer.writeheader()
                    writer.writerows(intermediate)
                    result_data = output.getvalue()
                else:
                    result_data = ""
            else:
                result_data = str(intermediate)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                data=result_data,
                metadata={
                    "from_format": from_format.value,
                    "to_format": to_format.value,
                    "size": len(str(result_data))
                },
                processing_time=processing_time,
                errors=[]
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            return ProcessingResult(
                success=False,
                data=None,
                metadata={"from_format": from_format.value, "to_format": to_format.value},
                processing_time=processing_time,
                errors=[str(e)]
            )


class FileManager:
    """
    Enterprise-grade file management utilities
    """
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    async def save_file(
        self,
        filename: str,
        content: Union[str, bytes, Dict[str, Any]],
        subdirectory: Optional[str] = None
    ) -> str:
        """Save file to disk"""
        
        try:
            # Create subdirectory if specified
            if subdirectory:
                file_path = self.base_path / subdirectory / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                file_path = self.base_path / filename
            
            # Handle different content types
            if isinstance(content, dict):
                content = json.dumps(content, indent=2)
                if not filename.endswith('.json'):
                    filename += '.json'
                    file_path = file_path.with_suffix('.json')
            
            if aiofiles is None:
                raise ImportError("aiofiles is required for FileManager.save_file")
            # Save file
            if isinstance(content, str):
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(content)
            else:
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(content)
            
            logger.info(f"Saved file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save file {filename}: {str(e)}")
            raise
    
    async def load_file(
        self,
        filename: str,
        subdirectory: Optional[str] = None
    ) -> Union[str, bytes]:
        """Load file from disk"""
        
        try:
            if subdirectory:
                file_path = self.base_path / subdirectory / filename
            else:
                file_path = self.base_path / filename
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if aiofiles is None:
                raise ImportError("aiofiles is required for FileManager.load_file")
            
            # Try to load as text first
            try:
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                return content
            except UnicodeDecodeError:
                # Load as binary
                async with aiofiles.open(file_path, 'rb') as f:
                    content = await f.read()
                return content
                
        except Exception as e:
            logger.error(f"Failed to load file {filename}: {str(e)}")
            raise
    
    async def list_files(
        self,
        subdirectory: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> List[str]:
        """List files in directory"""
        
        try:
            if subdirectory:
                directory = self.base_path / subdirectory
            else:
                directory = self.base_path
            
            if not directory.exists():
                return []
            
            files = []
            for file_path in directory.iterdir():
                if file_path.is_file():
                    if pattern is None or file_path.match(pattern):
                        files.append(file_path.name)
            
            return sorted(files)
            
        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}")
            raise
    
    async def delete_file(
        self,
        filename: str,
        subdirectory: Optional[str] = None
    ) -> bool:
        """Delete file from disk"""
        
        try:
            if subdirectory:
                file_path = self.base_path / subdirectory / filename
            else:
                file_path = self.base_path / filename
            
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete file {filename}: {str(e)}")
            raise


class HTTPClient:
    """
    Enterprise-grade HTTP client utilities
    """
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make GET request"""
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                return {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "content": await response.text(),
                    "url": str(response.url)
                }
        except Exception as e:
            logger.error(f"GET request failed: {str(e)}")
            raise
    
    async def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make POST request"""
        
        try:
            async with self.session.post(
                url, 
                data=data, 
                json=json_data, 
                headers=headers
            ) as response:
                return {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "content": await response.text(),
                    "url": str(response.url)
                }
        except Exception as e:
            logger.error(f"POST request failed: {str(e)}")
            raise


class CryptoUtils:
    """
    Cryptographic utilities
    """
    
    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """Generate secure API key"""
        
        import secrets
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash password with salt"""
        
        if salt is None:
            salt = secrets.token_hex(16)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        return base64.b64encode(password_hash).decode('utf-8'), salt
    
    @staticmethod
    def verify_password(password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash"""
        
        computed_hash, _ = CryptoUtils.hash_password(password, salt)
        return hmac.compare_digest(password_hash, computed_hash)
    
    @staticmethod
    def sign_data(data: str, secret: str) -> str:
        """Sign data with HMAC"""
        
        signature = hmac.new(
            secret.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    @staticmethod
    def verify_signature(data: str, signature: str, secret: str) -> bool:
        """Verify HMAC signature"""
        
        expected_signature = CryptoUtils.sign_data(data, secret)
        return hmac.compare_digest(signature, expected_signature)


class ValidationUtils:
    """
    Data validation utilities
    """
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address"""
        
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL"""
        
        import re
        pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_uuid(uuid_string: str) -> bool:
        """Validate UUID"""
        
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_timestamp(timestamp: str) -> bool:
        """Validate ISO timestamp"""
        
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        except ValueError:
            return False


class LoggingUtils:
    """
    Logging utilities
    """
    
    @staticmethod
    def setup_logger(
        name: str,
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[str] = None
    ) -> logging.Logger:
        """Setup logger with file and console output"""
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.value.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_function_call(func: Callable) -> Callable:
        """Decorator to log function calls"""
        
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                logger.info(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {str(e)}")
                raise
        
        return wrapper


class AsyncUtils:
    """
    Asynchronous utilities
    """
    
    @staticmethod
    async def run_in_thread(func: Callable, *args, **kwargs) -> Any:
        """Run function in thread pool"""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    @staticmethod
    async def run_in_process(func: Callable, *args, **kwargs) -> Any:
        """Run function in process pool"""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    @staticmethod
    async def gather_with_limit(
        coroutines: List[Callable],
        limit: int = 10
    ) -> List[Any]:
        """Run coroutines with concurrency limit"""
        
        semaphore = asyncio.Semaphore(limit)
        
        async def limited_coro(coro):
            async with semaphore:
                return await coro
        
        return await asyncio.gather(*[limited_coro(coro) for coro in coroutines])
    
    @staticmethod
    async def timeout_after(
        coro: Callable,
        timeout_seconds: float
    ) -> Any:
        """Run coroutine with timeout"""
        
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")


class MetricsUtils:
    """
    Metrics and monitoring utilities
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        
        metric_data = {
            "name": name,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
            "tags": tags or {}
        }
        
        self.metrics[name].append(metric_data)
        
        # Keep only last 1000 entries per metric
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter"""
        
        self.counters[name] += value
        
        # Record as metric too
        self.record_metric(f"{name}_count", self.counters[name], tags)
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get metric summary"""
        
        if name not in self.metrics:
            return {"count": 0, "avg": 0, "min": 0, "max": 0}
        
        values = [m["value"] for m in self.metrics[name]]
        
        return {
            "count": len(values),
            "avg": sum(values) / len(values) if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "latest": values[-1] if values else 0
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        
        return {
            "metrics": dict(self.metrics),
            "counters": dict(self.counters)
        }


# Global utility instances
data_processor = DataProcessor()
file_manager = FileManager()
crypto_utils = CryptoUtils()
validation_utils = ValidationUtils()
logging_utils = LoggingUtils()
async_utils = AsyncUtils()
metrics_utils = MetricsUtils()


# Helper functions
def generate_uuid() -> str:
    """Generate UUID"""
    return str(uuid.uuid4())


def generate_timestamp() -> str:
    """Generate ISO timestamp"""
    return datetime.utcnow().isoformat()


def hash_string(text: str) -> str:
    """Hash string with SHA256"""
    return hashlib.sha256(text.encode()).hexdigest()


def encode_base64(data: str) -> str:
    """Encode string to base64"""
    return base64.b64encode(data.encode()).decode()


def decode_base64(data: str) -> str:
    """Decode base64 string"""
    return base64.b64decode(data).decode()


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string"""
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"
