"""
Belfry Labs Client
Main client interface for AI safety evaluation
"""

from .sync_client import BelfryLabsClient
from .async_client import AsyncBelfryLabsClient
from .security import SecurityClient
from .utils import (
    data_processor,
    file_manager,
    crypto_utils,
    validation_utils,
    logging_utils,
    async_utils,
    metrics_utils,
    generate_uuid,
    generate_timestamp,
    hash_string,
    encode_base64,
    decode_base64,
    format_bytes,
    format_duration
)

# Re-export main classes
__all__ = [
    'BelfryLabsClient',
    'AsyncBelfryLabsClient', 
    'SecurityClient',
    'data_processor',
    'file_manager',
    'crypto_utils',
    'validation_utils',
    'logging_utils',
    'async_utils',
    'metrics_utils',
    'generate_uuid',
    'generate_timestamp',
    'hash_string',
    'encode_base64',
    'decode_base64',
    'format_bytes',
    'format_duration'
]

# Version information
__version__ = "2.0.0"
__author__ = "Belfry Labs"
__description__ = "Enterprise AI Safety Evaluation Platform"