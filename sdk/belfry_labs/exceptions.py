"""
Exception classes for Belfry Labs SDK.
"""


class BelfryLabsError(Exception):
    """Base exception for Belfry Labs SDK."""
    
    def __init__(self, message: str, status_code: int = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}
    
    def __str__(self):
        return self.message


class AuthenticationError(BelfryLabsError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class NotFoundError(BelfryLabsError):
    """Raised when a resource is not found."""
    
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)


class ValidationError(BelfryLabsError):
    """Raised when request validation fails."""
    
    def __init__(self, message: str = "Validation error", details: dict = None):
        super().__init__(message, status_code=422, details=details)


class RateLimitError(BelfryLabsError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)


class ServerError(BelfryLabsError):
    """Raised when server returns an error."""
    
    def __init__(self, message: str = "Server error", status_code: int = 500):
        super().__init__(message, status_code=status_code)


class TimeoutError(BelfryLabsError):
    """Raised when an operation times out."""
    
    def __init__(self, message: str = "Operation timed out"):
        super().__init__(message)


class ConfigurationError(BelfryLabsError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str = "Configuration error"):
        super().__init__(message)
