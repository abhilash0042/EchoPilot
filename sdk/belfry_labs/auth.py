"""
Authentication module for Belfry Labs SDK

Handles authentication, token management, and user session management
for the Belfry Labs Python SDK.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import httpx
from pydantic import BaseModel

from .exceptions import AuthenticationError, ValidationError


class TokenInfo(BaseModel):
    """Token information model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    expires_at: Optional[datetime] = None
    user: Optional[Dict[str, Any]] = None
    tenant: Optional[Dict[str, Any]] = None


class AuthManager:
    """Manages authentication for Belfry Labs SDK"""
    
    def __init__(self, base_url: str, client: Optional[httpx.AsyncClient] = None):
        self.base_url = base_url.rstrip('/')
        self.client = client
        self._token_info: Optional[TokenInfo] = None
        self._should_close_client = client is None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client, creating one if needed"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    @client.setter
    def client(self, value: Optional[httpx.AsyncClient]):
        """Set HTTP client"""
        self._client = value
    
    @property 
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        if not self._token_info:
            return False
        
        if self._token_info.expires_at:
            return datetime.utcnow() < self._token_info.expires_at
        
        # If no expires_at, check expires_in
        if hasattr(self._token_info, '_created_at') and self._token_info.expires_in:
            created_at = getattr(self._token_info, '_created_at')
            expires_at = created_at + timedelta(seconds=self._token_info.expires_in)
            return datetime.utcnow() < expires_at
        
        return True
    
    @property
    def access_token(self) -> Optional[str]:
        """Get current access token"""
        if self._token_info and self.is_authenticated:
            return self._token_info.access_token
        return None
    
    @property
    def tenant_id(self) -> Optional[str]:
        """Get current tenant ID"""
        if self._token_info and self._token_info.tenant:
            return self._token_info.tenant.get('id')
        return None
    
    @property
    def user_id(self) -> Optional[str]:
        """Get current user ID"""
        if self._token_info and self._token_info.user:
            return self._token_info.user.get('id')
        return None
    
    async def login(
        self, 
        email: str, 
        password: str,
        tenant_subdomain: Optional[str] = None
    ) -> TokenInfo:
        """
        Authenticate user with email/password
        
        Args:
            email: User email address
            password: User password
            tenant_subdomain: Optional tenant subdomain
            
        Returns:
            TokenInfo with authentication details
            
        Raises:
            AuthenticationError: If authentication fails
            ValidationError: If credentials are invalid
        """
        if not email or not password:
            raise ValidationError("Email and password are required")
        
        login_data = {
            "email": email,
            "password": password
        }
        
        if tenant_subdomain:
            login_data["tenant_subdomain"] = tenant_subdomain
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/auth/login",
                json=login_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 401:
                raise AuthenticationError("Invalid credentials")
            elif response.status_code == 404:
                raise AuthenticationError("Tenant not found")
            elif response.status_code != 200:
                raise AuthenticationError(f"Login failed: {response.text}")
            
            token_data = response.json()
            self._token_info = TokenInfo(**token_data)
            
            # Store creation time for token expiry calculation
            self._token_info._created_at = datetime.utcnow()
            
            if self._token_info.expires_in:
                self._token_info.expires_at = (
                    self._token_info._created_at + 
                    timedelta(seconds=self._token_info.expires_in)
                )
            
            return self._token_info
            
        except httpx.RequestError as e:
            raise AuthenticationError(f"Network error during login: {str(e)}")
        except Exception as e:
            if isinstance(e, (AuthenticationError, ValidationError)):
                raise
            raise AuthenticationError(f"Login failed: {str(e)}")
    
    async def register(
        self,
        email: str,
        username: str,
        password: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        tenant_subdomain: Optional[str] = None,
        tenant_name: Optional[str] = None
    ) -> TokenInfo:
        """
        Register new user account
        
        Args:
            email: User email address
            username: Unique username
            password: User password
            first_name: Optional first name
            last_name: Optional last name
            tenant_subdomain: Optional tenant subdomain (creates new tenant)
            tenant_name: Optional tenant name (required if creating tenant)
            
        Returns:
            TokenInfo with authentication details
            
        Raises:
            AuthenticationError: If registration fails
            ValidationError: If input data is invalid
        """
        if not all([email, username, password]):
            raise ValidationError("Email, username, and password are required")
        
        register_data = {
            "email": email,
            "username": username,
            "password": password
        }
        
        if first_name:
            register_data["first_name"] = first_name
        if last_name:
            register_data["last_name"] = last_name
        if tenant_subdomain:
            register_data["tenant_subdomain"] = tenant_subdomain
            if tenant_name:
                register_data["tenant_name"] = tenant_name
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/auth/register",
                json=register_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 400:
                error_detail = response.json().get("detail", "Registration failed")
                raise ValidationError(error_detail)
            elif response.status_code != 200:
                raise AuthenticationError(f"Registration failed: {response.text}")
            
            token_data = response.json()
            self._token_info = TokenInfo(**token_data)
            
            # Store creation time for token expiry calculation
            self._token_info._created_at = datetime.utcnow()
            
            if self._token_info.expires_in:
                self._token_info.expires_at = (
                    self._token_info._created_at + 
                    timedelta(seconds=self._token_info.expires_in)
                )
            
            return self._token_info
            
        except httpx.RequestError as e:
            raise AuthenticationError(f"Network error during registration: {str(e)}")
        except Exception as e:
            if isinstance(e, (AuthenticationError, ValidationError)):
                raise
            raise AuthenticationError(f"Registration failed: {str(e)}")
    
    async def refresh_token(self) -> TokenInfo:
        """
        Refresh access token using refresh token
        
        Returns:
            New TokenInfo with refreshed tokens
            
        Raises:
            AuthenticationError: If token refresh fails
        """
        if not self._token_info or not self._token_info.refresh_token:
            raise AuthenticationError("No refresh token available")
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/auth/refresh",
                json={"refresh_token": self._token_info.refresh_token},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 401:
                # Clear stored tokens
                self._token_info = None
                raise AuthenticationError("Refresh token expired or invalid")
            elif response.status_code != 200:
                raise AuthenticationError(f"Token refresh failed: {response.text}")
            
            token_data = response.json()
            self._token_info = TokenInfo(**token_data)
            
            # Store creation time for token expiry calculation
            self._token_info._created_at = datetime.utcnow()
            
            if self._token_info.expires_in:
                self._token_info.expires_at = (
                    self._token_info._created_at + 
                    timedelta(seconds=self._token_info.expires_in)
                )
            
            return self._token_info
            
        except httpx.RequestError as e:
            raise AuthenticationError(f"Network error during token refresh: {str(e)}")
        except Exception as e:
            if isinstance(e, AuthenticationError):
                raise
            raise AuthenticationError(f"Token refresh failed: {str(e)}")
    
    async def logout(self) -> None:
        """
        Logout user and clear tokens
        """
        if self.is_authenticated:
            try:
                # Call logout endpoint to invalidate server-side tokens
                headers = {"Authorization": f"Bearer {self.access_token}"}
                await self.client.post(
                    f"{self.base_url}/api/v1/auth/logout",
                    headers=headers
                )
            except Exception:
                # Continue with logout even if server call fails
                pass
        
        # Clear local tokens
        self._token_info = None
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests
        
        Returns:
            Dictionary with authorization headers
            
        Raises:
            AuthenticationError: If not authenticated or token refresh fails
        """
        if not self.is_authenticated:
            if self._token_info and self._token_info.refresh_token:
                # Try to refresh token
                await self.refresh_token()
            else:
                raise AuthenticationError("Not authenticated")
        
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    async def get_profile(self) -> Dict[str, Any]:
        """
        Get current user profile
        
        Returns:
            User profile information
            
        Raises:
            AuthenticationError: If not authenticated
        """
        headers = await self.get_auth_headers()
        
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v1/auth/profile",
                headers=headers
            )
            
            if response.status_code == 401:
                raise AuthenticationError("Authentication required")
            elif response.status_code != 200:
                raise AuthenticationError(f"Failed to get profile: {response.text}")
            
            return response.json()
            
        except httpx.RequestError as e:
            raise AuthenticationError(f"Network error getting profile: {str(e)}")
    
    async def update_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user profile
        
        Args:
            profile_data: Profile fields to update
            
        Returns:
            Update result
            
        Raises:
            AuthenticationError: If not authenticated
            ValidationError: If profile data is invalid
        """
        headers = await self.get_auth_headers()
        
        try:
            response = await self.client.put(
                f"{self.base_url}/api/v1/auth/profile",
                json=profile_data,
                headers=headers
            )
            
            if response.status_code == 401:
                raise AuthenticationError("Authentication required")
            elif response.status_code == 400:
                raise ValidationError("Invalid profile data")
            elif response.status_code != 200:
                raise AuthenticationError(f"Failed to update profile: {response.text}")
            
            return response.json()
            
        except httpx.RequestError as e:
            raise AuthenticationError(f"Network error updating profile: {str(e)}")
    
    async def change_password(self, current_password: str, new_password: str) -> None:
        """
        Change user password
        
        Args:
            current_password: Current password
            new_password: New password
            
        Raises:
            AuthenticationError: If not authenticated or current password incorrect
            ValidationError: If new password is invalid
        """
        headers = await self.get_auth_headers()
        
        password_data = {
            "current_password": current_password,
            "new_password": new_password
        }
        
        try:
            response = await self.client.put(
                f"{self.base_url}/api/v1/auth/password",
                json=password_data,
                headers=headers
            )
            
            if response.status_code == 401:
                raise AuthenticationError("Authentication required")
            elif response.status_code == 400:
                error_detail = response.json().get("detail", "Password change failed")
                raise ValidationError(error_detail)
            elif response.status_code != 200:
                raise AuthenticationError(f"Failed to change password: {response.text}")
            
        except httpx.RequestError as e:
            raise AuthenticationError(f"Network error changing password: {str(e)}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._should_close_client and self._client:
            await self._client.aclose()


# Sync wrapper functions for convenience
def login_sync(
    base_url: str,
    email: str, 
    password: str,
    tenant_subdomain: Optional[str] = None
) -> TokenInfo:
    """Synchronous wrapper for login"""
    async def _login():
        async with AuthManager(base_url) as auth:
            return await auth.login(email, password, tenant_subdomain)
    
    return asyncio.run(_login())


def register_sync(
    base_url: str,
    email: str,
    username: str,
    password: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    tenant_subdomain: Optional[str] = None,
    tenant_name: Optional[str] = None
) -> TokenInfo:
    """Synchronous wrapper for registration"""
    async def _register():
        async with AuthManager(base_url) as auth:
            return await auth.register(
                email, username, password, first_name, last_name,
                tenant_subdomain, tenant_name
            )
    
    return asyncio.run(_register())
