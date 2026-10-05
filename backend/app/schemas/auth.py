from typing import Literal
from pydantic import BaseModel, EmailStr, Field

Role = Literal['administrator', 'analyst', 'viewer']
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)
class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)
class AuthResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'
    expires_in: int
    role: Role
class UserResponse(BaseModel):
    id: str
    email: EmailStr
    role: Role
    is_active: bool
