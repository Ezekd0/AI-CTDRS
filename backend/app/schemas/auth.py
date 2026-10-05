from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

Role = Literal['administrator', 'analyst', 'viewer']
class RegisterRequest(BaseModel):
    full_name: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("full_name", mode="before")
    @classmethod
    def validate_name(cls, value):
        return value.strip() if isinstance(value, str) else value

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
    model_config = ConfigDict(from_attributes=True)
    full_name: str | None = None
    created_at: datetime
    id: str
    email: EmailStr
    role: Role
    is_active: bool
