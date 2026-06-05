from pydantic import BaseModel, EmailStr
from datetime import datetime

class UserCreate(BaseModel):
    name:     str
    email:    EmailStr
    password: str

class UserUpdate(BaseModel):
    name:      str | None = None
    is_active: bool | None = None

class UserResponse(BaseModel):
    # id:         int
    pid:        str
    name:       str
    email:      str
    is_active:  bool
    # created_at: datetime

    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    email:    EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"