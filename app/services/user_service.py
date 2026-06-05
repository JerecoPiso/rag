from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.repositories.user_repo import UserRepository
from app.schemas.user import UserCreate, UserUpdate, UserLogin, TokenResponse
from app.core.security import hash_password, verify_password, create_access_token

class UserService:
    def __init__(self, db: Session):
        self.repo = UserRepository(db)

    def get_all(self):
        return self.repo.find_all()

    def get_by_id(self, id: int):
        user = self.repo.find_by_id(id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

    def create(self, data: UserCreate):
        if self.repo.find_by_email(data.email):
            raise HTTPException(status_code=400, detail="Email already exists")
        hashed = hash_password(data.password)
        return self.repo.create(data.name, data.email, hashed)

    def update(self, id: int, data: UserUpdate):
        self.get_by_id(id)
        return self.repo.update(id, data.model_dump(exclude_none=True))

    def delete(self, id: int):
        self.get_by_id(id)
        return self.repo.delete(id)

    def login(self, data: UserLogin) -> TokenResponse:
        user = self.repo.find_by_email(data.email)
        if not user or not verify_password(data.password, user.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_access_token({"sub": str(user.id)})
        return TokenResponse(access_token=token)