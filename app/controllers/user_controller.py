from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.user_service import UserService
from app.schemas.user import UserCreate, UserUpdate, UserLogin

class UserController:
    @staticmethod
    def index(db: Session = Depends(get_db)):
        return UserService(db).get_all()

    @staticmethod
    def show(id: int, db: Session = Depends(get_db)):
        return UserService(db).get_by_id(id)

    @staticmethod
    def store(data: UserCreate, db: Session = Depends(get_db)):
        return UserService(db).create(data)

    @staticmethod
    def update(id: int, data: UserUpdate, db: Session = Depends(get_db)):
        return UserService(db).update(id, data)

    @staticmethod
    def destroy(id: int, db: Session = Depends(get_db)):
        return UserService(db).delete(id)

    @staticmethod
    def login(data: UserLogin, db: Session = Depends(get_db)):
        return UserService(db).login(data)