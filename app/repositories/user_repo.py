from sqlalchemy.orm import Session
from app.models.user import User

class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def find_all(self):
        return self.db.query(User).all()

    def find_by_id(self, id: int):
        return self.db.query(User).filter(User.id == id).first()

    def find_by_email(self, email: str):
        return self.db.query(User).filter(User.email == email).first()

    def create(self, name: str, email: str, hashed_password: str):
        user = User(name=name, email=email, password=hashed_password)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def update(self, id: int, data: dict):
        self.db.query(User).filter(User.id == id).update(data)
        self.db.commit()
        return self.find_by_id(id)

    def delete(self, id: int):
        user = self.find_by_id(id)
        if user:
            self.db.delete(user)
            self.db.commit()
        return user