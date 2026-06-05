from fastapi import APIRouter
from app.controllers.user_controller import UserController
from app.schemas.user import UserResponse, UserCreate, UserUpdate, TokenResponse
from typing import List

router = APIRouter(prefix="/users", tags=["Users"])

router.get("/",           response_model=List[UserResponse])(UserController.index)
router.get("/{id}",       response_model=UserResponse)(UserController.show)
router.post("/",          response_model=UserResponse)(UserController.store)
router.put("/{id}",       response_model=UserResponse)(UserController.update)
router.delete("/{id}",    response_model=UserResponse)(UserController.destroy)
router.post("/login",     response_model=TokenResponse)(UserController.login)