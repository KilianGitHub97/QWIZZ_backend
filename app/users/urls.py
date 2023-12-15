from rest_framework import routers

from .views import (
    CheckSession,
    GetCSRFToken,
    UserLoginView,
    UserLogoutViewSet,
    UserRegisterViewSet,
)

userRouter = routers.DefaultRouter()
userRouter.register(r"login", UserLoginView, basename="login")
userRouter.register(r"register", UserRegisterViewSet, basename="register")
userRouter.register(r"logout", UserLogoutViewSet, basename="logout")
userRouter.register(r"get-csrf-token", GetCSRFToken, basename="get-csrf-token")
userRouter.register(r"auth-status", CheckSession, basename="auth-status")
