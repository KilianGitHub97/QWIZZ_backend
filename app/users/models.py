from django.contrib.auth.models import (
    AbstractBaseUser,
    BaseUserManager,
    Group,
    Permission,
    PermissionsMixin,
)
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.db import models
from django.utils.translation import gettext_lazy as _


class CustomAccountManager(BaseUserManager):
    def create_superuser(self, email, password, **other_fields):
        other_fields.setdefault("is_staff", True)
        other_fields.setdefault("is_superuser", True)
        other_fields.setdefault("is_active", True)

        if other_fields.get("is_staff") is not True:
            raise ValueError("Superuser must be assigned to is_staff=True.")
        if other_fields.get("is_superuser") is not True:
            raise ValueError(
                "Superuser must be assigned to is_superuser=True."
            )

        return self.create_user(
            username="admin", email=email, password=password, **other_fields
        )

    def create_user(self, email, password, username, **other_fields):
        if not email:
            raise ValueError(_("You must provide an email address"))

        email = self.normalize_email(email)
        user = self.model(email=email, **other_fields)
        user.username = username
        try:
            validate_password(password)
            user.set_password(password)
            user.is_active = True
            user.save()
            return user
        except ValidationError as e:
            raise ValueError(e)


class CustomUser(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(_("email address"), unique=True)
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=False)

    # Set unique related_name values
    groups = models.ManyToManyField(Group, related_name="default_users")
    user_permissions = models.ManyToManyField(
        Permission, related_name="default_users"
    )

    objects = CustomAccountManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS: list = []

    def __str__(self):
        return self.email
