"""app URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from api.views import (
    ChatViewSet,
    CreateChatPDF,
    DocumentViewSet,
    MessageViewSet,
    MultipleFileDeleteViewSet,
    NoteViewSet,
    ProjectViewSet,
    landing_page,
    CreateDocumentDetails,
)
from django.contrib import admin
from django.urls import include, path
from rest_framework import routers
from users.urls import userRouter

router = routers.DefaultRouter()
router.register(r"messages", MessageViewSet, basename="messages")
router.register(r"documents", DocumentViewSet, basename="documents")
router.register(
    r"delete-multiple-files",
    MultipleFileDeleteViewSet,
    basename="delete-files",
)
router.register(r"projects", ProjectViewSet, basename="project")
router.register(r"chats", ChatViewSet, basename="chats")
router.register(r"get-chat-pdf", CreateChatPDF, basename="pdf")
router.register(r"notes", NoteViewSet, basename="notes")
router.register(r"create-doc-details", CreateDocumentDetails, basename="details")




urlpatterns = [
    path("api/", include(router.urls)),
    path("api/users/", include(userRouter.urls)),
    path("admin/", admin.site.urls),
    path("", landing_page),
]
