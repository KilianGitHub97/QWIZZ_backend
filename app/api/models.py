# from django.contrib.sessions.models import Session
from collections import defaultdict

from django.core.validators import MinLengthValidator
from django.db import models
from users.models import CustomUser


# Create your models here.
class Timestamp(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True
        ordering = ["-created_at"]


class Project(Timestamp):
    name = models.CharField(max_length=200, validators=[MinLengthValidator(1)])
    content = models.TextField(blank=True)
    description = models.CharField(max_length=200)
    user = models.ForeignKey(
        CustomUser,
        on_delete=models.CASCADE,
        related_name="created_by",
        null=True,
        blank=True,
    )

    def __str__(self):
        return self.name


class Document(Timestamp):
    STATUS = (
        (
            "Pending",
            "Pending",
        ),
        (
            "Completed",
            "Completed",
        ),
        (
            "Error",
            "Error",
        ),
    )

    name = models.CharField(max_length=200)
    summary = models.TextField(blank=True)
    word_cloud = models.ImageField(upload_to="wordclouds/", blank=True)
    file = models.FileField(blank=True)
    user = models.ForeignKey(
        CustomUser,
        on_delete=models.CASCADE,
        related_name="uploaded_by",
        null=True,
        blank=True,
    )
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, null=True, blank=True
    )
    summary_status = models.CharField(
        max_length=50, choices=STATUS, default="Pending"
    )
    word_cloud_status = models.CharField(
        max_length=50, choices=STATUS, default="Pending"
    )
    qna_status = models.CharField(
        max_length=50, choices=STATUS, default="Pending"
    )

    def __str__(self):
        return self.name


class Chat(Timestamp):
    name = models.CharField(max_length=200)
    user = models.ForeignKey(
        CustomUser,
        on_delete=models.CASCADE,
        related_name="belongs_to",
        null=True,
        blank=True,
    )
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, null=True, blank=True
    )

    def __str__(self):
        return self.name


class Message(Timestamp):
    MSG_TYPE = (
        (
            0,
            "User",
        ),
        (
            1,
            "LLM",
        ),
    )

    content = models.TextField(blank=True)
    msg_type = models.BooleanField(choices=MSG_TYPE, default=0)
    selected_documents_string = models.TextField(null=True, blank=True)
    transcript = models.TextField(null=True, blank=True)
    favourite  = models.BooleanField(default=False)
    answer = models.ForeignKey(
        "self", on_delete=models.CASCADE, null=True, blank=True
    )
    chat = models.ForeignKey(
        Chat, on_delete=models.CASCADE, null=True, blank=True
    )

    def __str__(self):
        return self.content

    @classmethod
    def get_last_n_messages(self, chat_id: int, limit=10) -> dict:
        """
        Get the last N messages for a chat.

        Args:
            chat_id (int): The chat ID to query.
            limit (int, optional): Number of messages to return. Defaults
              to 6.

        Returns:
            dict: Dictionary with keys "input" and "output" containing the
            last N message contents grouped by message type.
        """
        qs = (
            Message.objects.filter(chat=chat_id)
            .order_by("-id")[:limit]
            .values()
        )

        memory = defaultdict(list)

        for msg in qs:
            if not msg["msg_type"]:
                memory["input"].append((msg["id"], msg["content"]))
            else:
                memory["output"].append((msg["id"], msg["transcript"]))

        return memory


class Note(Timestamp):
    name = models.CharField(max_length=200)
    content = models.TextField(blank=True)
    user = models.ForeignKey(
        CustomUser,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, null=True, blank=True
    )

    def __str__(self):
        return self.name


class DocumentQnA(Timestamp):
    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, null=True, blank=True
    )
    split_id = models.IntegerField(blank=True, null=True)
    question = models.CharField(max_length=200, blank=True)
    answer = models.TextField(blank=True)

    def __str__(self):
        return str(self.document.id)
