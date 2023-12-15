import json

from api.models import Chat, Document, DocumentQnA, Message, Note, Project
from rest_framework import serializers


class ProjectSerializer(serializers.ModelSerializer):
    number_of_documents = serializers.IntegerField(read_only=True)
    number_of_chats = serializers.IntegerField(read_only=True)
    number_of_notes = serializers.IntegerField(read_only=True)
    class Meta:
        model = Project
        fields = "__all__"
        extra_fields = ["number_of_chats", "number_of_documents","number_of_notes"]


class ChatSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chat
        fields = "__all__"


class SelectedDocumentSerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField()
    file = serializers.CharField()


class MessageSerializer(serializers.ModelSerializer):
    selected_docs = serializers.SerializerMethodField("selected_documents")

    def selected_documents(self, obj):
        docs = obj.selected_documents_string
        json_data = json.loads(docs)
        serializer = SelectedDocumentSerializer(data=json_data, many=True)

        serializer.is_valid(raise_exception=True)
        return serializer.data

    class Meta:
        model = Message
        fields = "__all__"

class llmSettingsValidationSerializer(serializers.Serializer):
    temperature = serializers.FloatField()
    LLM         = serializers.CharField()
    answerLength= serializers.CharField()

class MessageValidationSerializer(serializers.Serializer):
    content = serializers.CharField()
    ids = serializers.ListField(child=serializers.CharField())
    chat_id = serializers.IntegerField()
    settings = llmSettingsValidationSerializer()
    



class ChatValidationSerializer(serializers.Serializer):
    name = serializers.CharField()
    project = serializers.IntegerField()


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = "__all__"


class DocumentQnASerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentQnA
        fields = "__all__"


class NoteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Note
        fields = "__all__"
