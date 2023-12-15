import json
import os
import re
from datetime import datetime
from io import BytesIO

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.db.models import Count

# generate pdf from hteml
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
from django.template.loader import get_template
from haystack_wrappers.qna_agent import HaystackQnAAgent
from haystack_wrappers.doc_handler import HaystackDocHandler
from haystack_wrappers.intent_classifier import HaystackIntentClassifier
from haystack_wrappers.utils import extract_referenced_documents
from rest_framework import status, viewsets
from rest_framework.authentication import (
    BasicAuthentication,
    SessionAuthentication,
)

# from django.contrib.sessions.models import Session
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated  # AllowAny,
from rest_framework.response import Response
from utils.logs import logger
from xhtml2pdf import pisa

from . import signals
from .document_reader import File, PDFFile  # noqa

# from django.http import HttpResponse
from .models import Chat, Document, DocumentQnA, Message, Note, Project
from .serializers import (
    ChatSerializer,
    ChatValidationSerializer,
    DocumentQnASerializer,
    DocumentSerializer,
    MessageSerializer,
    MessageValidationSerializer,
    llmSettingsValidationSerializer,
    NoteSerializer,
    ProjectSerializer,
)
import mimetypes   

class ProjectViewSet(viewsets.ModelViewSet):
    """
    API viewset for managing projects.

    This viewset provides endpoints to list, retrieve, and create projects
    for the authenticated user. Each project is associated with the user who
    created it. The viewset also includes annotations for the number of related
    documents and chats associated with each project.

    Permissions: Authenticated user

    Authentication: SessionAuthentication, BasicAuthentication

    Allowed HTTP methods:
    - GET: List projects
    - GET: Retrieve a specific project by ID
    - POST: Create a new project
    - DELETE: Delete a project (TODO: Delete from vector database)

    Returns:
    - 200: Success
    - 204: No content (success)
    - 400: Bad request (missing required fields or empty name)
    - 401: Not authenticated
    - 403: Forbidden
    - 404: Project not found
    """

    serializer_class = ProjectSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [SessionAuthentication, BasicAuthentication]

    def get_queryset(self):
        """
        Returns a queryset of projects filtered to only contain
        projects belonging to the current authenticated user.

        The queryset is annotated with:
        - number_of_documents: Count of related documents
        - number_of_chats: Count of related chats

        Returns:
        Django queryset object
        """
        queryset = Project.objects.annotate(
            number_of_documents=Count("document", distinct=True),
            number_of_chats=Count("chat", distinct=True),
            number_of_notes=Count("note", distinct=True),
        )
        return queryset.filter(user=self.request.user)

    def list(self, request):
        """
        List all projects for the authenticated user.

        Returns a JSON response containing a list of serialized project
        objects.

        Permissions: Authenticated user

        HTTP Methods: GET

        Returns:
        - 200: Success
        - 401: Not authenticated
        """
        queryset = self.get_queryset()

        # serialize queryset
        serializer = ProjectSerializer(queryset, many=True)

        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        """
        Get a specific project by ID.

        Returns a JSON response containing a serialized representation
        of the requested project object.

        Permissions: Authenticated user

        HTTP Methods: GET

        Returns:
        - 200: Success
        - 401: Not authenticated
        - 404: Project not found
        """
        queryset = self.get_queryset()

        # Throw 404 if project does not exist or user cannot access it
        project = get_object_or_404(queryset, pk=pk)

        # serialize queryset
        serializer = ProjectSerializer(project)

        return Response(serializer.data)

    def partial_update(self, request, *args, **kwargs):
        """
        This method is responsible for handling partial updates of objects
        in the API.

        Returns:
            200 Success
            400 if partial update failed
        Example:

        """
        instance = self.get_object()
        user = request.user
        modified_data = request.data.copy()
        # add correct user id to the requst data
        modified_data["user"] = user.id

        serializer = self.get_serializer(
            instance, data=modified_data, partial=True
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(serializer.data)

    def create(self, request):
        """
        Create a new project
        Saves the project data to the database.
        Returns the serialized project object.

        Permission: Authenticated user

        Allowed HTTP methods: POST

        Accepts: application/json

        Responds:
        - 200: Success
        - 400: If any required field is missing or if name is empty
        - 403: Not authenticated
        """
        # check if all required fields are contained in request
        serializer = ProjectSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        user = request.user
        project_name = request.data["name"]
        project_description = request.data["description"]

        # save the project in the DB
        project = Project.objects.create(
            name=project_name, description=project_description, user=user
        )

        # save it in db as another mssg with the same Session id
        serializer = ProjectSerializer(project)

        # return answer
        return Response(serializer.data)

    def destroy(self, request, pk=None):
        """
        Delete a specific project by ID.

        Permissions: Authenticated user

        HTTP Method: DELETE

        Returns:
        - 204: No content (success)
        - 401: Not authenticated
        - 403: Forbidden
        - 404: Project not found

        TODO: Delete from Documents from vector database
        """
        queryset = (
            self.get_queryset()
        )  # Fetch the project instance to be deleted

        # Throw 404 if project does not exist or user cannot access it
        project = get_object_or_404(queryset, pk=pk)

        project.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(
        methods=["get"],
        detail=True,
        permission_classes=[IsAuthenticated],
        url_path="chats",
        url_name="chats",
    )
    def get_chats(self, request, pk):
        """
        Get all chats for a project.

        Returns a list of chats associated with the project and requesting
        user.

        Permission: Authenticated user

        Allowed HTTP methods: GET

        Parameters:
        - request: Request object
        - pk: Primary key of the project

        Returns:
        - 200: Success with list of chats
        - 400: If project ID is invalid
        - 403: If user is not authenticated
        """
        user = request.user

        try:
            project = Project.objects.get(id=pk)
        except Project.DoesNotExist:
            return Response(
                {"error": "Project id is not valid or it does not exist"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        chats = Chat.objects.filter(project=project, user=user)
        serializer = ChatSerializer(chats, many=True)

        return Response(serializer.data)

    @action(
        methods=["get"],
        detail=True,
        permission_classes=[IsAuthenticated],
        url_path="documents",
        url_name="documents",
    )
    def get_documents(self, request, pk):

        """
        Get all documents for a project.

        Returns a list of documents associated with the project and requesting
        user.

        Permission: Authenticated user

        Allowed HTTP methods: GET

        Parameters:
        - request: Request object
        - pk: Primary key of the project

        Returns:
        - 200: Success with list of documents
        - 400: If project ID is invalid
        - 403: If user is not authenticated
        """
        user = request.user

        try:
            project = Project.objects.get(id=pk)
        except Project.DoesNotExist:
            return Response(
                {"error": "id is not valid or it does not exist"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        documents = Document.objects.filter(project=project, user=user)
        serializer = DocumentSerializer(documents, many=True)

        return Response(serializer.data)
    
    @action(
        methods=["get"],
        detail=True,
        permission_classes=[IsAuthenticated],
        url_path="notes",
        url_name="notes",
    )
    def get_notes(self, request, pk):
        """
        Get all notes for a project.
        Returns a list of notes associated with the project and requesting
        user.
        Permission: Authenticated user
        Allowed HTTP methods: GET
        Parameters:
        - request: Request object
        - pk: Primary key of the project
        Returns:
        - 200: Success with list of chats
        - 400: If project ID is invalid
        - 403: If user is not authenticated
        """
        user = request.user

        try:
            project = Project.objects.get(id=pk)
        except Project.DoesNotExist:
            return Response(
                {"error": "Project id is not valid or it does not exist"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        notes = Note.objects.filter(project=project, user=user)
        serializer = NoteSerializer(notes, many=True)

        return Response(serializer.data)





class ChatViewSet(viewsets.ModelViewSet):
    """
    A ViewSet that allows creating and deleting chat instances.
    Allowed HTTP methods:GET,POST,DELETE
    Usage:
        This ViewSet allows creating new chat instances using the POST method
        and deleting existing chat instances using the DELETE method. To get
        the messages for specific chat use
        http://domain/api/chats/<id>/messages/

    Example:
        To create a new chat instance, send a POST request with body
        {
        "name": "",
        "user": null, -->id
        "project": null --> id
        }
        To delete an existing chat instance, send a DELETE request to the
        endpoint with the id of the chat you want to delete

    Note:
        - Authentication is required for both creating and deleting
          chat instances.
        - Only the GET, POST, DELETE, PATCH methods are allowed for this
          ViewSet.
    """

    serializer_class = ChatSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [SessionAuthentication, BasicAuthentication]
    http_method_names = ["get", "post", "delete", "patch"]

    def get_queryset(self):
        """
        Returns a queryset of chats filtered to only contain
        chats belonging to the current authenticated user.

        Returns:
        Django queryset object
        """
        queryset = Chat.objects.all()
        return queryset.filter(user=self.request.user)

    def partial_update(self, request, *args, **kwargs):
        """
        This method is responsible for handling partial updates of
        objects in the API.

        Returns:
            200 Success
            400 if partial update failed
        Example:
            {
            "id": 8,
            "name": "Newrfffft3444",
            "project": 1
        }
        """
        instance = self.get_object()
        serializer = self.get_serializer(
            instance, data=request.data, partial=True
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

    @action(
        methods=["get"],
        detail=True,
        permission_classes=[IsAuthenticated],
        url_path="messages",
        url_name="messages",
    )
    def get_messages(self, request, pk=None):
        """
        Get all messages for a chat.

        Returns a list of messages associated with the chat and
        requesting user, ordered by created date.

        Permission: Authenticated user

        Allowed HTTP methods: GET

        Parameters:
        - request: Request object
        - pk: Primary key of the chat

        Returns:
        - 200: Success with list of messages
        - 400: If chat ID is invalid
        - 403: If user is not authenticated
        """
        user = request.user

        try:
            chat = Chat.objects.get(id=pk, user=user)
        except Chat.DoesNotExist:
            return Response(
                {"error": "chat id is not valid or it does not exist"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        messages = Message.objects.filter(chat=chat).order_by("created_at")

        serializer = MessageSerializer(messages, many=True)

        return Response(serializer.data)

    def create(self, requst):
        serializer = ChatValidationSerializer(data=requst.data)
        if serializer.is_valid():
            # name = serializer.validated_data.get("name")
            project_id = serializer.validated_data.get("project")
            user = requst.user
        else:
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )
        project = get_object_or_404(
            Project.objects.filter(user=user, id=project_id)
        )
        current_datetime = datetime.now()

        dateName = f"{current_datetime.day}.{current_datetime.month}.{current_datetime.year}|{current_datetime.hour}:{current_datetime.minute}:{current_datetime.second}"  # noqa
        chat = Chat.objects.create(name=dateName, user=user, project=project)
        serializer = ChatSerializer(chat)

        return Response(serializer.data)


class MessageViewSet(viewsets.ModelViewSet):
    """
    ViewSet for handling message creation and answer generation using Haystack.

    Allow POST:
        Create a new message and generate an answer using Haystack.


    content-type: application/json

    Body example:
    {
    "content":"demo message",
     "ids": ["2","5"]  #ids of documents to consider for the query
     "chat_id": "1"
     "settings":{
        "temperature": 0.3,
        "LLM": "gpt-3.5"
        "answerLength": "short"
     }
    }

    Returns:
    - 200: Success
    - 400: Bad request (missing required fields )
    - 401: Not authenticated
    - 403: Forbidden
    """

    permission_classes = [IsAuthenticated]
    serializer_class = MessageSerializer
    authentication_classes = [SessionAuthentication, BasicAuthentication]
    http_method_names = ["post","patch", "get",'delete']

    def get_queryset(self):
        """
        Returns all messages

        Returns:
        Django queryset object
        """
        return Message.objects.all()
    
    def destroy(self, request, pk=None):
        """
        Delete a specific Question-Answer-pair by ID.

        Permissions: Authenticated user

        HTTP Method: DELETE

        Returns:
        - 204: No content (success)
        - 401: Not authenticated
        - 403: Forbidden
        - 404: question not found


        """
        queryset = (
            self.get_queryset()
        )  # Fetch the message instance to be deleted

        # Throw 404 if answer does not exist or user cannot access it
        answer = get_object_or_404(queryset, pk=pk)
        #get question
        question = queryset.filter(answer=answer)
        #delete the answer pair if it exists
        try:
            answer.delete()
        except:
            pass
        question.delete()
       
        

        #question.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    
    def retrieve(self, request, pk=None):
        """
        Get a specific message by ID.

        Returns a JSON response containing a serialized representation
        of the requested message object.

        Permissions: Authenticated user

        HTTP Methods: GET

        Returns:
        - 200: Success
        - 401: Not authenticated
        - 404: message not found
        """
        queryset = self.get_queryset()

        # Throw 404 if message does not exist or user cannot access it
        message = get_object_or_404(queryset, pk=pk)

        # serialize queryset
        serializer = MessageSerializer(message)

        return Response(serializer.data)

    def create(self, request):
        logger.info("Receiving message...")
        serializer = MessageValidationSerializer(data=request.data)
        if serializer.is_valid():
            content = serializer.validated_data.get("content")
            doc_ids = serializer.validated_data.get("ids")
            chat_id = serializer.validated_data.get("chat_id")
            settings = serializer.validated_data.get("settings")
            print(settings)
        else:
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )

        chat = get_object_or_404(
            Chat.objects.filter(user=request.user), id=chat_id
        )

        documents = Document.objects.filter(id__in=doc_ids)

        # Convert the queryset to a list of dictionaries
        data = []
        for item in documents:
            data.append(
                {"id": item.id, "name": item.name, "file": item.file.url}
            )

        # Convert the list of dictionaries to a JSON string
        json_string = json.dumps(data)

        logger.info("Save user message to relational DB...")
        userMsg = Message.objects.create(
            chat=chat,
            msg_type=0,
            content=content,
            selected_documents_string=json_string,
        )

        # Determine model parameters
        chat_history = Message.get_last_n_messages(chat_id)
        temperature = settings['temperature']
        llm = settings['LLM']
        answer_lenght = settings["answerLength"]
        logger.info(f"llm: {llm}, temperature: {temperature}, answer_lenght: {answer_lenght}")

        # Run Haystack Pipeline to generate answer
        logger.info("Initiate HaystackIntentClassifier...")
        intent_classifier = HaystackIntentClassifier(
            model_name=llm,
            temparature=temperature,
            answer_length=answer_lenght,
            chat_history=chat_history,
        )
        haystack_wrapper = intent_classifier.classify_intent(query=content)

        # Initialize Haystackgent
        logger.info("Start Haystack Agent and inject memory...")
        haystack_agent = haystack_wrapper(
            model_name=llm,
            temparature=temperature,
            answer_length=answer_lenght,
            chat_history=chat_history,
        )

        # Run Haystack Agent to generate answer
        answer, transcript = haystack_agent.run(
            query=content,
            doc_ids=doc_ids,
        )

        answer, referenced_docs = extract_referenced_documents(answer)
        answer_formatted = re.sub(r'\n', r'\\n', answer) 

        # Filter the model by the list of IDs
        filtered_objects = Document.objects.filter(id__in=referenced_docs)

        # Create a dictionary for each filtered object
        result = []
        for obj in filtered_objects:
            obj_dict = {
                'id': obj.id,
                'name': obj.name,
                'file': obj.file.url,
                # Add more fields as needed
            }
            result.append(obj_dict)
        # Convert the list of dictionaries to a JSON string
        json_string = json.dumps(result)
        
        logger.info("Save LLM message to relational DB...")
        chatbotMessage = Message.objects.create(
            chat=chat,
            msg_type=1,
            content=answer_formatted,
            selected_documents_string=json_string,
            transcript=transcript,
        )
        # Update the answer attribute in order to
        # relate the user message with the answer
        userMsg.answer = chatbotMessage
        # Save the updated object back to the database
        userMsg.save()
        serializer = MessageSerializer(chatbotMessage)

        # return answer
        return Response(serializer.data)
    
    def partial_update(self, request, *args, **kwargs):
        """
        This method is responsible for handling partial updates of objects
        in the API.

        Returns:
            200 Success
            400 if partial update failed
        Example:

        """
        instance = self.get_object()

        serializer = self.get_serializer(
            instance, data=request.data, partial=True
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(serializer.data)


class DocumentViewSet(viewsets.ModelViewSet):
    """
    API  ViewSet for creating  and retrieving files .


    Permissions: Authenticated user

    Authentication: SessionAuthentication, BasicAuthentication

    Allowed HTTP methods:
    - GET: retrieves a specific document or a list of documents http://localhost:8000/api/documents/<pk:integer>/  # noqa
    - POST: create a file that belong to a project
    content-type: form-data

    form data example:
       file : PDF, DOCX or TXT file
       project_id : ID of the project where the file should be saved
    Returns:
    - 200: Success
    - 204: No content (success)
    - 400: Bad request (missing required fields or empty name)
    - 401: Not authenticated
    - 403: Forbidden
    """

    permission_classes = [IsAuthenticated]
    authentication_classes = [SessionAuthentication, BasicAuthentication]
    serializer_class = DocumentSerializer
    http_method_names = ["post", "get"]
    pagination_class = None

    def get_queryset(self):
        queryset = Document.objects.all()
        return queryset.filter(user=self.request.user)

    def create(self, request):
        if "file" in request.FILES and "project_id" in request.data:
            logger.info("Receiving documents...")
            # check docx
            uploaded_file = request.FILES["file"]
            project_id = request.data["project_id"]
            # Access the file type (MIME type)
            file_type = uploaded_file.content_type
            file_content = uploaded_file.read()
            file_name = uploaded_file.name

            # organize a path for the file in bucket
            file_directory_within_bucket = (
                "user_upload_files/{user_id}".format(user_id=request.user.id)
            )

            # synthesize a full file path; note that we included the filename
            file_path_within_bucket = os.path.join(
                file_directory_within_bucket, file_name
            )
            # Save in S3 Bucket
            file_path = default_storage.save(  # noqa
                file_path_within_bucket, uploaded_file
            )

            project = get_object_or_404(
                Project.objects.filter(user=request.user), id=project_id
            )
            logger.info(f"Encode file with mime-type {file_type}...")
            file = PDFFile.create_file(file_name, file_type, file_content)

            text = file.read()

            # pass the document type
            logger.info("Save document to relational DB...")
            document = Document(
                name=file_name,
                user=request.user,
                project=project,
            )
            document.file.save(file_name, ContentFile(file_content))

            document.save()

            try:
                # Run HaystackDocHandler to store Documents
                logger.info("Initiate HaystackDocHandler...")
                haystack_doc_handler = HaystackDocHandler()
                logger.info("Upload document to Pinecone...")
                haystack_doc_handler.add_document(
                    {
                        "content": text,
                        "meta": {
                            "name": file_name,
                            "doc_type": file_type,
                            "description": "abc",
                            "doc_id": str(document.id),
                        },
                    }
                )
                logger.info("Upload to Pinecone was successful...")

            except Exception as error:
                # handle the exception
                logger.info(
                    "An exception occurred during Pinecone upload: ", error
                )
                document.delete()
                return Response(status.HTTP_400_BAD_REQUEST)
            """
            logger.info(
                "Initiate process to create information of the "
                "exploration page..."
            )
            signals.pinecone_upload_done.send(
                sender=request, doc_id=document.id, text=text
            )
            """
            serializer = DocumentSerializer(document)
            return Response(serializer.data)

        else:
            return Response(
                {"error": "file or project_id is not in the request"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    @action(
        methods=["get"],
        detail=True,
        permission_classes=[IsAuthenticated],
        url_path="qna",
        url_name="qna",
    )
    def get_qna(self, request, pk):
        """
        Get all qna for a document.

        Returns a list of qna associated with the document and requesting
        user.

        Permission: Authenticated user

        Allowed HTTP methods: GET

        Parameters:
        - request: Request object
        - pk: Primary key of the project

        Returns:
        - 200: Success with list of qna
        - 400: If project ID is invalid
        - 403: If user is not authenticated
        """

        try:
            document = Document.objects.get(id=pk)
        except Document.DoesNotExist:
            return Response(
                {"error": "id is not valid or it does not exist"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        qnas = DocumentQnA.objects.filter(document=document)
        serializer = DocumentQnASerializer(qnas, many=True)

        return Response(serializer.data)


class MultipleFileDeleteViewSet(viewsets.ViewSet):
    """
    API  ViewSet for handling multiple file deletion.


    Permissions: Authenticated user

    Authentication: SessionAuthentication, BasicAuthentication

    Allowed HTTP methods:

    - POST: Delete files that belong to a project

    content-type: application/json

    Body example:
        {
        "keys" ["1","2"]
        }

    Returns:
    - 204: No content (success)
    - 400: Bad request (missing required fields or empty name)
    - 401: Not authenticated
    - 403: Forbidden
    """

    permission_classes = [IsAuthenticated]
    authentication_classes = [SessionAuthentication, BasicAuthentication]
    http_method_names = ["post"]

    def get_queryset(self):
        queryset = Document.objects.all()
        return queryset.filter(user=self.request.user)

    def create(self, request):
        try:
            ids_to_delete = request.data["keys"]
        except:
            Response(
                {"error": "keys(ids) are not in the request body"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Delete the objects with the specified IDs and from the same user
        # that sends the request
        documents = self.get_queryset().filter(
            id__in=ids_to_delete,
        )

        if documents.exists():
            serializer = DocumentSerializer(documents, many=True)
            data = serializer.data
            logger.info(ids_to_delete)

            # Run HaystackWrappter to delete document
            logger.info("Initiate HaystackDocHandler...")
            haystack_doc_handler = HaystackDocHandler()
            logger.info("Delete document from Pinecone...")
            ids_to_delete_str = [str(i) for i in ids_to_delete]
            haystack_doc_handler.delete_document(doc_ids=ids_to_delete_str)

            documents.delete()
            logger.info("Deleted document successfully...")
        else:
            data = [{"response": "data not exist"}]
            return Response(data, status=status.HTTP_204_NO_CONTENT)
        return Response(data, status=status.HTTP_204_NO_CONTENT)


class CreateChatPDF(viewsets.ModelViewSet):
    """
    A viewset for creating PDF files from HTML templates based on chat data.

    Allowed HTTP methods:
    To download the PDF file for specifc chat:
    - GET: localhost:8080/api/get-chat-pdf/<pk:integer>/
    """

    permission_classes = [IsAuthenticated]
    authentication_classes = [SessionAuthentication, BasicAuthentication]
    http_method_names = ["get"]
    serializer_class = ChatSerializer

    def get_queryset(self):
        """
        Returns a queryset of chats filtered to only contain
        chats belonging to the current authenticated user.

        Returns:
        Django queryset object
        """
        queryset = Chat.objects.all()
        return queryset.filter(user=self.request.user)

    def generate_pdf_from_html(self, template_src, chat_name, context_dict={}):
        """
        Generates a PDF file from an HTML template.

        Args:
            template_src (str): The path or name of the HTML template file.
            chat_name (str): The name of the chat.
            context_dict (dict, optional): Additional context data for
                rendering the template. Defaults to an empty dictionary.

        Returns:
            HttpResponse: The generated PDF file as an HTTP response.

        """
        template = get_template(template_src)
        html = template.render(context_dict)
        result = BytesIO()
        pdf = pisa.pisaDocument(BytesIO(html.encode("ISO-8859-1")), result)
        if pdf.err:
            return HttpResponse(
                "Invalid PDF", status_code=400, content_type="text/plain"
            )
        response = HttpResponse(
            result.getvalue(), content_type="application/pdf"
        )
        response[
            "Content-Disposition"
        ] = f"attachment; filename={chat_name}.pdf"
        return response

    def retrieve(self, request, pk=None):
        """
        Retrieves a specific chat and its associated messages, and
        generates a PDF file.

        Args:
            request (HttpRequest): The HTTP request object.
            pk (int, optional): The primary key of the chat to retrieve.
                Defaults to None.

        Returns:
            HttpResponse: The generated PDF file as an HTTP response.

        """
        chat = get_object_or_404(Chat.objects.filter(user=request.user), id=pk)
        messages = Message.objects.filter(chat=chat).order_by("created_at")
        serializer = MessageSerializer(messages, many=True)
        context = {
            "messages": messages,
            "chat_name": chat.name,
            "serializer": serializer.data,
        }
        chat_name = chat.name
        return self.generate_pdf_from_html(
            "chat_template.html", chat_name, context
        )


def landing_page(request):
    return render(request, "landing_page.html")


class NoteViewSet(viewsets.ModelViewSet):

    permission_classes = [IsAuthenticated]
    authentication_classes = [SessionAuthentication, BasicAuthentication]
    serializer_class = NoteSerializer
    http_method_names = ["get", "post", "delete", "patch"]
    pagination_class = None

    def get_queryset(self):
        queryset = Note.objects.all()
        return queryset.filter(user=self.request.user)

    def list(self, request):
        """
        List all Notes for the authenticated user.

        Returns a JSON response containing a list of serialized project
        objects.

        Permissions: Authenticated user

        HTTP Methods: GET

        Returns:
        - 200: Success
        - 401: Not authenticated
        """
        queryset = self.get_queryset()

        # serialize queryset
        serializer = NoteSerializer(queryset, many=True)

        return Response(serializer.data)

    def create(self, request):
        """
        Create a new note.

        Validates the request data and saves a new note object.

        Permissions: Authenticated user
        HTTP Methods: POST
        Returns:
        201: Note created successfully
        400: Invalid request data
        """
        serializer = NoteSerializer(data=request.data)
        if serializer.is_valid():
            content = serializer.validated_data.get("content")
            name = serializer.validated_data.get("name")
            proj = serializer.validated_data.get("project")

        else:
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )

        project = get_object_or_404(
            Project.objects.filter(user=request.user, id=proj.id)
        )

        note = Note(
            content=content,
            name=name,
            user=request.user,
            project=project,
        )
        note.save()
        serializer = NoteSerializer(note)

        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def partial_update(self, request, *args, **kwargs):
        """
        This method is responsible for handling partial updates of objects
        in the API.

        Returns:
            200 Success
            400 if partial update failed
        Example:

        """
        instance = self.get_object()
        user = request.user
        modified_data = request.data.copy()
        # add correct user id to the requst data
        modified_data["user"] = user.id

        serializer = self.get_serializer(
            instance, data=modified_data, partial=True
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(serializer.data)

    def destroy(self, request, pk=None):
        """
        Delete a specific Note by ID.

        Permissions: Authenticated user

        HTTP Method: DELETE

        Returns:
        - 204: No content (success)
        - 401: Not authenticated
        - 403: Forbidden
        - 404: Note not found


        """
        queryset = (
            self.get_queryset()
        )  # Fetch the notes instance to be deleted

        # Throw 404 if note does not exist or user cannot access it
        note = get_object_or_404(queryset, pk=pk)

        note.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class CreateDocumentDetails(viewsets.ViewSet):
   
    permission_classes = [IsAuthenticated]
    authentication_classes = [SessionAuthentication, BasicAuthentication]
    http_method_names = ["post"]

    def get_queryset(self):
        queryset = Document.objects.all()
        return queryset.filter(user=self.request.user)
    

    def create(self, request):
      
        if "doc_id" in request.data:
            doc_id = request.data["doc_id"]   
            document = get_object_or_404(Document, pk=doc_id )
            # Access the file type (MIME type)
            file_mimetype, _ = mimetypes.guess_type(document.file.url)
            file_name = document.file.name
            file_content = document.file.read()
            file = PDFFile.create_file(file_name, file_mimetype, file_content)
            file_text_string = file.read()
           

            # Dispatch the signal that creates summary and wordcloud
            signals.pinecone_upload_done.send(
                sender=request, doc_id=document.id, text=file_text_string
            )
           

            return Response(status=status.HTTP_200_OK)

        else:
            return Response(
                {"error": "doc_id is not provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )