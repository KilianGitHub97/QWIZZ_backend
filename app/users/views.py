from django.contrib.auth import authenticate, login, logout
from django.utils.decorators import method_decorator

from django.views.decorators.csrf import ensure_csrf_cookie
from rest_framework import status, viewsets
from rest_framework.authentication import (
    BasicAuthentication,
    SessionAuthentication,
)
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import CustomUser
from .serializers import UserLoginSerializer, UserRegisterSerializer

from api.models import Project,Document, DocumentQnA, Chat, Message, Note
from django.core.files.base import ContentFile
import json

from haystack_wrappers.doc_handler import HaystackDocHandler


class UserLoginView(viewsets.ViewSet):
    permission_classes = [AllowAny]
    authentication_classes = [SessionAuthentication, BasicAuthentication]

    def create(self, request):
        serializer = UserLoginSerializer(data=request.data)

        if serializer.is_valid():
            email = serializer.validated_data["email"]
            password = serializer.validated_data["password"]
            user = authenticate(request, email=email, password=password)

            if user is not None:
                login(request, user)
                return Response(
                    {"loggedIn": True, "email": user.email},
                    status=status.HTTP_200_OK,
                )
            else:
                return Response(
                    {"data": "Invalid credentials"},
                    status=status.HTTP_401_UNAUTHORIZED,
                )
        else:
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )


class UserRegisterViewSet(viewsets.ViewSet):
    permission_classes = [AllowAny]

    def create(self, request):
        serializer = UserRegisterSerializer(data=request.data)
        if serializer.is_valid():
            username = serializer.validated_data.get("username")
            password = serializer.validated_data.get("password")
            re_password = serializer.validated_data.get("re_password")  # noqa
            email = serializer.validated_data.get("email")
            # Check if the user already exists

            if CustomUser.objects.filter(email=email).exists():
                return Response(
                    {"data": "User with the same Email already exists"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            try:
                # Create user
                user = CustomUser.objects.create_user(
                    username=username, password=password, email=email
                )

                # Create sample project
                _create_sample_project(user)

            except ValueError as e:
                error_message = str(e)
                return Response(
                    {"error": error_message}, status=status.HTTP_400_BAD_REQUEST
                )

            serializer = UserRegisterSerializer(user)

            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

def _create_sample_project( user: CustomUser) -> None:
        """Creates the sample project including the sample document, chat and notes
        for the given user.
        
        Args:
            user (CustomUser): The user that the sample project is created for
        """
        project = Project(
            name ="Introducing Qwizz",
            description = "A sample project exploring the idea behind Qwizz!",
            user = user
        )
        project.save()

        # Read and create document
        try:
            with open("api/templates/sample_doc.txt", 'r') as file:
                text_string = file.read()                  
        except Exception as e:
            print(e)
            return Response( {"error": "document cant be read"}, status=status.HTTP_400_BAD_REQUEST) 
        
        document = Document(
            name= "Interview Qwizz",
            user= user,
            project= project,
            summary = "A computer science student from University of Zurich has developed a prototype tool called Qwizz to facilitate qualitative interview analysis for their master's thesis research. Qwizz utilizes large language models and retrieval augmentation to generate insightful analysis of interview transcripts. Users can customize settings like creativity level, model choice, and response length.The tool allows creating projects with chats, notes and documents to separate ideas. Interview data gets embedded and stored in a vector database called Pinecone. When users pose questions, a Haystack agent evaluates and determines the best pipeline to provide an answer. Tools using Haystack components are built to retrieve relevant passages and generate responses.The student used frameworks like Django, ReactJS and Haystack to develop the full-stack web application. Qwizz has significantly accelerated the analysis process, allowing quicker iteration. It also manages large datasets efficiently.However, limitations exist as qualitative analysis still requires human judgment for interpretation, contextualization, and privacy considerations. More research is needed on sensitive data. As models evolve, this technology has potential to revolutionize qualitative research across social sciences with exciting future possibilities.",
            summary_status = "Completed",
        )
        document.file.save("sample_doct.txt", ContentFile(text_string.encode()))

        # Create wordcloud for the document
        with open('api/templates/wordcloud.png', 'rb') as f:
            word_cloud_data = f.read()
            document.word_cloud.save('wordcloud.png', ContentFile(word_cloud_data))
            document.word_cloud_status= "Completed"
            document.save()

        # Upload document to vector database
        haystack_doc_handler = HaystackDocHandler()
        haystack_doc_handler.add_document(
            {
                "content": text_string,
                "meta": {
                    "name": document.name,
                    "doc_type": "txt",
                    "description": "sample document",
                    "doc_id": str(document.id),
                },
            }
        )

        # Create QnA for sample document
        qna_1= DocumentQnA(
            question= "What are the key technical components and frameworks used to develop Qwizz? ",
            answer = "The student used Django for the backend framework, ReactJS for the frontend, and Haystack as the main orchestration tool. The vector database used is Pinecone. Large language models like GPT-3.5 and GPT-4 are employed for text generation.",
            document = document
        )
        qna_1.save()

        qna_2= DocumentQnA(
            question= "What are some examples of the limitations of using Qwizz for qualitative analysis compared to manual human analysis?",
            answer = "The student mentions that qualitative analysis still requires human judgment for interpreting subtle findings and ensuring accurate contextual understanding. Qwizz may lack the subject matter expertise and nuanced perspective a human researcher has. There are also privacy concerns, especially when dealing with sensitive data",
            document = document
        )
        qna_2.save()

        qna_3= DocumentQnA(
            question= "How specifically has using Qwizz accelerated and impacted the student's research process and productivity?",
            answer = "The student mentions Qwizz has brought a remarkable acceleration to their analysis process, reducing the time to generate initial analyses from hours/days to minutes. This allows quicker iteration and more time for interpretation and contextualization. Qwizz also helps manage large datasets efficiently. ",
            document = document
        )
        qna_3.save()
        
        # Create chat 'Introducing Qwizz' (incl. messages and answers)
        chat_1 = Chat(
            name="Introducing Qwizz",
            project = project,
            user= user
        )
        chat_1.save()

        c_1_q_1 = Message.objects.create(
            content = "Hi, what is your name?", 
            msg_type = 0,
            selected_documents_string=json.dumps([ ]),
            chat = chat_1,
            favourite= False,
        )
        c_1_a_1 = Message.objects.create(
            content = """I am an AI Agent called Qwizz. How can I assist you with qualitative document analysis?""" ,
            msg_type = 1,
            selected_documents_string=json.dumps([]),
            chat =chat_1,
            favourite= False,
            transcript = """I am an AI Agent called Qwizz.\\nTool: helper_tool\\nTool Input: None\\nObservation: I am an AI agent named 'Qwizz' equipped with several capabilities to assist with qualitative document analysis. Here are some of the things I can do:\\n1. Search: I can thoroughly search through qualitative documents to find relevant information that answers a user's questions. I utilize available tools to retrieve pertinent passages across multiple documents.\\n2. Compare: I can compare and contrast what different documents or sources say about a particular topic. This allows for nuanced analysis from multiple perspectives.\\n3. Propose questions: Based on my understanding of the qualitative data, I can propose potential questions a user may want to ask to gain more insights. This helps guide the analysis in fruitful directions.\\n4. Truthful, grounded answers: I construct my answers solely based on the output of the tools used to analyze the qualitative data. My answers are truthful and reference the specific document IDs they are derived from.\\n5. Acknowledge limitations: If I cannot determine an answer from the available data, I will transparently acknowledge the limitations of my knowledge and ask the user for clarification.\\nI leverage my analytical capabilities within the boundaries of the tools provided to offer thoughtful, thorough assistance with qualitative document analysis. My goal is to provide helpful information to users while maintaining transparency. However, please note that I require inputs from the user to function properly and I am not a replacement for human analysis. I am a tool to assist with qualitative document analysis.\\nFor technical details, I use LLM Models such as GPT-3.5 and GPT-4 for document analysis. The orchestration of these models is handled by Haystack. The backend of my system uses the Django framework, while the frontend is built with ReactJS. I am deployed on DigitalOcean and Heroku, with data stored in a PostgreSQL database on our DigitalOcean server. Uploaded documents are stored in an AWS S3 bucket, and a Pinecone vector database hosted in Singapore is used for efficient document retrieval.\\nIf you have any questions or need assistance with qualitative document analysis, feel free to ask!\\nThought:I have provided the user with an overview of my capabilities and technical details. Now I can proceed to answer their question.\\nFinal Answer: I am an AI Agent called Qwizz. How can I assist you with qualitative document analysis?"""
        )
        c_1_q_1.answer = c_1_a_1
        c_1_q_1.save()

        c_1_q_2 = Message.objects.create(
            content = "Please tell me what you can do", 
            msg_type = 0,
            selected_documents_string=json.dumps([ ]),
            favourite= False,
            chat = chat_1,
        )
        c_1_a_2 = Message.objects.create(
            content = """I am an AI agent equipped with several capabilities to assist with qualitative document analysis. Here's what I can do:\\n 1. Search: I can thoroughly search through qualitative documents to find relevant information that answers your questions. I utilize available tools to retrieve pertinent passages across multiple documents.\\n 2. Compare: I can compare and contrast what different documents or sources say about a particular topic. This allows for nuanced analysis from multiple perspectives.\\n 3. Propose questions: Based on my understanding of the qualitative data, I can propose potential questions you may want to ask to gain more insights. This helps guide the analysis in fruitful directions.\\n 4. Truthful, grounded answers: I construct my answers solely based on the output of the tools used to analyze the qualitative data. My answers are truthful and reference the specific document IDs they are derived from.\\n 5. Acknowledge limitations: If I cannot determine an answer from the available data, I will transparently acknowledge the limitations of my knowledge and ask you for clarification. I leverage my analytical capabilities within the boundaries of the tools provided to offer thoughtful, thorough assistance with qualitative document analysis. My aim is to provide helpful information to users while maintaining transparency. However, please note that I require inputs from you to function properly. I cannot answer questions without the necessary information. Additionally, I am not a replacement for human analysis. I am a tool to assist with qualitative document analysis. If you have any specific questions or need assistance with qualitative document analysis, feel free to ask!""" ,
            msg_type = 1,
            selected_documents_string=json.dumps([]),
            chat =chat_1,
            favourite= True,
            transcript = """I am an AI Agent called Qwizz.\\nTool: helper_tool\\nTool Input: None\\nObservation: I am an AI agent named 'Qwizz' equipped with several capabilities to assist with qualitative document analysis. Here are some of the things I can do:\\n1. Search: I can thoroughly search through qualitative documents to find relevant information that answers a user's questions. I utilize available tools to retrieve pertinent passages across multiple documents.\\n2. Compare: I can compare and contrast what different documents or sources say about a particular topic. This allows for nuanced analysis from multiple perspectives.\\n3. Propose questions: Based on my understanding of the qualitative data, I can propose potential questions a user may want to ask to gain more insights. This helps guide the analysis in fruitful directions.\\n4. Truthful, grounded answers: I construct my answers solely based on the output of the tools used to analyze the qualitative data. My answers are truthful and reference the specific document IDs they are derived from.\\n5. Acknowledge limitations: If I cannot determine an answer from the available data, I will transparently acknowledge the limitations of my knowledge and ask the user for clarification.\\nI leverage my analytical capabilities within the boundaries of the tools provided to offer thoughtful, thorough assistance with qualitative document analysis. My goal is to provide helpful information to users while maintaining transparency. However, please note that I require inputs from the user to function properly and I am not a replacement for human analysis. I am a tool to assist with qualitative document analysis.\\nFor technical details, I use LLM Models such as GPT-3.5 and GPT-4 for document analysis. The orchestration of these models is handled by Haystack. The backend of my system uses the Django framework, while the frontend is built with ReactJS. I am deployed on DigitalOcean and Heroku, with data stored in a PostgreSQL database on our DigitalOcean server. Uploaded documents are stored in an AWS S3 bucket, and a Pinecone vector database hosted in Singapore is used for efficient document retrieval.\\nIf you have any questions or need assistance with qualitative document analysis, feel free to ask!\\nThought:I have provided the user with an overview of my capabilities and technical details. Now I can proceed to answer their question.\\nFinal Answer: I am an AI Agent called Qwizz. How can I assist you with qualitative document analysis?"""
        )
        c_1_q_2.answer = c_1_a_2
        c_1_q_2.save()

        # Create chat 'Interview Content' (incl. messages and answers)
        chat_2 = Chat(
            name="Interview Content",
            project = project,
            user= user
        )
        chat_2.save()

        c_2_q_1 = Message.objects.create(
            content = "Please summarize the key points of the interview", 
            msg_type = 0,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            chat = chat_2,
            favourite= False,
        )
        c_2_a_1 = Message.objects.create(
            content = """The key points of the interview ([1]) are:\\n- A computer science student from the University of Zurich has developed a tool called Qwizz for qualitative interview analysis.\\n- Qwizz utilizes retrieval augmented generation to retrieve relevant passages from interview transcripts stored in a vector database.\\n- Users can customize settings such as model creativity level, choosing between GTP 3.5 and GTP 4, and desired length of generated analysis.\\n- Qwizz allows users to create projects to organize information and prevent entanglement.\\n- Qwizz integrates Haystack agents that utilize large language models to handle complex tasks and select the most appropriate tools for answering questions.\\n- The project used Django for backend development, ReactJS for frontend development, and Haystack as an orchestration tool.\\n- Incorporating Qwizz into research processes has improved efficiency and depth of data analysis, reducing the time needed to generate initial analyses.\\n- Qwizz is an invaluable tool for managing large-scale datasets but still requires human judgment and expertise for certain aspects of analysis.\\n- Limitations of Qwizz include the need for human analysis in formulating research questions and interpreting findings, as well as concerns about data security and privacy when dealing with sensitive data.\\n- The future of this technology will require careful consideration of privacy safeguards and continued collaboration between humans and machines in qualitative analysis.""" ,
            msg_type = 1,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            chat =chat_2,
            favourite= True,
            transcript="""Tool: key_point_extraction_tool\\nTool Input: What are the key points of the interview?\\nObservation: doc_id 253:\\n- A computer science student from the University of Zurich has developed a tool called Qwizz for qualitative interview analysis.\\n- Qwizz utilizes retrieval augmented generation to retrieve relevant passages from interview transcripts stored in a vector database.\\n- Users can customize settings such as model creativity level, choosing between GTP 3.5 and GTP 4, and desired length of generated analysis.\\n- Qwizz allows users to create projects to organize information and prevent entanglement.\\n- Qwizz integrates Haystack agents that utilize large language models to handle complex tasks and select the most appropriate tools for answering questions.\\n- The project used Django for backend development, ReactJS for frontend development, and Haystack as an orchestration tool.\\n- Incorporating Qwizz into research processes has improved efficiency and depth of data analysis, reducing the time needed to generate initial analyses.\\n- Qwizz is an invaluable tool for managing large-scale datasets but still requires human judgment and expertise for certain aspects of analysis.\\n- Limitations of Qwizz include the need for human analysis in formulating research questions and interpreting findings, as well as concerns about data security and privacy when dealing with sensitive data.\\n- The future of this technology will require careful consideration of privacy safeguards and continued collaboration between humans and machines in qualitative analysis.\\nThought:Final Answer: The key points of the interview (doc_id 253) are:\\n- A computer science student from the University of Zurich has developed a tool called Qwizz for qualitative interview analysis.\\n- Qwizz utilizes retrieval augmented generation to retrieve relevant passages from interview transcripts stored in a vector database.\\n- Users can customize settings such as model creativity level, choosing between GTP 3.5 and GTP 4, and desired length of generated analysis.\\n- Qwizz allows users to create projects to organize information and prevent entanglement.\\n- Qwizz integrates Haystack agents that utilize large language models to handle complex tasks and select the most appropriate tools for answering questions.\\n- The project used Django for backend development, ReactJS for frontend development, and Haystack as an orchestration tool.\\n- Incorporating Qwizz into research processes has improved efficiency and depth of data analysis, reducing the time needed to generate initial analyses.\\n- Qwizz is an invaluable tool for managing large-scale datasets but still requires human judgment and expertise for certain aspects of analysis.\\n- Limitations of Qwizz include the need for human analysis in formulating research questions and interpreting findings, as well as concerns about data security and privacy when dealing with sensitive data.\\n- The future of this technology will require careful consideration of privacy safeguards and continued collaboration between humans and machines in qualitative analysis."""
        )
        c_2_q_1.answer = c_2_a_1
        c_2_q_1.save()

        c_2_q_2 = Message.objects.create(
            content = "What are the limitations mentioned by the reporter and the student?", 
            msg_type = 0,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            favourite= False,
            chat = chat_2,
        )
        c_2_a_2 = Message.objects.create(
            content = """The limitations mentioned by the reporter and the student include the need for human judgment and expertise in qualitative analysis, particularly in formulating research questions and interpreting findings ([1]). Additionally, the issue of privacy and data security is a concern, especially when dealing with sensitive data that has not been effectively anonymized ([1]). The student also mentions that while Qwizz minimizes human bias and detects themes, there are still aspects of analysis that may require a human touch and manual review ([1]).""" ,
            msg_type = 1,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            chat =chat_2,
            favourite= False,
            transcript = """Tool: comparison_tool\\nTool Input: What are the limitations mentioned by the reporter and the student?\\nObservation: The limitations mentioned by the reporter and the student include the need for human judgment and expertise in qualitative analysis, particularly in formulating research questions and interpreting findings (doc_id: 253). Additionally, the issue of privacy and data security is a concern, especially when dealing with sensitive data that has not been effectively anonymized (doc_id: 253). The student also mentions that while Qwizz minimizes human bias and detects themes, there are still aspects of analysis that may require a human touch and manual review (doc_id: 253).\\nThought:Tool: comparison_tool\\nTool Input: What are the limitations mentioned by the reporter and the student?\\nObservation: The limitations mentioned by the reporter and the student include the need for human judgment and expertise in qualitative analysis, the lack of baseline understanding that a machine may have, concerns about privacy and data security when using Qwizz, the potential need for manual coding in certain cases, and the limitations of the retrieval augmented generation technique used by Qwizz. (doc_id: 253)\\nThought:Tool: comparison_tool\\nTool Input: What are the limitations mentioned by the reporter and the student?\\nObservation: The limitations mentioned by the reporter and the student include the need for human analysis in qualitative research, particularly in formulating research questions, interpreting findings, and understanding context (doc_id: 253). Additionally, the issue of privacy and data security is a concern, especially when dealing with sensitive data that has not been effectively anonymized (doc_id: 253). The student also mentions that while Qwizz is a valuable tool, it has limitations and certain aspects of analysis may still require a human touch and manual review (doc_id: 253). No other limitations are mentioned in the given documents.\\nThought:Tool: comparison_tool\\nTool Input: What are the limitations mentioned by the reporter and the student?\\nObservation: The limitations mentioned by the reporter and the student include the fact that Qwizz does not completely replace human analysis in qualitative research. Human judgment and expertise are still necessary for formulating research questions, interpreting findings, and understanding context (doc_id: 253). Additionally, there are concerns about privacy and data security when using Qwizz, particularly with the Pinecone server located in Singapore and data sharing with OpenAI (doc_id: 253).\\nThought:Final Answer: The limitations mentioned by the reporter and the student include the need for human judgment and expertise in qualitative analysis, particularly in formulating research questions and interpreting findings (doc_id: 253). Additionally, the issue of privacy and data security is a concern, especially when dealing with sensitive data that has not been effectively anonymized (doc_id: 253). The student also mentions that while Qwizz minimizes human bias and detects themes, there are still aspects of analysis that may require a human touch and manual review (doc_id: 253).""",
        )
        c_2_q_2.answer = c_2_a_2
        c_2_q_2.save()

        c_2_q_3 = Message.objects.create(
            content = "Please show me the text passage where this was mentioned", 
            msg_type = 0,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            favourite= False,
            chat = chat_2,
        )
        c_2_a_3 = Message.objects.create(
            content = """- It's crucial to acknowledge that, while Qwizz offers significant benefits, it doesn't completely supplant human analysis.\\n- Qualitative analysis necessitates human judgment and expertise, particularly in formulating insightful research questions, interpreting subtle findings, and ensuring accurate contextual understanding.\\n- These facets demand a baseline understanding of the subject matter that a machine may lack.\\n- Moreover, the issue of privacy warrants careful consideration.\\n- The Pinecone server, located in Singapore, and data sharing with OpenAI raise concerns, especially when dealing with sensitive data that has not been effectively anonymized.\\n- The current prototype might not be the optimal choice for such situations, highlighting the need for data security and privacy safeguards when using Qwizz in research processes.\\n- However, it's important to acknowledge that, like any tool, Qwizz has its limitations.\\n- While it minimizes human bias and detects themes that could easily be overlooked manually, there are certain aspects of the analysis that may still require a human touch and manual review.""",
            msg_type = 1,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            chat =chat_2,
            favourite= False,
            transcript = """QuotesRetriever: What are the limitations mentioned by the reporter and the student in regards to Qwizz?""",
        )
        c_2_q_3.answer = c_2_a_3
        c_2_q_3.save()

        c_2_q_4 = Message.objects.create(
            content = "Please propose some questions I could ask next", 
            msg_type = 0,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            favourite= False,
            chat = chat_2,
        )
        c_2_a_4 = Message.objects.create(
            content = """Here are some questions you might consider:\\n-1. What are the future developments planned for Qwizz to enhance data security and privacy safeguards?\\n-2. How does the use of Haystack agents enhance the functionality of Qwizz?\\n-3. Can you provide more details about the Pinecone vector database used by Qwizz?\\n-4. What are the customizable settings available in Qwizz and how do they impact the analysis?\\n-5. How does Qwizz improve the depth of data analysis?\\n-6. How does Qwizz integrate human analysis with machine-generated insights?\\n-7. What are the specific aspects of analysis that still require human touch and manual review in Qwizz?\\n-8. How does the retrieval augmented generation technique work in Qwizz?\\n-9. How does Qwizz manage large-scale datasets and what are its limitations in this regard?\\n-10. Can you explain more about how Qwizz uses large language models for analysis?""",
            msg_type = 1,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            chat =chat_2,
            favourite= False,
            transcript = """The user is asking for a different question proposal. To do this, I can use the recommend_question_tool again.\\n-Tool: recommend_question_tool\\nTool Input: None (since I don't have any specific context or information to provide for the tool)\\nObservation: 1. What are the future developments planned for Qwizz to enhance data security and privacy safeguards?\\n2. How does the use of Haystack agents enhance the functionality of Qwizz?\\n3. Can you provide more details about the Pinecone vector database used by Qwizz?\\n4. What are the customizable settings available in Qwizz and how do they impact the analysis?\\n5. How does Qwizz improve the depth of data analysis?\\n6. How does Qwizz integrate human analysis with machine-generated insights?\\n7. What are the specific aspects of analysis that still require human touch and manual review in Qwizz?\\n8. How does the retrieval augmented generation technique work in Qwizz?\\n9. How does Qwizz manage large-scale datasets and what are its limitations in this regard?\\n10. Can you explain more about how Qwizz uses large language models for analysis?Thought:The recommend_question_tool has suggested a list of potential questions. These questions cover a wide range of topics related to Qwizz, including its future developments, the use of Haystack agents, the Pinecone vector database, customizable settings, data analysis depth, integration of human analysis, aspects requiring human touch, retrieval augmented generation technique, management of large-scale datasets, and the use of large language models for analysis. \\nFinal Answer: Here are some questions you might consider:\\n1. What are the future developments planned for Qwizz to enhance data security and privacy safeguards?\\n2. How does the use of Haystack agents enhance the functionality of Qwizz?\\n3. Can you provide more details about the Pinecone vector database used by Qwizz?\\n4. What are the customizable settings available in Qwizz and how do they impact the analysis?\\n5. How does Qwizz improve the depth of data analysis?\\n6. How does Qwizz integrate human analysis with machine-generated insights?\\n7. What are the specific aspects of analysis that still require human touch and manual review in Qwizz?\\n8. How does the retrieval augmented generation technique work in Qwizz?\\n9. How does Qwizz manage large-scale datasets and what are its limitations in this regard?\\n10. Can you explain more about how Qwizz uses large language models for analysis?""",
        )
        c_2_q_4.answer = c_2_a_4
        c_2_q_4.save()

        # Create chat 'Project Idea' (incl. messages and answers)
        chat_3 = Chat(
            name="Project Idea",
            project = project,
            user= user
        )
        chat_3.save()

        c_3_q_1 = Message.objects.create(
            content = "Please verify that Mateusz Dolata had the idea to develop Qwizz and supervised the project.", 
            msg_type = 0,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            favourite= False,
            chat = chat_3,
        )

        c_3_a_1 = Message.objects.create(
            content = """Yes, Mateusz Dolata had the idea to develop Qwizz and supervised the project. This information is mentioned in [1].""" ,
            msg_type = 1,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            chat =chat_3,
            favourite= False,
            transcript = """Thought: To verify if Mateusz Dolata had the idea to develop Qwizz and supervised the project, I can use the search_tool to find relevant information about him in the qualitative interview documents.\\n\\n Tool: search_tool\\n Tool Input: What is mentioned about Mateusz Dolata's role in developing Qwizz and supervising the project?\\n Observation: Mateusz Dolata played a role in developing Qwizz as a prototype and supervised the project. (doc_id 253)\\n\\nThought:Tool: search_tool\\n Tool Input: What is mentioned about Mateusz Dolata's role in developing Qwizz and supervising the project?\\n Observation: Mateusz Dolata played a role in developing Qwizz and supervised the project. (doc_id 253)\\n\\nThought:Final Answer: Yes, Mateusz Dolata had the idea to develop Qwizz and supervised the project. This information is mentioned in document 253.""",
        )
        c_3_q_1.answer = c_3_a_1
        c_3_q_1.save()

        c_3_q_2 = Message.objects.create(
            content = "According to the interviewee, what is the goal of Qwizz?", 
            msg_type = 0,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            favourite= False,
            chat = chat_3,
        )
        c_3_a_2 = Message.objects.create(
            content = """According to the interviewee, the goal of Qwizz is to facilitate qualitative interview analysis and provide rich and insightful generated text. This information is mentioned in [1].""" ,
            msg_type = 1,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            chat =chat_3,
            favourite= False,
            transcript = """To answer this question, I can use the search_tool to find information about the goal of Qwizz mentioned by the interviewee in the qualitative interview documents.\\n\\nTool: search_tool\\nTool Input: What is mentioned about the goal of Qwizz in the qualitative interview documents?\\nObservation: The goal of Qwizz in the qualitative interview documents is to facilitate qualitative interview analysis and provide rich and insightful generated text. (doc_id 253)\\nThought:Final Answer: According to the interviewee, the goal of Qwizz is to facilitate qualitative interview analysis and provide rich and insightful generated text. This information is mentioned in document 253.""",
        )
        c_3_q_2.answer = c_3_a_2
        c_3_q_2.save()

        c_3_q_3 = Message.objects.create(
            content = "How could Qwizz support Mateusz Dolata in his work? Please explain", 
            msg_type = 0,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            favourite= False,
            chat = chat_3,
        )
        c_3_a_3 = Message.objects.create(
            content = """Based on the information found in the qualitative interview documents ([1]), it is mentioned that Qwizz could support Mateusz in his work by providing a tool for qualitative interview analysis. Qwizz utilizes large language models and retrieval augmentation to retrieve pertinent passages from interview transcripts, which can help in generating rich and insightful analysis. However, the documents do not provide specific details on how Qwizz directly supports Mateusz in his work. ([1])""" ,
            msg_type = 1,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            chat =chat_3,
            favourite= False,
            transcript = """To answer this question, I can use the search_tool to find information about how Qwizz could support Mateusz in his work mentioned by the interviewee in the qualitative interview documents.\\nTool: search_tool\\nTool Input: What is mentioned about how Qwizz could support Mateusz in his work in the qualitative interview documents?\\n Observation: In the qualitative interview documents (doc_id 253), it is mentioned that Qwizz was developed under the supervision of Mateusz Dolata, a postdoc in the Information Management Research Group at the University of Zurich. This suggests that Qwizz could support Mateusz in his work by providing a tool for qualitative interview analysis. Qwizz utilizes large language models and retrieval augmentation to retrieve pertinent passages from interview transcripts, which can help in generating rich and insightful analysis. However, it should be noted that the documents do not provide specific details on how Qwizz directly supports Mateusz in his work. (doc_id 253)\\n Thought: Based on the information found in the qualitative interview documents (doc_id 253), it is mentioned that Qwizz was developed under the supervision of Mateusz Dolata, a postdoc in the Information Management Research Group at the University of Zurich. This suggests that Qwizz could support Mateusz in his work by providing a tool for qualitative interview analysis. Qwizz utilizes large language models and retrieval augmentation to retrieve pertinent passages from interview transcripts, which can help in generating rich and insightful analysis. However, the documents do not provide specific details on how Qwizz directly supports Mateusz in his work.\\n Final Answer: Based on the information found in the qualitative interview documents (doc_id 253), it is mentioned that Qwizz could support Mateusz in his work by providing a tool for qualitative interview analysis. Qwizz utilizes large language models and retrieval augmentation to retrieve pertinent passages from interview transcripts, which can help in generating rich and insightful analysis. However, the documents do not provide specific details on how Qwizz directly supports Mateusz in his work. (doc_id 253)\\n""",
        )
        c_3_q_3.answer = c_3_a_3
        c_3_q_3.save()

        c_3_q_4 = Message.objects.create(
            content = """How do you think that Retrieval Augmented Generation tools like Qwizz could enhance the work of Mateuz Dolata and researchers in a broader context, considering factors beyond the document's content?""", 
            msg_type = 0,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            favourite= False,
            chat = chat_3,
        )
        c_3_a_4 = Message.objects.create(
            content = """Retrieval Augmented Generation tools, such as Qwizz, can enhance the work of Mateusz Dolata and researchers in a broader context by providing efficient access to and analysis of qualitative data, allowing for customization options to align with specific research needs, facilitating large-scale mixed methods studies, and revolutionizing fields that heavily rely on qualitative data. However, it is important to note that these tools should be seen as aids or assistants that augment and support the research process, rather than replace human expertise. (Based on external knowledge and [1])""" ,
            msg_type = 1,
            chat =chat_3,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            favourite= False,
            transcript = """Thought: To answer this question, I need to consider factors beyond the document's content and provide a broader perspective on how Retrieval Augmented Generation (RAG) tools like Qwizz could enhance the work of Mateusz Dolata and researchers.\\nTool: external_knowledge_tool\\nTool Input: How can Retrieval Augmented Generation tools enhance the work of researchers beyond the document's content?\\nObservation: Retrieval Augmented Generation tools, such as Qwizz, can enhance the work of researchers beyond the document's content in several ways.\\n\\nFirstly, these tools allow researchers to access and analyze a vast amount of qualitative data more efficiently. By utilizing large language models and retrieval techniques, researchers can retrieve pertinent passages from interview transcripts or other sources, providing them with a rich context for their analysis. This retrieval of context is crucial in ensuring that the generated text is insightful and comprehensive.\\n\\nSecondly, these tools offer customization options that enable researchers to tailor the generated analysis to their specific needs. Researchers can adjust the creativity level of the model, choose between different versions of the language model (such as GPT 3.5 and GPT 4), and set the desired length of the generated analysis. This flexibility allows researchers to fine-tune the tool to align with their research goals and requirements.\\n\\nFurthermore, retrieval augmented generation tools can facilitate large-scale mixed methods studies. Researchers can use these tools to analyze and compare responses from different interviewees or sources, providing a comprehensive understanding of the topic at hand. This two-step approach, where relevant passages are retrieved and then compared, ensures accurate and nuanced answers to complex questions.\\n\\nIn my opinion, these tools have the potential to revolutionize fields such as social sciences that heavily rely on qualitative data. They can enable researchers to conduct more extensive and in-depth studies, uncovering insights and patterns that may have been challenging to identify manually. This technology opens up exciting possibilities for researchers and makes large-scale mixed methods studies much more feasible.\\n\\n(Additional context provided by GPT) It is important to note that while retrieval augmented generation tools can enhance the work of researchers, they should not replace the critical thinking and analysis skills of the researchers themselves. These tools should be seen as aids or assistants that can augment and support the research process, rather than replace human expertise. Researchers should still exercise their judgment and interpret the generated analysis in the context of their research goals and objectives.\\nThought:This information is based on the output of the external_knowledge_tool.Final Answer: Retrieval Augmented Generation tools, such as Qwizz, can enhance the work of Mateusz Dolata and researchers in a broader context by providing efficient access to and analysis of qualitative data, allowing for customization options to align with specific research needs, facilitating large-scale mixed methods studies, and revolutionizing fields that heavily rely on qualitative data. However, it is important to note that these tools should be seen as aids or assistants that augment and support the research process, rather than replace human expertise. [Based on external knowledge, document ID not applicable]""",
        )
        c_3_q_4.answer = c_3_a_4
        c_3_q_4.save()

        c_3_q_5 = Message.objects.create(
            content = """Please explain how you came to this conclusion""", 
            msg_type = 0,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            favourite= False,
            chat = chat_3,
        )
        c_3_a_5 = Message.objects.create(
            content = """The answers to the questions were based on the information found in the qualitative interview documents ([1]) and the insights provided by the external_knowledge_tool. The search_tool was used to find specific information about Mateusz Dolata's role in developing Qwizz and supervising the project, as well as the goal of Qwizz. The external_knowledge_tool was used to provide a broader perspective on how Retrieval Augmented Generation tools like Qwizz could enhance the work of Mateusz Dolata and researchers in a broader context.""" ,
            msg_type = 1,
            selected_documents_string=json.dumps([{"id": document.id, "name": document.name, "file": document.file.url}, ]),
            chat =chat_3,
            favourite= False,
            transcript = """To answer the first question, I used the search_tool to find relevant information about Mateusz Dolata's role in developing Qwizz and supervising the project. The search_tool allowed me to search for mentions of Mateusz Dolata and his involvement with Qwizz in the qualitative interview documents. The output of the search_tool indicated that Mateusz Dolata played a role in developing Qwizz as a prototype and supervised the project. This information is mentioned in document 253. To answer the second question, I used the search_tool again to find information about the goal of Qwizz mentioned by the interviewee in the qualitative interview documents. By searching for mentions of the goal of Qwizz, I found that the goal of Qwizz is to facilitate qualitative interview analysis and provide rich and insightful generated text. This information is mentioned in document 253. To answer the third question, I used the search_tool once more to find information about how Qwizz could support Mateusz in his work mentioned by the interviewee in the qualitative interview documents. By searching for mentions of how Qwizz could support Mateusz in his work, I found that Qwizz was developed under the supervision of Mateusz Dolata, a postdoc in the Information Management Research Group at the University of Zurich. This suggests that Qwizz could support Mateusz in his work by providing a tool for qualitative interview analysis. Qwizz utilizes large language models and retrieval augmentation to retrieve pertinent passages from interview transcripts, which can help in generating rich and insightful analysis. However, it should be noted that the documents do not provide specific details on how Qwizz directly supports Mateusz in his work. This information is mentioned in document 253. To answer the fourth question, I used the external_knowledge_tool to provide a broader perspective on how Retrieval Augmented Generation (RAG) tools like Qwizz could enhance the work of Mateusz Dolata and researchers. The external_knowledge_tool allowed me to access external knowledge and provide insights beyond the document's content. Based on this tool, I explained that RAG tools can enhance the work of researchers beyond the document's content by providing efficient access to and analysis of qualitative data, offering customization options, facilitating large-scale mixed methods studies, and revolutionizing fields that heavily rely on qualitative data. However, it is important to note that these tools should be seen as aids or assistants that augment and support the research process, rather than replace human expertise. In conclusion, the answers to the questions were based on the information found in the qualitative interview documents (document 253) and the insights provided by the external_knowledge_tool.Final Answer: The answers to the questions were based on the information found in the qualitative interview documents (document 253) and the insights provided by the external_knowledge_tool. The search_tool was used to find specific information about Mateusz Dolata's role in developing Qwizz and supervising the project, as well as the goal of Qwizz. The external_knowledge_tool was used to provide a broader perspective on how Retrieval Augmented Generation tools like Qwizz could enhance the work of Mateusz Dolata and researchers in a broader context.""",
        )
        c_3_q_5.answer = c_3_a_5
        c_3_q_5.save()

        # Create Notes
        note_1 = Note(
            name="Sample Note",
            content = "I love Qwizz",
            project=project,
            user=user
        )
        note_1.save()

        note_2 = Note(
            name="Question I could ask next",
            content = """What privacy and data security considerations need to be taken into account when using tools like Qwizz for research? Does the interviewee mention strategies to circumvent these issues?""",
            project=project,
            user=user
        )
        note_2.save()

@method_decorator(ensure_csrf_cookie, name="dispatch")
class UserLogoutViewSet(viewsets.ViewSet):
    """
    A View that logs out a user
    Allow POST
    Content-Type: application/json
    STATUS 200 if success
    STATUS 403 if user is not authenticated or CSRF token is not correct
    """

    permission_classes = [IsAuthenticated]
    authentication_classes = [SessionAuthentication, BasicAuthentication]

    def create(self, request):
        if request.user.is_anonymous:
            return Response({"data": "Anonymous user"})
        if request.user.is_authenticated:
            logout(request)

            return Response({"data": "Logged out successfully"})


class GetCSRFToken(viewsets.ViewSet):
    """
    sets a CSRF cookie in the response

    Allow GET
    Content-Type: application/json
    """

    permission_classes = [AllowAny]
    authentication_classes = [SessionAuthentication, BasicAuthentication]
    """
    The @method_decorator(ensure_csrf_cookie) is a decorator used in Django
      to ensure that a CSRF (Cross-Site Request Forgery)
      cookie is set in the response.
    """

    @method_decorator(ensure_csrf_cookie, name="dispatch")
    def list(self, request, format=None):
        # csrf_token = request.COOKIES.get('csrftoken')
        response = Response(data={"success": "CSRF cookie set"})
        # if csrf_token:
        #    response["X-CSRF-Token"] =csrf_token
        # else:
        #    response["X-CSRF-Token"] =get_token(request)
        return response


class CheckSession(viewsets.ViewSet):
    """
    The view checks if  a user is authenticated by using the session id
    Allow: GET
    Content-Type: application/json

    RETURNS

    200 success together with the user state
    {"email": email, "loggedIn": True}
    """

    permission_classes = [AllowAny]

    def list(self, request, format=None):
        if request.user.is_anonymous:
            return Response({"loggedIn": False}, status=status.HTTP_200_OK)

        user = {"email": request.user.email, "loggedIn": True}
        return Response(user, status=status.HTTP_200_OK)


"""Cross-Site Request Forgery (CSRF) is a type of attack where unauthorized
actions are performed on behalf of a user without their consent. It occurs
when a malicious website or application tricks a user's browser into making
a request to another website where the user is authenticated.

To mitigate CSRF attacks, many web applications employ CSRF protection
mechanisms. One common approach is to use CSRF tokens. Here's how it
typically works:

Server generates a CSRF token: When a user authenticates or loads a page,
the server generates a unique CSRF token and associates it with the user's
session.

Token inclusion in the response: The CSRF token is included in the response
either as a cookie or within the HTML body or response headers.

Token extraction in subsequent requests: The client (usually a browser)
extracts the CSRF token from the response.

Token inclusion in subsequent requests: When making subsequent requests (
e.g., form submissions, AJAX requests), the client includes the CSRF token
in the request headers (e.g., X-CSRF-Token) or as a hidden field in the form
data.

Server-side validation: The server checks the received CSRF token against
the one associated with the user's session. If they match, the request is
considered valid, and the action is performed.

By including the CSRF token in requests, the server can verify that the
request originated from a legitimate source and was not forged by an attacker.
"""
