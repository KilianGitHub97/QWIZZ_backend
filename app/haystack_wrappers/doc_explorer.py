import requests
import itertools
from api.models import Document, DocumentQnA
from django.core.files import File
from django.core.files.temp import NamedTemporaryFile
from haystack import Pipeline
from haystack.nodes import (
    AnswerParser,
    DocumentMerger,
    FilterRetriever,
    PromptNode,
    PromptTemplate,
)
from haystack.pipelines import QuestionGenerationPipeline
from haystack.schema import Document as HaystackDocument
from haystack_wrappers.wrapper import HaystackWrapper
from haystack_wrappers.haystack_custom import (
    HuggingFaceTransformersQuestionGenerator
)
from haystack_wrappers.utils import remove_stop_words
from haystack.nodes.prompt.invocation_layer.chatgpt import ChatGPTInvocationLayer
from haystack.nodes.prompt.prompt_model import PromptModel

from utils.logs import logger


class HaystackDocExplorer(HaystackWrapper):
    """
    A class for exploring documents using Haystack.

    Provides methods for document summarization, question generation,
    and visualization like word clouds.     

    Attributes:
        document_store (DocumentStore): Haystack DocumentStore

    Methods:
        create_exploration_page: Creates an exploration page for a document
            with summary, word cloud etc.
        generate_questions: Generates questions for input documents
        create_summary: Summarizes a document
        create_word_cloud: Creates a word cloud image for a document

    Example:
        ```python
        explorer = HaystackDocExplorer()
        doc_id = 123
        text = "Sample text..."
        explorer.create_exploration_page(doc_id=doc_id, text=text)
        ```
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo-16k", temparature: float = 0.5, answer_length: str = "medium", use_hugging_face: bool = True):
        """Initializes the HaystackWrapper object.
        
        Args:
            model_name (str, optional): OpenAI model to use for text generation. 
                Valid values are "gpt-3.5-turbo-16k" and "gpt-4". Defaults to 
                "gpt-3.5-turbo-16k".
            temparature (int, optional): Sampling temparature for OpenAI model. 
                Defaults to 0.5.
            answer_length (str, optional): Length of generated text. Valid values are 
                "short", "medium", or "long". Defaults to "medium".
            use_hugging_face (bool, optional): Whether to use the HuggingFace
                QuestionGenerator. Defaults to True. If false, PromptNodes
                are used for question generation.
        """
        super().__init__(model_name=model_name, temparature=temparature, answer_length=answer_length)
        self.use_hugging_face = use_hugging_face
        self.filter_retriever = FilterRetriever(
            document_store=self.document_store,
        )

    def create_doc_exploration_page(self, doc_id: int, text: str) -> None:
        """Creates an exploration page for a document. The exploration page
        includes a word cloud and a summary.

        Generates a summary and word cloud for the document and saves them.

        Args:
            doc_id (int): ID of the document to create exploration page for.
            text (str): Full text content of the document.

        Returns:
            None

        Example:
            ```python
            doc_id = 123
            text = "Sample document text..."
            explorer = HaystackDocExplorer()

            explorer.create_doc_exploration_page(doc_id, text)
            ```
        """
        # Get document object
        document = Document.objects.get(id=doc_id)
        try:
            # Create summary
            summary = self.create_summary(str(doc_id))
            # Create word cloud
            word_cloud = self.create_word_cloud(text)

            # Save output to document object
            document.summary = summary
            document.word_cloud.save("wordcloud.png", word_cloud)

            # Update status to completed
            document.summary_status = "Completed"
            document.word_cloud_status = "Completed"
        except Exception as e:
            logger.info(e)
            # Update status to error if it fails
            document.summary_status = "Error"
            document.word_cloud_status = "Error"
        finally:
            # Save document changes
            document.save()

    def create_question_exploration_page(self, doc_id: int) -> None:
        """Creates an exploration page containing questions and answers a
        researcher could ask about the document.

        Generates questions and answers to these questions for a given
        document.

        Args:
            doc_id (int): ID of the document to create exploration page for.

        Returns:
            None

        Example:
            ```python
            doc_id = 123
            explorer = HaystackDocExplorer()

            explorer.create_question_exploration_page(doc_id)
            ```
        """
        # Get document object
        document = Document.objects.get(id=doc_id)
        try:
            # Generate questions for the document
            self.generate_questions_and_answers(doc_id=str(doc_id))

            # Update status to completed
            document.qna_status = "Completed"
        except Exception as e:
            logger.info(e)
            # Update status to error if it fails
            document.qna_status = "Error"
        finally:
            # Save document changes
            document.save()

    def create_summary(self, doc_id: str) -> str:
        """Generates a summary for the document with the given ID.

        Uses the HuggingFaceTransformersSummarizer to recursively summarize 
        the document.

        Args:
            doc_id (str): ID of the document to summarize

        Returns:
            str: Summary string for the document
        """
        # Get documents to summarize
        docs = self.get_document_by_id(doc_id=doc_id)

        prompt_model = PromptModel(
            api_key=self.openai_api_key,
            # could be anything
            model_name_or_path=self.text_gen_model,
            max_length = 500,
            invocation_layer_class=ChatGPTInvocationLayer,
        )

        prompt_template = PromptTemplate(
            prompt="""
            You are a highly skilled AI trained in language comprehension and 
            summarization. Read the following text and summarize 
            it into a concise abstract paragraph. Aim to retain the most 
            important points, providing a coherent and readable summary that 
            could help a person understand the main points of the interview 
            without needing to read the entire text. Note: It's very important to 
            stick to the facts and not hallucinate. Also, avoid unnecessary details
            or tangential points and do not make any stuff up.
            Documents: {join(documents)};   
            Summary:
            """,
        )

        # Initialize summarizer model
        self.summarizer = PromptNode(
            model_name_or_path=prompt_model,
            default_prompt_template=prompt_template,
        )

        # Recursively summarize document
        result = self._recursive_summarizer(docs)

        return result

    def create_word_cloud(self, text: str) -> File:
        """Generates a word cloud image for the given text.

        Uses QuickChart API to create the word cloud.

        Args:
            text (str): Input text to generate word cloud for.

        Returns:
            File: Word cloud image file.

        """
        # Remove all stop words from text
        cleaned_text = remove_stop_words(text=text)
        # Create temp file to store image
        img = NamedTemporaryFile()

        # Call QuickChart API to generate word cloud
        resp = requests.post(
            'https://quickchart.io/wordcloud', 
            json={
                "format": "png",
                "width": 1000,
                "height": 1000,
                "fontScale": 15,
                "scale": "linear",
                "removeStopwords": True,
                "minWordLength": 4,
                "text": cleaned_text,
            },
        )

        # Write image data to temp file and flush
        img.write(resp.content)
        img.flush()

        # Create django File object and return it
        file = File(img)
        return file

    def generate_questions_and_answers(
        self,
        doc_id: str,
    ) -> None:
        """Generates questions for the input document and stores them.

        This method takes a document ID, retrieves the corresponding
        document, generates questions based on the document content using a
        QuestionGenerator and QuestionGenerationPipeline, prompts an AI model
        to generate answers for each question, and stores the questions and
        answers in the database.

        Args:
            doc_id (str): ID of the document to generate questions for.

        Returns:
            None: Generated questions and answers are stored to the database.
        """
        # Get documents to generate questions for
        doc_passages = self.get_document_by_id(doc_id=doc_id)
        doc_model = Document.objects.get(id=doc_id)

        # Initialize the question generator and question generation pipeline
        self._initialize_question_generation_pipeline()

        for doc in doc_passages:
            # Generate questions for each document passage
            queries = self._generate_questions(doc)
            
            for query in queries:
                # For each generated question, prompt the model to generate an
                # answer
                if query != "":
                    answer = self._generate_answers(query, doc) 
                    # Save the question and answer pair to the database
                    self._save_qa_to_db(doc_model, doc, query, answer)

    def _generate_answers(
        self, query: str, document: HaystackDocument
    ) -> str:
        """Generates an answer for a question based on a document.

        Uses the model and prompt template to generate an answer for the
        given question string based on the provided HaystackDocument.

        Args:
            query (str): The question string to generate an answer for.
            document (HaystackDocument): The document to use for answer
                generation.

        Returns:
            str: The generated answer string.
        """
        # Create a prompt template to send questions and documents to the
        # LLM model
        qa_prompt_template = PromptTemplate(
            prompt="""
            Answer the question concisely and truthfully based solely on 
            the given documents. Note that the documents always contain the 
            answer to the question. In the unlikely case you cannot answer 
            the question, say that the question cannot be answer based on
            the given documents.
            Keep your answer short.
            Documents:{join(documents)} 
            Question:{query} 
            Answer:
            """,
            output_parser=AnswerParser(),
        )

        # Create the prompt node for generating answers
        prompt_node = PromptNode(
            model_name_or_path=self.text_gen_model,
            api_key=self.openai_api_key,
            default_prompt_template=qa_prompt_template,
        )

        prompt_answer = prompt_node(
            query=query,
            documents=[document],
        )

        answer = prompt_answer[0].answer

        return answer

    def _generate_questions(self, document: HaystackDocument) -> list[str]:
        """Generates questions based on a HaystackDocument using
        QuestionGenerationPipeline.

        Args:
            document (HaystackDocument): The document to generate questions
                for.

        Returns:
            list[str]: The generated list of question strings.
        """
        result = self._question_generation_pipeline.run(documents=[document])
        
        # HuggingFaceTransformersQuestionGenerator has different output than PromptNode
        if self.use_hugging_face:
            queries = result["generated_questions"][0]["questions"]
        else:
            queries = result["answers"][0].answer.split("\n")
            # Remove numbering
            queries = [item.split('. ', 1)[-1] for item in queries]

        return queries

    def _initialize_question_generation_pipeline(self, ) -> None:
        """Initializes a QuestionGenerator and QuestionGenerationPipeline
        to generate questions for the provided document.
        """
        if self.use_hugging_face:
            # Initialize the question generator and question generation pipeline
            question_generator = HuggingFaceTransformersQuestionGenerator(
                split_length=250, split_overlap=20
            )

            self._question_generation_pipeline = QuestionGenerationPipeline(
                question_generator
            )
        else:
            # Initialize the question generator pipeline using PromptNodes
            qa_prompt_template = PromptTemplate(
                prompt="""
                Given the documents below please generate a list of (max. 3) open-ended 
                questions. Try to ask questions that can be answered given the content 
                of document and that explore different aspects of the topic. Make sure 
                each question asks about a single topic or idea.
                Separate each question with a new line.

                Documents:{join(documents)} 
                Question:
                """,
                output_parser=AnswerParser(),
            )

            # Create the prompt node for generating answers
            prompt_node = PromptNode(
                model_name_or_path=self.text_gen_model,
                api_key=self.openai_api_key,
                default_prompt_template=qa_prompt_template,
            )

            self._question_generation_pipeline = Pipeline()
            self._question_generation_pipeline.add_node(
                component=prompt_node,
                name="prompt_node",
                inputs=["Query"],
            )


    def _recursive_summarizer(
        self, docs: list[HaystackDocument], window_size=12
    ) -> str:
        """Recursively summarizes documents into a single summary string.

        Uses windowing to merge document summaries at each recursion level.

        Args:
            docs (list): List of documents to summarize.
            window_size (int): Number of document summaries to merge per
                recursion.

        Returns:
            str: Final merged summary string.
        """
        # Initialize document merger
        document_merger = DocumentMerger(separator=" ")

        # Merge documents in windows
        merged_docs = []
        step_size = window_size // 2
        step_size_adj = step_size if step_size < len(docs) else 1
        max_range = len(docs) - window_size + 1

        if window_size >= len(docs):
            merged_doc = document_merger.merge(docs)
            merged_docs.extend(merged_doc)
        else:
            for i in range(0, max_range, step_size_adj):
                window_start = i
                # Ensure the window_end is within bounds
                window_end = min(i + window_size, len(docs))
                window_docs = docs[window_start:window_end]

                merged_doc = document_merger.merge(window_docs)

                merged_docs.extend(merged_doc)

        # Generate summaries
        response = self.summarizer.run_batch(documents=merged_docs)
        summaries = list(itertools.chain(*response[0]['results']))

        # Base case - single doc
        if len(summaries) == 1:
            return summaries[0]

        # Convert summaries to HaystackDocuments
        new_docs=[HaystackDocument(content=summary) for summary in summaries]

        # Recursive call on summarised docs
        return self._recursive_summarizer(new_docs)

    def _save_qa_to_db(
        self,
        model: Document,
        document: HaystackDocument,
        query: str,
        answer: str,
    ) -> None:
        """Saves a question-answer pair to the database.

        Args:
            model (Document): The Document database model instance.
            document (HaystackDocument): The original Haystack document.
            query (str): The generated question string.
            answer (str): The model generated answer string.

        Returns:
            None: The qna object is saved to the database.

        """
        split_id = document.meta["_split_id"]

        qna = DocumentQnA(
            document=model, split_id=split_id, question=query, answer=answer
        )

        qna.save()
