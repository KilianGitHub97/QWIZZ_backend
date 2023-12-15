from abc import ABC
import random

from django.conf import settings
from haystack.document_stores import PineconeDocumentStore
from haystack.schema import Document

from utils.logs import logger

class HaystackWrapper(ABC):
    """
    Abstract base class for Haystack document handling.

    This class initializes the Pinecone document store and OpenAI API key.
    It serves as a base class for concrete implementations of Haystack
    pipelines for working with documents.

    Attributes:
        document_store (PineconeDocumentStore): PineconeDocumentStore for storing documents
        openai_api_key (str): API key for accessing OpenAI models
        embedding_model (str): Name of embedding model to use
        text_gen_model (str): Name of text generation model to use
        answer_length (int): Length of generated text in tokens
        model_temparature (float): Sampling temperature for text generation model
    
    Methods:
        get_document_by_id(doc_id): Retrieves a document by ID
        __get_answer_length(length): Returns integer token length for a given length
        __get_openai_api_key(): Returns OpenAI API key
        __validate_model_name(name): Validates model name
        __validate_temperature(temp): Validates sampling temperature
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo-16k", temparature: float = 0.5, answer_length: str = "medium"):
        """Initializes the HaystackWrapper object.
        
        Args:
            model_name (str, optional): OpenAI model to use for text generation. 
                Valid values are "gpt-3.5-turbo-16k" and "gpt-4". Defaults to 
                "gpt-3.5-turbo-16k".
            temparature (int, optional): Sampling temparature for OpenAI model. 
                Defaults to 0.5.
            answer_length (str, optional): Length of generated text. Valid values are 
                "short", "medium", or "long". Defaults to "medium".
        """
        self.document_store = PineconeDocumentStore(
            api_key=settings.PINECONE_API_KEY,
            environment="asia-southeast1-gcp-free",
            index="qwizz",
            similarity="cosine",
            embedding_dim=1536,
        )
        self.openai_api_key = self.__get_openai_api_key()
        self.embedding_model = "text-embedding-ada-002"
        self.text_gen_model = self.__validate_model_name(model_name)
        self.answer_length = self.__get_answer_length(answer_length)
        self.model_temparature = self.__validate_temperature(temparature)

    def get_document_by_id(self, doc_id: str) -> list[Document]:
        """Retrieves a document from the document store by ID.
        
        Args:
            doc_id (str): Document ID to retrieve
            
        Returns:
            list[Document]: List containing the requested document
        """
        docs = self.document_store.get_all_documents(
            filters={"doc_id": {"$in": [doc_id]}}
        )

        # Sort by doc_id ascending
        sorted_docs = sorted(docs, key=lambda x: x.meta["_split_id"])

        return sorted_docs

    def __get_answer_length(self, answer_length: str) -> int:
        """Validates and returns the integer token length for the given answer length.
        
        Args:
            answer_length (str): Answer length ("short", "medium", or "long")
        
        Returns:
            int: Token length 
        
        Raises:
            ValueError: If invalid answer length given
        """
        valid_lengths = ["short", "medium", "long"]
        if answer_length not in valid_lengths:
            raise ValueError(
                f"Invalid answer length. Valid lengths are {valid_lengths}"
            )
        if answer_length == "short":
            return 256
        elif answer_length == "medium":
            return 512
        elif answer_length == "long":
            return 1024

    def __get_openai_api_key(self) -> str:
        """Returns the OpenAI API key.
        
        Returns:
            str: OpenAI API key
        """
        return random.choice([
            settings.OPENAI_API_KEY_1, 
            settings.OPENAI_API_KEY_2, 
            settings.OPENAI_API_KEY_3
        ])

    def __validate_model_name(self, model_name: str) -> str:
        """Validates that the model name is valid.
    
        Args:
            model_name (str): Name of OpenAI model
            
        Returns:
            str: Input model name
            
        Raises:
            ValueError: If invalid model name
        """
        # Convert to lowercase
        model_name = model_name.lower()
        # Check if valid model name
        valid_model_names = ["gpt-3.5-turbo-16k", "gpt-4"]
        if model_name not in valid_model_names:
            raise ValueError(
                f"Invalid model name. Valid model names are {valid_model_names}"
            )
        return model_name

    def __validate_temperature(self, temperature: float) -> float:
        """Validates that the temperature is within range. The temperature must 
        be between 0.0 and 2.0. The temperature determines the creativity of the 
        the large language model.
    
        Args:
            temperature (float): Sampling temperature
            
        Returns:
            float: Rounded temperature between 0.0 and 2.0 
            
        Raises:
            ValueError: If temperature is out of bounds
        """
        # Check if valid temperature
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError(
                f"Invalid temperature. Temperature must be between 0.0 and 2.0"
            )
        return round(temperature, 2)
    
