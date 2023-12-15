import requests
import time

from haystack.nodes.base import BaseComponent
from haystack.nodes.preprocessor import PreProcessor
from haystack.schema import Document

from django.conf import settings


class HuggingFaceTransformersQuestionGenerator(BaseComponent):
    """
    """

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: str = "valhalla/t5-base-e2e-qg",
        num_beams: int = 4,
        max_length: int = 256,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.5,
        early_stopping: bool = True,
        split_length: int = 50,
        split_overlap: int = 10,
        prompt: str = "generate questions:",
        num_queries_per_doc: int = 1,
        sep_token: str = "<sep>",
        batch_size: int = 16,
        progress_bar: bool = True,
    ):
        self.model_name_or_path=model_name_or_path
        self.max_length=max_length
        self.__api_key = settings.HUGGINGFACE_API_KEY
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name_or_path}"
        self.headers = {"Authorization": f"Bearer {self.__api_key}"}
        self.num_beams = num_beams
        self.max_length = max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.preprocessor = PreProcessor()
        self.prompt = prompt
        self.num_queries_per_doc = min(num_queries_per_doc, 3)
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.sep_token = sep_token



    def run(self, documents: list[Document]):  # type: ignore
        generated_questions = []
        for d in documents:
            questions = self.generate(d.content)
            curr_dict = {"document_id": d.id, "questions": questions}
            generated_questions.append(curr_dict)
            
        output = {"generated_questions": generated_questions, "documents": documents}
        return output, "output_1"

    def run_batch(self):
        """Method is not implemented"""
        pass

    def generate(self, text: str) -> list[str]:
        # Performing splitting because T5 has a max input length
        # Also currently, it seems that it only generates about 3 questions for the beginning section of text
        split_texts_docs = self.preprocessor.split(
            document={"content": text},
            split_by="word",
            split_respect_sentence_boundary=True,
            split_overlap=self.split_overlap,
            split_length=self.split_length,
        )
        split_texts = [
            f"{self.prompt} {text.content}" if self.prompt not in text.content else text.content
            for text in split_texts_docs
        ]
        
        print(split_texts)
        # Make API request for summarization
        api_responses = self._query(
            {
                "inputs": split_texts,
                'options': {
                    'wait_for_model': True
                }
            }
        )
        print(api_responses)



        ret = []
        for i, excerpt in enumerate(api_responses):
            for question in next(iter((excerpt.values()))).split(self.sep_token):
                question = question.strip()
                if question and question not in ret:
                    ret.append(question)

        return ret


    def _query(self, payload, max_retries=2, retry_interval=2):
        """
        Sends an HTTP POST request to a remote API endpoint using the provided 
        payload and retrieves the response in JSON format. Retries the request
        if the status code is 503 (Service Unavailable).

        Args:
            payload (dict): A dictionary containing the data to be sent in the 
                request body.
            max_retries (int): The maximum number of retry attempts. Defaults to 2.
            retry_interval (int): The interval between retry attempts in seconds.
                Defaults to 2 seconds.

        Returns:
            dict: A dictionary representing the JSON response from the API.

        Raises:
            requests.exceptions.RequestException: If an error occurs while making 
                the HTTP request.
        """
        for attempt in range(max_retries + 1):
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503 and attempt < max_retries:
                # Service Unavailable, wait and retry
                time.sleep(retry_interval)
                continue
            else:
                response.raise_for_status()  # Raise an exception for other errors

        raise requests.exceptions.RequestException(
            f"Failed to get a successful response after {max_retries} attempts."
            )
    