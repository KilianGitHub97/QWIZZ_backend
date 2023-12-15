from haystack import Pipeline
from haystack.nodes import (
    AnswerParser,
    EmbeddingRetriever,
    PromptNode,
    PromptTemplate,
)
from retry import retry
from haystack_wrappers.wrapper import HaystackWrapper
from haystack_wrappers.utils import recreate_conversation

from utils.logs import logger

class HaystackQuotesRetriever(HaystackWrapper):
    """
    A class for retrieving passages from filtered documents
    using Haystack.

    This provides methods to filter documents before retrieving
    passages. It sets up a pipeline using components like
    FilterRetriever.

    Attributes:
        document_store (DocumentStore): Haystack DocumentStore

    Methods:
        run: Runs the pipeline to retrieve passages from filtered
            documents
        __determine_question: Reformulates the question to improve
            retrieval

    Example:
        ```python
        filter_retriever = HaystackFilterRetriever()
        filter_retriever.run(query, doc_ids, chat_history)
        ```
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo-16k", temparature: float = 0.5, answer_length: str = "medium", chat_history: dict = {"input": [], "output": []}):
        super().__init__(model_name=model_name, temparature=temparature, answer_length=answer_length)
        self._chat_history = chat_history
        self.embedding_retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=self.embedding_model,
            api_key=self.openai_api_key,
            max_seq_len=1536,
        )


    @retry(tries=2, delay=1)
    def run(
        self, query: str, doc_ids: list[str],
    ) -> tuple[str, str]:
        """
        Run the pipeline to retrieve a relevant passage from the
        filtered documents.
        
        This method is decorated with a @retry decorator that will retry calling 
        the method up to 2 times if it fails, with a delay of 1 second between retries.
        This allows the method to retry in case of temporary errors.

        Args:
            query (str): The question to be answered.
            doc_ids (list[str]): List of document IDs to filter on.

        Raises:
            RuntimeError: If unable to generate an answer after the 
                maximum number of retries.
            
        Returns:
            str: The retrieved passage text.
            str: The query used to extract the text passage.
        """        
        new_query = self._determine_question()
        logger.info(f"NEW QUERY: {new_query}")

        # Configure prompt template to extract relevant text passages
        prompt_template_doc_selector = PromptTemplate(
            prompt="""
            Extract all existing sentences from the following documents that answer the
            question: {query}
            Include any sources or references if applicable.
            Documents: {' - '.join(['doc_id '+d.meta['doc_id']+': '+d.content + {newline} for d in documents])};  
            Answer:
            """,
            output_parser=AnswerParser(),
        )

        # Create PromptNode
        prompt_node_doc_selector = PromptNode(
            model_name_or_path=self.text_gen_model,
            api_key=self.openai_api_key,
            default_prompt_template=prompt_template_doc_selector,
            max_length=self.answer_length*2,
            model_kwargs={"temperature": self.model_temparature},
        )

        # Construct pipeline with Retriever
        retrieving_pipeline = Pipeline()
        retrieving_pipeline.add_node(
            component=self.embedding_retriever,
            name="retriever",
            inputs=["Query"],
        )
        retrieving_pipeline.add_node(
            component=prompt_node_doc_selector,
            name="prompt_node_doc_selector",
            inputs=["retriever"],
        )

        result = retrieving_pipeline.run(
            query=new_query,
            params={
                "top_k": 2,
                "filters": {"doc_id": {"$in": doc_ids}},
            },
        )

        text_passage = result["answers"][0].answer

        # There is no transcript
        transcript = f"QuotesRetriever: {new_query}"

        return text_passage, transcript

    def _determine_question(self):
        """
        Reformulate the question to improve passage retrieval.

        Args:
            query (str): The original question.

        Returns:
            str: The reformulated question.
        """
        # Reconstruct relevant part of conversation
        conversation = recreate_conversation(
            self._chat_history, 
            reversed_order=True, 
            current_message=False
        )
        # Configure prompt template to reconstruct the question that was to to retrieve
        # the interview passages
        prompt_template = PromptTemplate(
            prompt=f"""
            Given the conversation below, determine the tool input that was forwarded
            to the retriever to generate the LLM output.
            Reformulate your answer as a question that can be used to retrieve the
            relevant text passages.
            Conversation:  {conversation};
            Final Answer:
            """,
            output_parser=AnswerParser(),
        )

        # Create PromptNode
        prompt_node = PromptNode(
            model_name_or_path=self.text_gen_model,
            api_key=self.openai_api_key,
            default_prompt_template=prompt_template,
            max_length=self.answer_length,
            model_kwargs={"temperature": self.model_temparature},
        )

        identified_question = prompt_node.run()
        new_query = identified_question[0]["answers"][0].answer

        return new_query
