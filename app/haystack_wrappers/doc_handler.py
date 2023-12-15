from typing import Union

from haystack.nodes import EmbeddingRetriever, PreProcessor
from haystack.schema import Document
from haystack_wrappers.wrapper import HaystackWrapper


class HaystackDocHandler(HaystackWrapper):
    """
    Handler for documents using Haystack.

    This class provides methods to add, delete and process documents
    using a Haystack pipeline.

    Attributes:
        processor: PreProcessor object to process documents before adding
        embedding_retriever: EmbeddingRetriever to handle updating embeddings

    Methods:
        add_document: Add documents to the document store after processing
        delete_document: Delete documents from the document store
        __add_questions_to_docs: Generate and add questions to documents

    Examples:
        ```python
        handler = HaystackDocHandler()

        docs = [{'content': 'Sample text 1'},
                {'content': 'Sample text 2'}]

        handler.add_document(docs)
        ```
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo-16k", temparature: float = 0.5, answer_length: str = "medium"):
        super().__init__(model_name=model_name, temparature=temparature, answer_length=answer_length)
        self.processor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=True,
            remove_substrings=None,
            split_by="passage",
            split_length=2,
            split_overlap=1,
            split_respect_sentence_boundary=False,
        )
        self.embedding_retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model="text-embedding-ada-002",
            api_key=self.openai_api_key,
            max_seq_len=1536,
        )

    def add_document(
        self, documents: Union[dict, Document, list[Union[dict, Document]]]
    ) -> None:
        """Add documents to the document store.

        Documents are first processed using a PreProcessor.
        Embeddings are then updated in the document store.

        Args:
            documents (list): The documents to add to the document store.

        Example:
        ```python
        docs = [{'content': 'Sample text 1'},
                {'content': 'Sample text 2'}]
        wrapper.add_document(docs)
        ```
        """
        docs = self.processor.process(documents)
        self.document_store.write_documents(docs)
        self.document_store.update_embeddings(
            self.embedding_retriever, update_existing_embeddings=False
        )

    def delete_document(self, doc_ids: list[str]) -> None:
        """
        Delete a document from the document store.

        Removes the document with the specified ID from the
        document store. This will delete the document text,
        metadata, and embedding.

        Args:
            doc_ids (list[str]): List of IDs of the documents to delete

        Returns:
            None
        """
        self.document_store.delete_documents(
            filters={"doc_id": {"$in": doc_ids}}
        )
