
from typing import Optional
from haystack.schema import Document, FilterType
from haystack.document_stores import BaseDocumentStore
from haystack.nodes import (
    EmbeddingRetriever,
    JoinDocuments,
)

from utils.logs import logger

class MultiDocEmbeddingRetriever(EmbeddingRetriever):
    
    def __init__(
            self,
            document_store, 
            embedding_model, 
            api_key,
            max_seq_len,
            **kwargs
            ):
        super().__init__(
            document_store=document_store,
            embedding_model=embedding_model,
            api_key=api_key,
            max_seq_len=max_seq_len,
            **kwargs
            )
        

    def retrieve(
        self,
        query: str,
        filters: Optional[FilterType] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> list[Document]:
        
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
            )
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        query_emb = self.embed_queries(queries=[query])

        doc_list: list = []
        doc_ids = filters["doc_id"]["$in"]

        for doc_id in doc_ids:
            filter = {"doc_id": {"$in": [doc_id]}}
            documents = document_store.query_by_embedding(
                query_emb=query_emb[0], filters=filter, top_k=top_k, index=index, headers=headers, scale_score=scale_score
            )
            doc_list.extend(documents)
       
        return doc_list
    