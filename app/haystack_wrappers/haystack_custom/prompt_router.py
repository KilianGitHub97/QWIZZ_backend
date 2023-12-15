from typing import Any, List, Tuple, Dict, Optional, Union
from collections import defaultdict
import itertools

from haystack.nodes import (
    AnswerParser,
    DocumentMerger,
    FilterRetriever,
    PromptNode,
    PromptTemplate,
)
from haystack.nodes.base import BaseComponent
from haystack.schema import Document
from haystack.nodes.prompt.invocation_layer.chatgpt import ChatGPTInvocationLayer
from haystack.nodes.prompt.prompt_model import PromptModel
from utils.logs import logger


class PromptRouter(BaseComponent):
    """
    A node to split a list of `Document`s by `content_type` or by the values of a metadata field and route them to
    different nodes.
    """

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: Union[str, PromptModel] = "gpt-3.5-turbo-16k",
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
        max_length: Optional[int] = 450,
        api_key: Optional[str] = None,
        split_by: str = "doc_id",
        return_remaining: bool = False,
        recursive_prompt: Optional[Union[str, PromptTemplate]] = None
    ):
        """
        :param split_by: Field to split the documents by, either `"content_type"` or a metadata field name.
            If this parameter is set to `"content_type"`, the list of `Document`s will be split into a list containing
            only `Document`s of type `"text"` (will be routed to `"output_1"`) and a list containing only `Document`s of
            type `"table"` (will be routed to `"output_2"`).
            If this parameter is set to a metadata field name, you need to specify the parameter `metadata_values` as
            well.
         :param metadata_values: A list of values to group `Document`s by metadata field. If the parameter `split_by`
            is set to a metadata field name, you must provide a list of values (or a list of lists of values) to
            group the `Document`s by.
            If `metadata_values` is a list of strings, then the `Document`s whose metadata field is equal to the
            corresponding value will be routed to the output with the same index.
            If `metadata_values` is a list of lists, then the `Document`s whose metadata field is equal to the first
            value of the provided sublist will be routed to `"output_1"`, the `Document`s whose metadata field is equal
            to the second value of the provided sublist will be routed to `"output_2"`, and so on.
        :param return_remaining: Whether to return all remaining documents that don't match the `split_by` or
            `metadata_values` into an additional output route. This additional output route will be indexed to plus one
             of the previous last output route. For example, if there would normally be `"output_1"` and `"output_2"`
             when return_remaining  is False, then when return_remaining is True the additional output route would be
             `"output_3"`.
        """
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.api_key = api_key
        self.split_by = split_by
        self.return_remaining = return_remaining

        if recursive_prompt is None:
            self.recursive_prompt = prompt_template
        else:
            self.recursive_prompt = recursive_prompt

    def _get_metadata_values_index(self, metadata_values: Union[List[str], List[List[str]]], value: str) -> int:
        for idx, item in enumerate(metadata_values):
            if isinstance(item, list):
                if value in item:
                    return idx
            else:
                if value == item:
                    return idx
        return len(metadata_values)

    def _split_by_metadata_values(
        self, metadata_values: Union[List, List[List]], documents: List[Document]
    ) -> Dict[str, List[Document]]:
        # We need also to keep track of the excluded documents so we add 2 to the number of metadata_values
        output_keys = [f"output_{i}" for i in range(1, len(metadata_values) + 2)]
        split_documents: dict[str, list[Document]] = {k: [] for k in output_keys}
        # This is the key used for excluded documents
        remaining_key = output_keys[-1]

        for doc in documents:
            current_metadata_value = doc.meta.get(self.split_by, remaining_key)
            index = self._get_metadata_values_index(metadata_values, current_metadata_value)
            output = output_keys[index]
            split_documents[output].append(doc)

        if not self.return_remaining:
            if len(split_documents[remaining_key]) > 0:
                logger.warning(
                    "%s documents were skipped because they were either missing the metadata field '%s' or the"
                    " corresponding metadata value is not included in `metadata_values`.",
                    len(split_documents[remaining_key]),
                    self.split_by,
                )
            del split_documents[remaining_key]

        return split_documents

    def run(self, documents: list[Document]) -> Tuple[dict, str]:  # type: ignore
        
        self.metadata_values = {doc.meta[self.split_by] for doc in documents}

        if self.metadata_values:
            split_documents = self._split_by_metadata_values(self.metadata_values, documents)
        else:
            raise ValueError(
                "If split_by is set to the name of a metadata field, provide metadata_values if you want to split "
                "a list of Documents by a metadata field."
            )
        
        # Initialize prompt node
        self.prompt_node = PromptNode(
            model_name_or_path=self.model_name_or_path,
            api_key=self.api_key,
            max_length=self.max_length,
        )

        result_documents: list[Document] = []
        for key in split_documents:
            passages = split_documents[key]
            document_name = passages[0].meta.get("name", "unknown")
            document_id = passages[0].meta.get("doc_id", "unknown")

            sorted_passages = sorted(passages, key=lambda x: x.meta["_split_id"])

            result = self.__recursive_main_task(
                docs=sorted_passages, 
                prompt_template=self.prompt_template
                )

            result_document = Document(content=result, meta={"doc_id": document_id, "name": document_name})
            result_documents.append(result_document)
        
        output = {"documents": result_documents}
        return output, "output_1"

    def run_batch(self, documents: list[Document]) -> Tuple[dict, str]:
        return self.run(documents)  # type: ignore
    

    def __recursive_main_task(
        self, docs: list[Document], prompt_template: Optional[Union[str, PromptTemplate]], window_size=12
    ) -> Document:
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

        self.prompt_node.default_prompt_template = prompt_template

        # Merge documents in windows
        merged_docs = []
        max_range = len(docs) - window_size + 1
        
        if window_size >= len(docs):
            merged_doc = document_merger.merge(docs)
            merged_docs.extend(merged_doc)
        else:
            for i in range(0, max_range, window_size):
                window_start = i
                # Ensure the window_end is within bounds
                window_end = min(i + window_size, len(docs))
                window_docs = docs[window_start:window_end]

                merged_doc = document_merger.merge(window_docs)

                merged_docs.extend(merged_doc)

        # Generate summaries
        response = self.prompt_node.run_batch(documents=merged_docs)
        results = list(itertools.chain(*response[0]['results']))

        # Base case - single doc
        if len(results) == 1:
            return results[0]

        # Convert summaries to HaystackDocuments
        new_docs=[Document(content=result) for result in results]
        # Recursive call on summarised docs
        return self.__recursive_main_task(new_docs, prompt_template=self.recursive_prompt)
        