import re
from haystack.agents import Tool
from haystack.agents.base import ToolsManager
from typing import Optional, Tuple
from haystack import Pipeline
from haystack.nodes import BaseRetriever
from haystack.pipelines import (
    BaseStandardPipeline,
)
from utils.logs import logger

class HaystackTool(Tool):
    """
    The HaystackTool is a custom implementation of Tool.

    The main difference between the HaystackTool and the Tool is that the 
    HaystackTool only forwards the the required params to the pipeline 
    instead of all params.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def run(self, tool_input: str, params: Optional[dict] = None) -> str:
        # We can only pass params to pipelines but not to nodes
        # Different pipelines require different params
        if isinstance(self.pipeline_or_node, (Pipeline, BaseStandardPipeline)):

            if self.pipeline_or_node.get_node("filter-retriever"):
                # Create a shallow copy of the original dictionary
                params_subset = params.copy()
                # Remove "retriever" from the copy
                del params_subset["retriever"]

                result = self.pipeline_or_node.run(query=tool_input, params=params_subset)

            elif self.pipeline_or_node.get_node("retriever"):  
                # Create a shallow copy of the original dictionary
                params_subset = params.copy()
                # Remove "filter-retriever" from the copy
                del params_subset["filter-retriever"]

                result = self.pipeline_or_node.run(query=tool_input, params=params_subset)

            else:
                result = self.pipeline_or_node.run(query=tool_input)

        elif isinstance(self.pipeline_or_node, BaseRetriever):
            result = self.pipeline_or_node.run(query=tool_input, root_node="Query")

        elif callable(self.pipeline_or_node):
            result = self.pipeline_or_node(tool_input)
        
        else:
            result = self.pipeline_or_node.run(query=tool_input)
        
        return self._process_result(result)

class HaystackToolsManager(ToolsManager):
    """
    The HaystackToolsManager is a custom implementation of ToolsManager.
    The HaystackToolsManager manages tools for an Agent.

    The main difference between the HaystackToolsManager and the ToolsManager
    is that the HaystackToolsManager enforces that tool names are lowercase.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_tool_name_and_tool_input(self, llm_response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the tool name and the tool input from the PromptNode response.
        :param llm_response: The PromptNode response.
        :return: A tuple containing the tool name and the tool input.
        """
        tool_match = re.search(self.tool_pattern, llm_response)
        if tool_match:
            tool_name = tool_match.group(1)
            tool_input = tool_match.group(2) or tool_match.group(3)
            
            logger.info("Tool Name: " + tool_name.strip('" []\n').strip().lower())

            # Tool name must be lowercase
            return tool_name.strip('" []\n').strip().lower(), tool_input.strip('" \n')
        
        return None, None
    