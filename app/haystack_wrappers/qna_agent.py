import itertools
from retry import retry

from haystack import Pipeline
from haystack.agents import Agent
from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import (
    AnswerParser,
    EmbeddingRetriever,
    PromptNode,
    PromptTemplate,
    FilterRetriever,
    Shaper
)
from haystack.nodes.ranker import DiversityRanker, LostInTheMiddleRanker
from haystack.schema import Document
from haystack_wrappers.haystack_custom import HaystackTool as Tool
from haystack_wrappers.haystack_custom import HaystackToolsManager as ToolsManager
from haystack_wrappers.haystack_custom import MultiDocEmbeddingRetriever, PromptRouter

from haystack_wrappers.utils import register_custom_shaper_functions, resolver_function, recreate_conversation, prepare_data_for_memory
from haystack_wrappers.wrapper import HaystackWrapper

from utils.logs import logger


class HaystackQnAAgent(HaystackWrapper):
    """
    A class for QA over documents using Haystack and conversational agents.

    This provides methods to create a conversational agent for querying
    documents via natural language. It sets up a pipeline using components
    like EmbeddingRetriever and ConversationalAgent.

    Attributes:
        document_store (DocumentStore): Haystack DocumentStore
        embedding_model (str): Name of embedding model
        openai_api_key (str): API key for accessing OpenAI models

    Methods:
        run: Runs the QA pipeline to answer questions
        __setup_conversational_agent: Creates a ConversationalAgent
        __create_search_tool: Creates a SearchTool for QA over documents


    Example:
    ```python
    qna_agent = HaystackQnAAgent()
    qna_agent.run(query, doc_ids, chat_history)
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
        """Run the QA pipeline for a given query and document ID.

        This method is decorated with a @retry decorator that will retry calling 
        the method up to 2 times if it fails, with a delay of 1 second between retries.
        This allows the method to retry in case of temporary errors.
                
        Args:
            query (str): The question to be answered.
            doc_ids (list[str]): List of document IDs to search over.

        Raises:
            RuntimeError: If unable to generate an answer after the 
                maximum number of retries.  

        Returns:
            str: The generated answer string.
            str: The generated transcript. This includes the thoughts
                of the agent and the final answer.

        Example:
            ```python
            query = "What is the capital of France?"
            doc_ids = ["123", "456"]
            chat_history = {}

            qna_agent = HaystackQnAAgent()
            answer, transcript = qna_agent.run(query, doc_ids, chat_history)
            print(answer)
            # Paris
            ```
        """
        # Create memory and search tools
        memory_tool, summary_memory = self.__create_memory_tool()
        search_tool = self.__create_search_tool()
        comparison_tool = self.__create_comparison_tool()
        key_take_aways_tool = self.__create_key_take_aways_tool()
        external_knowledge_tool = self.__create_external_knowledge_tool()

        # Initialize conversational agent
        conversational_agent = self.__setup_conversational_agent(
            summary_memory,
            tools=[search_tool, comparison_tool, key_take_aways_tool, external_knowledge_tool],  # memory_tool
        )
        
        # Run agent and generate answer
        result = conversational_agent.run(
            query=query,
            params={ 
                "retriever": {
                    "top_k": 5,
                    "filters": {"doc_id": {"$in": doc_ids}}
                },
                "filter-retriever": {
                    "filters": {"doc_id": {"$in": doc_ids}}
                }        
            },
            max_steps=8,
        )

        answer = result["answers"][0].answer
        transcript = result["transcript"]

        if answer == "":
            answer = """
            Oops! Looks like something didn't quite work out as expected. 
            Could you please give it another shot and perhaps try rephrasing 
            your question? This should help me assist you better. 
            """

        return answer, transcript

    def __setup_conversational_agent(
        self, summary_memory: ConversationSummaryMemory, tools: list
    ) -> Agent:
        """Configure a ConversationalAgent using GPT-3.5 and tools.

        Args:
            summary_memory (ConversationSummaryMemory): The conversation memory
                object.
            tools (list): List of Tool objects to make available.

        Returns:
            Agent: Agent instance with:
            - GPT-3.5 PromptNode
            - Custom prompt template defining conversation flow and tool use
            - Access to provided tools
            - Temperature of 0.3 for deterministic responses
        """
        # Create prompt node with GPT-3.5 model
        agent_prompt_node = PromptNode(
            self.text_gen_model,
            api_key=self.openai_api_key,
            max_length=self.answer_length*3,
            stop_words=["Observation:"],
            model_kwargs={"temperature": self.model_temparature},
        )

        # Define multi-step prompt for conversational agent
        agent_prompt = """In the following conversation, a human user
        interacts with 'Qwizz' (an AI Agent) to analyse qualitative interview 
        documents. The human user poses questions, and the AI Agent goes through several
        steps to analyse the provide well-informed answers. The AI Agent must
        use the available tools to find the required information in the
        qualitative interview documents. The final answer to the question
        should be truthfully based solely on the output of the
        tools. The final answer should reference the document IDs (doc_id)
        that it was generated from. If the AI Agent cannot answer the question, 
        say I don't know or ask for the missing piece of information. The AI 
        Agent should try to answer the questions using these tools:
        {tool_names_with_descriptions}

        The following is the previous conversation between a human and The AI
        Agent:
        {memory};

        AI Agent responses must start with one of the following:

        Thought: [the AI Agent's reasoning process. The thoughts are NOT 
        forwarded to human user] Tool: [tool names] (on a
        new line) Tool Input: [input as a question for the selected tool
        WITHOUT quotation marks and on a new line] (These must always be
        provided together and on separate lines.) Observation: [tool's
        result] Final Answer: [final answer to the human user's question. 
        The AI agent must answer the question explicitely. AI Agent is NOT
        allowed to say that the answer is mentioned above or in...!.
        INSTEAD, the AI Agent must provide the answer explicitly. EVEN IF HE REPEATS HIMSELF!!!
        KEEP THE SAME STRUCTURE AS IN THE THOUGHTS. BULLET POINTS REMAIN BULLET POINTS.]
        When selecting a tool, the AI Agent must provide both the "Tool:"
        and "Tool Input:" pair in the same response, but on separate lines

        The AI Agent can ask the human user for additional information,
        clarification, or context if required. If the AI Agent cannot find
        a specific answer after exhausting available tools and approaches,
        it answers with Final Answer: I could not find the answer to your 
        question in the selected documents. Please consider providing
        additional documents or rephrasing your question. (If possible
        tell the human why you cannot answer the question and what you know.)

        Question: {query}
        Thought:
        {transcript}
        """

        # Create agent instance
        conversational_agent = Agent(
            agent_prompt_node,
            prompt_template=agent_prompt,
            prompt_parameters_resolver=resolver_function,
            memory=summary_memory,
            final_answer_pattern=r"Final Answer:([^\n]*(\n[^\n]*)*)",
            tools_manager=ToolsManager(tools),
        )

        return conversational_agent

    def __create_memory_tool(self) -> tuple[Tool, ConversationSummaryMemory]:
        """Create Memory Tool and initialize with provided memory state.

        Args:
            memory (dict): Existing conversation memory state

        Returns:
            ConversationSummaryMemory: The conversation memory object.
            Tool: Configured Memory Tool initialized with given state
        """
        # Create prompt node to power memory
        memory_prompt_node = PromptNode(
            self.text_gen_model, 
            api_key=self.openai_api_key, 
            max_length=2000,
            model_kwargs={"temperature": self.model_temparature},
        )

        # Build conversation summary memory component
        summary_memory = ConversationSummaryMemory(
            memory_prompt_node,
            prompt_template="{chat_transcript}",
        )

        # If existing memory provided, load it
        prepared_chat_hist = prepare_data_for_memory(self._chat_history)
        summary_memory.save(prepared_chat_hist)

        # Create Tool to encapsulate memory component
        memory_tool = Tool(
            name="memory_tool",
            pipeline_or_node=summary_memory,
            description="Your memory. Always access this tool first to "
            "remember what you have learned.;",
        )

        return memory_tool, summary_memory

    def __create_search_tool(self) -> Tool:
        """Create a SearchTool for interview question answering.

        This constructs a Pipeline with a Retriever and PromptNode
        using a prompt template specialized for answering questions
        based on interview documents.

        Returns:
            Tool: The configured Q&A SearchTool object.
        """
        # Configure prompt template for answering questions
        # based on interview documents
        prompt_template = PromptTemplate(
            prompt="""
            You will be provided with excerpts from various interviews, each paragraph is 
            accompanied by a document ID (doc_id) indicating its source. The provided 
            excerpts contain the questions of the interviewer as well as the answers of the
            interviewee. If multiple excerpts share the same document ID (doc_id), it 
            signifies the same interviewee. 
            Answer the question truthfully based solely on the given documents. Make sure to 
            reference all relevant document IDs (doc_id) for each answer. If the documents 
            do not contain the answer to the question, say that answering is not possible 
            given the available information.
            Documentation Content: {' - '.join(['doc_id '+d.meta['doc_id']+': '+d.content for d in documents])};  
            Question:{query} 
            Answer:
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

        # Create diversity ranker to optimze pipeline performance
        diversity_ranker = DiversityRanker()

        # Create "lost in the middle" ranker to optimze pipeline performance
        litm_ranker = LostInTheMiddleRanker(word_count_threshold=1024)

        # Construct pipeline with retriever, rankers and prompt node
        search_pipeline = Pipeline()
        search_pipeline.add_node(
            component=self.embedding_retriever,
            name="retriever",
            inputs=["Query"],
        )
        search_pipeline.add_node(
            component=diversity_ranker,
            name="diversity_ranker",
            inputs=["retriever"],
        )
        search_pipeline.add_node(
            component=litm_ranker,
            name="litm_ranker",
            inputs=["diversity_ranker"],
        )
        search_pipeline.add_node(
            component=prompt_node, name="prompt_node", inputs=["litm_ranker"]
        )

        # Create Tool encapsulating pipeline
        search_tool = Tool(
            name="search_tool",
            pipeline_or_node=search_pipeline,
            description="""Useful for when you need to answer questions about
            the interviews and to find out what people said.
            Use always questions as input. Never use this tool after you used
            recommend_conversation_question_tool!;""",
            output_variable="answers",
        )
        
        return search_tool

    def __create_comparison_tool(self) -> Tool:
        """Create a SearchTool for interview question answering.

        This constructs a Pipeline with a Retriever and PromptNode
        using a prompt template specialized for answering questions
        based on interview documents.

        Returns:
            Tool: The configured Q&A SearchTool object.
        """
        
        # Configure prompt template for answering questions
        # based on interview documents
        prompt_template = PromptTemplate(
            prompt="""
            You will be provided with excerpts from one or various interviews, each paragraph is 
            accompanied by a document ID (doc_id) indicating its source. The provided 
            excerpts contain the questions of the interviewer as well as the answers of the
            interviewee.   
            Answer the question truthfully based solely on the given documents. Make sure to 
            reference all relevant document IDs (doc_id) for each answer. If the documents 
            do not contain the answer to the question, say that answering is not possible 
            given the available information. Your answer should be no longer than 100 
            words.
            Documentation Content: {' - '.join([d.meta['doc_id']+': '+d.content for d in documents])};  
            Question:{query} 
            Answer:
            """,
            output_parser=AnswerParser(),
        )
        # Create PromptNode using GPT-3.5 Turbo model
        prompt_node = PromptNode(
            model_name_or_path=self.text_gen_model,
            api_key=self.openai_api_key,
            default_prompt_template=prompt_template,
            max_length=self.answer_length,
            model_kwargs={"temperature": self.model_temparature},
        )

        multi_doc_embedding_retriever = MultiDocEmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=self.embedding_model,
            api_key=self.openai_api_key,
            max_seq_len=1536,
        )

        # Create diversity ranker to optimze pipeline performance
        diversity_ranker = DiversityRanker()

        # Create "lost in the middle" ranker to optimze pipeline performance
        litm_ranker = LostInTheMiddleRanker(word_count_threshold=1024)

        # Construct pipeline with retriever, rankers and prompt node
        pipeline = Pipeline()
        pipeline.add_node(
            component=multi_doc_embedding_retriever,
            name="retriever",
            inputs=["Query"],
        )
        pipeline.add_node(
            component=diversity_ranker,
            name="diversity_ranker",
            inputs=["retriever"],
        )
        pipeline.add_node(
            component=litm_ranker,
            name="litm_ranker",
            inputs=["diversity_ranker"],
        )
        pipeline.add_node(
            component=prompt_node, name="prompt_node", inputs=["litm_ranker"]
        )

        # Create Tool encapsulating pipeline
        comparison_tool = Tool(
            name="comparison_tool",
            pipeline_or_node=pipeline,
            description="Useful for when you need to compare the statements or characteristics"
            "of different poeple.;",
            output_variable="answers",
        )

        return comparison_tool
    
    def __create_key_take_aways_tool(self) -> Tool:

        filter_retriever = FilterRetriever(
            document_store=self.document_store,
        )

        prompt_router_template = PromptTemplate(
            prompt="""
            You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about.
            Documentation Content: {join(documents)};  
            Key points:
            """,
        )

        recursive_prompt_template = PromptTemplate(
            prompt="""
            You are a proficient AI with a specialty in summarizing key information. Based on the extracted main points from the previous text, further summarize these key points into a concise overview. Distill the core ideas into short bullet point summaries that capture the essence of each point. Your goal is to provide a simplified list that someone could read to quickly understand the key ideas.
            Documentation Content: {join(documents)};  
            Key points:
            """,
        )

        # Create PromptNode using GPT-3.5 Turbo model
        prompt_router = PromptRouter(
            model_name_or_path=self.text_gen_model,
            api_key=self.openai_api_key,
            prompt_template=prompt_router_template,
            max_length=self.answer_length*3,
            split_by="doc_id",
            return_remaining=False,
            recursive_prompt=recursive_prompt_template,
        )

        # Create diversity ranker to optimze pipeline performance
        diversity_ranker = DiversityRanker()

        # Create "lost in the middle" ranker to optimze pipeline performance
        litm_ranker = LostInTheMiddleRanker(word_count_threshold=1024)

        prompt_template = PromptTemplate(
            prompt="""
            You are an AI assistant specialized in summarizing information. Below are bullet point lists summarizing key points extracted from multiple documents. Restructure these into a well-organized overview with the key points grouped under the corresponding document IDs. The goal is a concise and coherent summary structured as follows:
            doc_id:
            - Bullet points
            doc_id:
            - Bullet points
            The bullet points should distill the essence of each document's key points. The overall summary should allow a reader to quickly understand the main ideas from each original document. Focus on improving concision, clarity, and logical flow. Make edits to enhance readability while preserving accuracy.
            Documentation Content: {' - '.join(['doc_id '+d.meta['doc_id']+': '+d.content for d in documents])};  
            Question:{query} 
            Answer:
            """,
            output_parser=AnswerParser(),
        )
        # Create PromptNode
        prompt_node = PromptNode(
            model_name_or_path=self.text_gen_model,
            api_key=self.openai_api_key,
            default_prompt_template=prompt_template,
            max_length=self.answer_length*2,
            model_kwargs={"temperature": self.model_temparature},
        )

        # Construct pipeline with Retriever
        pipeline = Pipeline()
        pipeline.add_node(
            component=filter_retriever,
            name="filter-retriever",
            inputs=["Query"],
        )
        pipeline.add_node(
            component=prompt_router,
            name="prompt_router",
            inputs=["filter-retriever"],
        )

        pipeline.add_node(
            component=diversity_ranker,
            name="diversity_ranker",
            inputs=["prompt_router"],
        )
        pipeline.add_node(
            component=litm_ranker,
            name="litm_ranker",
            inputs=["diversity_ranker"],
        )

        pipeline.add_node(
            component=prompt_node, 
            name="prompt_node",
            inputs=["litm_ranker"]
        )

        # Create Tool encapsulating pipeline
        key_point_extraction_tool = Tool(
            name="key_point_extraction_tool",
            pipeline_or_node=pipeline,
            description="""Not recommended for general questions. Only useful when you need to summarize a document or extract the key points from a document.;""",
            output_variable="answers",
        )

        return key_point_extraction_tool  

    def __create_external_knowledge_tool(self) -> Tool:
        """Create an External Knowledge Tool.

        This constructs a Pipeline with PromptNode using a prompt 
        template specialized to interpret results.

        Returns:
            Tool: Tool that has access to external knowledge.
        """
        # Configure prompt template for answering questions
        # based on interview documents
        prompt_template = PromptTemplate(
            prompt="""
            You will be provided with excerpts from various interviews, each paragraph is 
            accompanied by a document ID (doc_id) indicating its source. The provided 
            excerpts contain the questions of the interviewer as well as the answers of the
            interviewee. If multiple excerpts share the same document ID (doc_id), it 
            signifies the same interviewee.
            
            Answer the question to the best of your capabilities. First seek to answer it
            using the facts and opinions stated in the referenced document sources. Then, 
            you may draw on your broader knowledge to supplement the response.

            Any additional knowledge, inferences or perspectives you add should align with the 
            context of the referenced interviews. Start with something like "In my opinion..." 
            to frame any supplemental information you provide. Conclude your response by stating 
            "(Additional context provided by GPT)" to distinguish supplemental information. 
            Also, include any sources or references (doc_id) if applicable.

            Documentation Content: {' - '.join(['doc_id '+d.meta['doc_id']+': '+d.content for d in documents])};  
            Question:{query} 
            Answer:
            """,
            output_parser=AnswerParser(),
        )

        # Create PromptNode
        prompt_node = PromptNode(
            model_name_or_path=self.text_gen_model,
            api_key=self.openai_api_key,
            default_prompt_template=prompt_template,
            max_length=self.answer_length,
            model_kwargs={"temperature": min(0, self.model_temparature-0.1)},
        )

        # Create diversity ranker to optimze pipeline performance
        diversity_ranker = DiversityRanker()

        # Create "lost in the middle" ranker to optimze pipeline performance
        litm_ranker = LostInTheMiddleRanker(word_count_threshold=1024)

        # Construct pipeline with retriever, rankers and prompt node
        external_knowledge_pipeline = Pipeline()
        external_knowledge_pipeline.add_node(
            component=self.embedding_retriever,
            name="retriever",
            inputs=["Query"],
        )
        external_knowledge_pipeline.add_node(
            component=diversity_ranker,
            name="diversity_ranker",
            inputs=["retriever"],
        )
        external_knowledge_pipeline.add_node(
            component=litm_ranker,
            name="litm_ranker",
            inputs=["diversity_ranker"],
        )
        external_knowledge_pipeline.add_node(
            component=prompt_node, name="prompt_node", inputs=["litm_ranker"]
        )

        # Create Tool encapsulating pipeline
        external_knowledge_tool = Tool(
            name="external_knowledge_tool",
            pipeline_or_node=external_knowledge_pipeline,
            description="""Useful when your interpretation, analysis, a generalized answer or reading between the lines is needed or the original documents do not contain enough information to fully answer the user's question.;""",
            output_variable="answers",
        )

        return external_knowledge_tool
