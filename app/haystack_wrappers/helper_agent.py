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
from haystack_wrappers.haystack_custom import MultiDocEmbeddingRetriever, PromptRouter
from haystack_wrappers.haystack_custom import HaystackToolsManager as ToolsManager

from haystack_wrappers.utils import prepare_data_for_memory, register_custom_shaper_functions, resolver_function, recreate_conversation
from haystack_wrappers.wrapper import HaystackWrapper

from utils.logs import logger


class HaystackHelperAgent(HaystackWrapper):
    """
    A class that allows to look inside the AI Agent's mind and see what it is capable 
    of and how it came up with its answer.
    
    Attributes:
        embedding_model (str): Name of embedding model
        openai_api_key (str): API key for accessing OpenAI models

    Methods:
        run: Runs the QA pipeline to answer questions
        __setup_conversational_agent: Creates a ConversationalAgent
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo-16k", temparature: float = 0.5, answer_length: str = "medium", chat_history: dict = {"input": [], "output": []}):
        super().__init__(model_name=model_name, temparature=temparature, answer_length=answer_length)
        self._chat_history = chat_history

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

        """
        
        # Create memory and search tools
        memory_tool, summary_memory = self.__create_memory_tool()
        helper_tool = self.__create_helper_tool()
        reccomended_question_tool = self.__create_recommended_question_tool()
        explainer_tool = self.__create_explainer_tool()

        # Initialize conversational agent
        conversational_agent = self.__setup_conversational_agent(
            summary_memory,
            tools=[helper_tool, reccomended_question_tool, explainer_tool],  # memory_tool        
        )
        
        # Run agent and generate answer
        result = conversational_agent.run(
            query=query,
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
            - Custom prompt template defining conversation flow and tool use
            - Access to provided tools
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
        interacts with you, 'Qwizz' (an AI Agent), to analyse qualitative interview 
        documents. The human user poses questions, and you should through a few
        steps to provide a detailed answers. You must
        use the available tools to find the required information in the
        qualitative interview documents. The final answer to the question
        should be truthfully based solely on the output of the
        tools and keep all references (doc_id) intact if any were given. If 
        you cannot answer the question, say I don't know or ask for 
        the missing piece of information. You should try to answer
        the questions using only these tools:
        {tool_names_with_descriptions};
        You cannot use any other tools!
        Note the output of the 'recommend_question_tool' is a question. Consequently,
        do not answer these questions. Forward the questions to human user.

        Your responses must start with one of the following:

        Thought: [the your reasoning process.] Tool: [tool names] (on a
        new line) Tool Input: [input as a question for the selected tool
        WITHOUT quotation marks and on a new line] (These must always be
        provided together and on separate lines.) Observation: [tool's
        result, can also be question] Final Answer: [Final answer to the 
        human user's request. Query: {query}; Your answer might also be 
        in form of a question. The content of your final answer can be found
        in your thoughts and observations. Ensure that answers is from 
        your perspective (the "I" perspective). You are NOT allowed to 
        say that the answer is mentioned above or in...!.

        When selecting a tool, you must provide both the "Tool:"
        and "Tool Input:" pair in the same response, but on separate lines

        The you can ask the human user for additional information,
        clarification, or context if required. If you cannot find
        a specific answer after exhausting available tools and approaches,
        it answers with Final Answer: I cannot answer this question. Please
        provide additional documents or rephrase your question. (If possible
        tell the human why you cannot answer the question and what you know.)

        The following is the previous conversation between a human and you:
        {memory};
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
            
    def __create_helper_tool(self) -> Tool:
        # Configure prompt template for answering questions
        # based on interview documents

        prompt_template = PromptTemplate(
            prompt="""
            You are an AI agent. Your name is 'Qwizz'. You are equipped with several capabilities to assist with qualitative document analysis:

            Search - 'Qwizz' can thoroughly search through qualitative documents to find relevant information that answers a user's questions. It utilizes available tools to retrieve pertinent passages across multiple documents.

            Compare - 'Qwizz' can compare and contrast what different documents or sources say about a particular topic. This allows for nuanced analysis from multiple perspectives.

            Propose questions - Based on its understanding of the qualitative data, 'Qwizz' can propose potential questions a user may want to ask to gain more insights. This helps guide the analysis in fruitful directions.

            Truthful, grounded answers - 'Qwizz' constructs its answers solely based on the output of the tools used to analyze the qualitative data. Answers are truthful and reference the specific document IDs they are derived from.

            Acknowledge limitations - If 'Qwizz' cannot determine an answer from the available data, it will transparently acknowledge the limitations of its knowledge and ask the user for clarification.

            'Qwizz' leverages its analytical capabilities within the boundaries of the tools provided to offer thoughtful, thorough assistance with qualitative document analysis. It aims to provide helpful information to users while maintaining transparency. However, it should be noted that 'Qwizz' requires inputs from the user to function properly. It cannot answer questions without the necessary information. It is also not a replacement for human analysis. It is a tool to assist with qualitative document analysis.

            Technical details:
                - LLM Models: GTP 3.5 with context window of 16k and GTP 4 with context window of 32k are used.
                - LLM orchestration is handled by Haystack.
                - The backend uses Django framework.
                - The frontend is built with ReactJS.
                - Deployment is on DigitalOcean and Heroku. Originally fully on Heroku but moved backend to DigitalOcean for more resources. Frontend remains on Heroku.
                - Generated data is stored in a PostgreSQL database on our DigitalOcean server. Uploaded documents are stored in an AWS S3 bucket and Pinecone vector database hosted in Singapore.

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

        # Construct pipeline with prompt node
        pipeline = Pipeline()
        pipeline.add_node(
            component=prompt_node, name="prompt_node", inputs=["Query"]
        )

        # Create Tool encapsulating pipeline
        helper_tool = Tool(
            name="helper_tool",
            pipeline_or_node=pipeline,
            description="Useful for when the human user wants to know what you can do or has questions about technical details.Do not use this tool after using recommend_question_tool;""",
            output_variable="answers",
        )

        return helper_tool

    def __create_recommended_question_tool(self) -> Tool:
        conversation = recreate_conversation(self._chat_history)
        prompt_template = PromptTemplate(
            prompt="""Given the conversation, please generate some interesting questions 
            that the user might ask next. Please make sure that the questions have not been 
            asked before. Formulate your answer as questions. Never answer the questions.
            
            If you are not given the conversation, tell the user that you can 
            only advise them with new questions if you know what they are interested in. 
            To do this, they need to interact with 'Qwizz' first. Instead, advise the 
            user to go to the Document 'Exploration Page' to gain an oversight over 
            the document.

            Conversation:""" + conversation + """;
            Answer:
            """,
            output_parser=AnswerParser(),
        )

        pipe = Pipeline()

        prompt_node = PromptNode(
            model_name_or_path=self.text_gen_model,
            api_key=self.openai_api_key,
            default_prompt_template=prompt_template,
            model_kwargs={"temperature": self.model_temparature},
            max_length=self.answer_length,
        )

        pipe.add_node(
            component=prompt_node, name="prompt_node", inputs=["Query"]
        )

        # Create Tool encapsulating pipeline
        recommend_question_tool = Tool(
            name="recommend_question_tool",
            pipeline_or_node=pipe,
            description="""Useful to propse questions that the user could ask next. Generated questions must not be answered.;""",
            output_variable="answers",
        )

        return recommend_question_tool
    
    def __create_explainer_tool(self) -> Tool:
        """
        """
        conversation = recreate_conversation(self._chat_history)
        # Configure prompt template for answering questions
        # based on interview documents
        prompt_template = PromptTemplate(
            prompt="""
            You will be provided with the conversation history, which 
            encompasses the user's queries and your thought process 
            leading to your response. 
            Please provide a detailed explanation of how you arrived at your previous response. 
            Break down the steps or thought process you followed to generate that information.
            Ensure that your explanation argues from your perspective (the "I" 
            perspective). Include any sources or references if applicable. Make sure the explanation is clear and concise. 
            Conversation: """ + conversation + """;
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

        # Construct pipeline with retriever, rankers and prompt node
        generative_pipeline = Pipeline()

        generative_pipeline.add_node(
            component=prompt_node, name="prompt_node", inputs=["Query"]
        )

        # Create Tool encapsulating pipeline
        explainer_tool = Tool(
            name="explainer_tool",
            pipeline_or_node=generative_pipeline,
            description="""Only useful when the human user wants to know how you came up with an answer or asks about your chain of thoughts. Do not use this tool after using recommend_question_tool;""",
            output_variable="answers",
        )

        return explainer_tool   
