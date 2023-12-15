from fuzzywuzzy import process
from haystack import Pipeline
from haystack.nodes import (
    AnswerParser,
    PromptNode,
    PromptTemplate,
)
from haystack_wrappers.quotes_retriever import HaystackQuotesRetriever
from haystack_wrappers.qna_agent import HaystackQnAAgent
from haystack_wrappers.helper_agent import HaystackHelperAgent
from haystack_wrappers.wrapper import HaystackWrapper
from haystack_wrappers.utils import recreate_conversation

from utils.logs import logger

class HaystackIntentClassifier(HaystackWrapper):
    """
    A class for classifying query intents and routing to Haystack pipelines.

    This class can classify a query into intents like "summaries",
    "text passage", etc. It then instantiates the corresponding Haystack
    component for that intent.

    Attributes:
        labels (List[str]): The possible intent labels
        query_classifier (TransformersQueryClassifier): The query classifier
            model
        intents (Dict[str, HaystackWrapper]): Mapping of intent to pipeline

    Methods:
        classify_intent - Classifies the query and returns a Haystack pipeline
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo-16k", temparature: float = 0.5, answer_length: str = "medium", chat_history: dict = {"input": [], "output": []}):
        super().__init__(model_name=model_name, temparature=temparature, answer_length=answer_length)
        self._chat_history = chat_history
        self.query_classifier = self._query_classifier()
        self.intents = {
            "quotes": HaystackQuotesRetriever,
            "tool": HaystackHelperAgent,
            "document": HaystackQnAAgent,
        }

    def classify_intent(self, query: str) -> HaystackWrapper:
        """
        Classify the query and return the corresponding Haystack pipeline.

        Args:
            query (str): The input natural language query

        Returns:
            HaystackWrapper: The Haystack pipeline for the predicted intent
        """
        result = self.query_classifier.run(query=query)
        intent = result[0]["answers"][0].answer.lower()
        # Get the closest match to the predicted intent, as LLM may return the 
        # intent slightly differently.
        closest_match = process.extractOne(intent, self.intents.keys())[0]

        try: 
            # Chose agent depending on intent
            agent = self.intents[closest_match]
            logger.info(f"Agent: {agent.__name__}")
            # If intent is unclear use HaystackQnAAgent
            if not agent:
                agent = HaystackQnAAgent
        except:
            agent = HaystackQnAAgent

        return agent

    def _query_classifier(self) -> Pipeline:
        """
        Instantiate the query classifier.

        Returns:
            Pipeline: The query classifier pipeline
        """
        conversation = recreate_conversation(self._chat_history, reversed_order=False, current_message=False)

        # Configure prompt template to reconstruct the question that was to to retrieve
        # the interview passages
        prompt_template = PromptTemplate(
            prompt="""
            Given a user's input and the conversation history, determine the intent of the 
            current query from the following predefined categories: "document", "quotes", 
            or "tool". Your response should be the intent category that best describes the input.

            documents: 
            general questions and queries that are related to content of the interview documents
            i.e.: "What did he say about X?", "Who is X?", "Who did...?", "Who said...?", "Who had..." "What does X do?", "What is important about X?", "How important is X to...?", "According to the interviewees,...", "Summarize the key points...", "Compare the...", "What is the difference between X and Y?", "What do you know about?", "What do you think about...?", "How do you think...?", "What is your opinion on...?", "Please expand on this...";

            quotes: 
            queries that seek to extract quotes from the interview documents
            i.e.: "Show me the text passages that contain the answer", "Quote the text source", "Show me the source of the answer";

            tool: queries that seek to extract information, capabilities or thinking process of this tool
            i.e.: "What is your name?", "What can you do?", "How can you support me?", "What tools do you have?", "What models do you use to generate your answers?", "Where is my data stored?", "Please propose a question I could ask you.", "What else might be of interest to me?" "How did come up with this answer?", "Please explain your thoughts?", "Explain how you came up with your answer", "Explain how you came to this conclusion";

             
            Conversation:""" + conversation + """;
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

        return prompt_node