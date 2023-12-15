import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from utils.logs import logger
from haystack.schema import Answer


def resolver_function(query, agent, agent_step):
    """Resolve API requests for an agent memory.

    Args:
        query (str): The original user query string
        agent (Agent): The Agent instance with access to memory
        agent_step (AgentStep): The AgentStep instance with latest
            dialogue state

    Returns:
        dict: A dictionary with the following contents:
            - query: The original user query string
            - tool_names_with_descriptions: Tool metadata from the Agent
            - transcript: The dialogue history
            - memory: The Agent's memory state
    """
    return {
        "query": query,
        "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),  # noqa
        "transcript": agent_step.transcript,
        "memory": agent.memory.load(),
    }


def extract_referenced_documents(text: str):
    """
    Extracts and replaces referenced document identifiers in a given text.

    This function searches for document references in the provided text and replaces them
    with a numerical representation enclosed in square brackets. It also returns a list of
    unique document identifiers found.

    Args:
        text (str): The input text in which document references should be extracted.

    Returns:
        str: The modified text with document references replaced.
        List[str]: A list of unique document identifiers found in the text.

    Example:
        input_text = "Please see document 123 and doc_id: 456 for more information."
        modified_text, unique_ids = extract_referenced_documents(input_text)
        # modified_text will be "Please see [1] and [2] for more information."
        # unique_ids will be ["123", "456"]
    """
    # Regex pattern to match (doc: NUMBER) with optional suffix
    pattern = r'(doc_id|document|Document|document ID|Document ID):?\s*(\d+)\b'

    # Find all matches 
    matches = re.findall(pattern, text)
    doc_ids = [match[1] for match in matches]
    # Replace text  
    replaced_text = text
    counter = 1
    for doc_id in doc_ids:
        # Replace match with the same number and keep suffix
        replaced_text = re.sub(
            r'(doc_id|document|Document|document ID|Document ID):?\s*' + doc_id + r'\b',
            f"[{str(counter)}]",
            replaced_text
        )
        counter += 1

    unique_doc_ids = sorted(set(doc_ids), key=doc_ids.index)
    
    return replaced_text, unique_doc_ids


def remove_stop_words(text: str) -> str:
    """
    Removes stop words from the given text.

    Args:
        text (str): The input text from which stop words will be removed.

    Returns:
        str: The text with stop words removed.

    Raises:
        LookupError: If the required NLTK resources are not found, they will be downloaded.

    Example:
         remove_stop_words("This is a sample sentence.")
        'sample sentence'
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    lower_case_text = text.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(lower_case_text)

    filtered_text = [w for w in word_tokens if w not in stop_words]
 
    concatenated = " ".join(filtered_text)
    return concatenated


def recreate_conversation(chat_history: dict, reversed_order: bool = True, current_message: bool = False) -> str:
    """
    Recreate the conversation from the chat history.

    Args:
        chat_history (dict): The chat history.
        reversed_order (bool, optional): Whether the order of questions and answers is reversed. Defaults to True.
        current_message (bool, optional): Whether to include the current message. Defaults to False.

    Returns:
        str: The conversation.
    """
    # Return empty string if there is no chat history
    if not chat_history["output"]:
        return ""
    
    # Reverse the order of the questions and answers
    if reversed_order:
        chat_history["input"].reverse()
        chat_history["output"].reverse()

    # Remove the current message
    if not current_message and reversed_order:
        chat_history["input"].pop(0)
    elif not current_message and not reversed_order:
        chat_history["input"].pop(-1)

    # Merge all inputs and outputs with labels (question or answer)
    chat_inputs = [(x[0], x[1], "input") for x in chat_history["input"]]
    chat_outputs = [(x[0], x[1], "output") for x in chat_history["output"]]

    # Merge and sort all inputs and outputs
    chat_history_sorted = chat_inputs + chat_outputs
    chat_history_sorted.sort(key=lambda x: x[0])

    # Recreate the conversation
    conversation_parts: list = []

    for m in chat_history_sorted:
        if m[2] == "input":
            conversation_parts.append("User: " + m[1])
        if m[2] == "output":
            conversation_parts.append("Qwizz: " + m[1])

    conversation = '\n'.join(conversation_parts)

    return conversation


def prepare_data_for_memory(chat_history: dict) -> dict:
    # Remove the current message
    chat_history["input"].pop(0)
    
    # Reverse the order of the questions and answers to 
    # chronological order
    chat_history["input"].reverse()
    chat_history["output"].reverse()

    # Merge all inputs and outputs with labels (question or answer)
    chat_inputs = [(x[0], x[1], "input") for x in chat_history["input"]]
    chat_outputs = [(x[0], x[1], "output") for x in chat_history["output"]]
    
    # Merge and sort all inputs and outputs
    chat_history_sorted = chat_inputs + chat_outputs
    chat_history_sorted.sort(key=lambda x: x[0])

    # Recreate the conversation
    conversation_parts = {"input": [], "output": []}

    for m in chat_history_sorted:
        if m[2] == "input":
            conversation_parts["input"].append(m[1])
        if m[2] == "output":
            conversation_parts["output"].append(m[1])

    return conversation_parts



def extract_answer_from_prompt_output(answers: list[Answer]) -> str:
    """Extracts the answer from a prompt output dictionary.

    This function assumes that the prompt template does not an `output_parser`
    and that the prompt node defines the `output_variable` as 'answer'. (see 
    quoting_tool)

    Args:
        output (dict): The output dictionary returned by the prompt.

    Returns:
        str: The original question string.
    """
    return answers[0].answer


def register_custom_shaper_functions(func):
    """Registers custom functions for Haystack's Shaper component.

    Extends the global REGISTERED_FUNCTIONS dict in haystack.nodes.other.shaper
    to allow additional functions to be called from a Shaper node.

    Args:
        func: The function to register.
    """
    from haystack.nodes.other import shaper  
    shaper.REGISTERED_FUNCTIONS[func.__name__] = func 