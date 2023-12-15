from api import signals
from django.dispatch import receiver
from haystack_wrappers.doc_explorer import HaystackDocExplorer


@receiver(signals.pinecone_upload_done)
def start_doc_explorer(sender, doc_id, text, **kwargs) -> None:
    """Initiates the generation process of the information that is
    required for the document summary page. It's initialized after the
    document has been uploaded to Pinecone.

    This receiver function is triggered when the pinecone_upload_done signal
    is emitted after a document is uploaded. It creates an exploration page
    for the document using the HaystackDocExplorer class.

    Args:
        sender (object): The object that emitted the signal.
        doc_id (str): The unique ID of the uploaded document.
        text (int): The text content of the uploaded document.
        **kwargs: Additional keyword arguments from the signal.

    Returns:
        None
    """
    doc_explorer = HaystackDocExplorer()
    doc_explorer.create_doc_exploration_page(doc_id=doc_id, text=text)


@receiver(signals.pinecone_upload_done)
def start_question_explorer(sender, doc_id, text, **kwargs) -> None:
    """Initiates the generation process of the information that is
    required for the question exploration page. It's initialized
    after the document has been uploaded to Pinecone.

    This receiver function is triggered when the pinecone_upload_done signal
    is emitted after a document is uploaded. It creates an questions
    exploration page for the document using the HaystackDocExplorer class.

    Args:
        sender (object): The object that emitted the signal.
        doc_id (str): The unique ID of the uploaded document.
        text (str): The text content of the uploaded document.
        **kwargs: Additional keyword arguments from the signal.

    Returns:
        None
    """
    doc_explorer = HaystackDocExplorer(use_hugging_face=False)
    doc_explorer.create_question_exploration_page(
        doc_id=doc_id,
    )
