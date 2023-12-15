class PineConeUploadError(Exception):
    """Raise this exception when file was not
    successfully uploaded to Pinecone.
    """


class PineConeDeletionError(Exception):
    """Raise this exception when file was not
    successfully deleted from Pinecone.
    """
