from django import dispatch

pinecone_upload_done = dispatch.Signal(["doc_id", "text"])
"""Signal emitted when a document has been uploaded to Pinecone.

Args:
doc_id (str): The unique ID of the uploaded document.
text (str): The text content of the uploaded document.

This signal is emitted after a document is successfully uploaded to the 
Pinecone service. Any receivers listening for this signal can perform actions 
such as creating an exploration page for the document, indexing it, etc. The 
doc_id and text are passed along with the signal to provide context and 
content.
"""
