import io

import docx
import PyPDF2


class File:
    def __init__(self, name, _type, content):
        self.name = name
        self._type = _type
        self.file_content = content

    @classmethod
    def create_file(cls, name, file_type, content):
        if file_type == "application/pdf":
            return PDFFile(name, file_type, content)
        elif file_type == "text/plain":
            return TXTFile(name, file_type, content)
        elif (
            file_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"  # noqa
        ):
            return DOCFile(name, file_type, content)
        else:
            raise ValueError(
                "Unsupported file format. Only PDF, TXT, and DOCX are "
                "supported."
            )


class PDFFile(File):
    def read(self) -> str:
        # Create a PDF reader object from the content
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(self.file_content))

        # Extract text from each page of the PDF
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text


class TXTFile(File):
    def read(self) -> str:
        return self.file_content.decode("utf-8")


class DOCFile(File):
    def read(self) -> str:
        doc = docx.Document(io.BytesIO(self.file_content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
