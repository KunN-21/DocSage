import os
import tempfile
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from .config import config

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from azure.core.credentials import AzureKeyCredential

_CONTENT_TYPE_MAP = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}

def _get_extension(file_name: str) -> str:
    return os.path.splitext(file_name)[1].lower()

def parse_with_azure(file_bytes: bytes, file_name: str) -> List[Document]:
    client = DocumentIntelligenceClient(
        endpoint= config.AZURE_DOC_INTEL_ENDPOINT,
        credential= AzureKeyCredential(config.AZURE_DOC_INTEL_KEY)
    )

    ext = _get_extension(file_name)
    content_type = _CONTENT_TYPE_MAP.get(ext, "application/octet-stream")

    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body= file_bytes,
        content_type= content_type,
        output_content_format= DocumentContentFormat.MARKDOWN,
    )
    result = poller.result()

    documents: List[Document] = []

    if result.pages: #Pages
        full_content = result.content or ""
        for page in result.pages:
            # Dùng page.spans để cắt text theo từng page
            page_text = ""
            if page.spans:
                for span in page.spans:
                    offset = span.offset
                    length = span.length
                    page_text += full_content[offset : offset + length]
            if page_text.strip():
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": file_name,
                            "page": page.page_number,
                            "parser": "azure_doc_intelligence",
                        },
                    )
                )
    else:
        # Fallback: 1 document duy nhất nếu không có pages
        documents.append(
            Document(
                page_content=result.content or "",
                metadata={"source": file_name, "parser": "azure_doc_intelligence"},
            )
        )
    return documents

        
def parse_with_pypdf(file_bytes: bytes) -> Tuple[List[Document], str]:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    return documents, tmp_file_path


def parse_document(uploaded_file) -> Tuple[List[Document], str, Optional[str]]:
   
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    ext = _get_extension(file_name)
    # Thử Azure trước
    if config.azure_available():
        try:
            documents = parse_with_azure(file_bytes, file_name)
            print(f"Parsed '{file_name}' with Azure Document Intelligence ({len(documents)} pages)")
            return documents, "azure", None
        except Exception as e:
            print(f"Azure DI failed: {e}, falling back to PyPDF...")
    # Fallback sang PyPDF (chỉ hỗ trợ PDF)
    if ext == ".pdf":
        documents, tmp_path = parse_with_pypdf(file_bytes)
        print(f"Parsed '{file_name}' with PyPDF ({len(documents)} pages)")
        return documents, "pypdf", tmp_path
    else:
        raise ValueError(
            f"Không thể parse file '{file_name}' (định dạng {ext}). "
            f"Cần cấu hình Azure Document Intelligence để hỗ trợ OCR cho ảnh."
        )