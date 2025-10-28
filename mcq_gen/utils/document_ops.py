
from pathlib import Path
from typing import Iterable, List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from mcq_gen.logger import GLOBAL_LOGGER as log
from mcq_gen.exception.custom_exception import ProjectException
from fastapi import UploadFile

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

def load_documents(paths: Iterable[Path]) -> List[Document]:
    log.info(f"Documents loading ....")
    log.info(f"Documents about to load: {str(paths)}")
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()
            log.info(f"trying to load {p} with {ext}")
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
            else:
                log.warning("Unsupported extension skipped", path=str(p))
                continue
            docs.extend(loader.load())
        log.info("Documents loaded", count=len(docs))
        return docs
    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise ProjectException("Error loading documents", e)

class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to a simple object with .name and .getbuffer()."""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()







