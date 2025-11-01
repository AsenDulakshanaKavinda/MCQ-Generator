
import sys
from pathlib import Path
from typing import Iterable, List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from fastapi import UploadFile
from mcq_gen.exception import ProjectException
from mcq_gen.logger import logging as log

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

def load_documents(paths: Iterable[Path]) -> List[Document]:
    log.info("load documents started...")

    docs: List[Document] = []
    try:
        for path in paths:
            ext = path.suffix.lower()

            if ext == ".pdf":
                loader = PyPDFLoader(str(path))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(path))
            elif ext == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
            else:
                log.info(f"Unsupported extension skipped, path={str(path)}")
                continue
            docs.extend(loader.load())

        log.info(f"load document complited, {len(docs)} documents loaded...")
        return docs
       
            
    except Exception as e:
        ProjectException(f"Failed loading documents, error={str(e)}", sys)












