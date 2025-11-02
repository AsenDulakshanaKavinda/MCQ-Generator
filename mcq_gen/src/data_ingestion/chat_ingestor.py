from datetime import datetime
import sys
import uuid
from typing import Optional, List, Iterable
from pathlib import Path
from mcq_gen.src.data_ingestion.faiss_manager import FaissManager

from mcq_gen.exception import ProjectException
from mcq_gen.logger import logging as log
from mcq_gen.utils.model_loader import ModelLoader
from mcq_gen.utils.file_io import save_uploaded_files
from mcq_gen.utils.document_ops import load_documents
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def generate_session_id() -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"

class ChatIngestor:
    def __init__(
            self,
            temp_base: str = "data", # keep data after load and give a clean name
            faiss_base: str = "faiss_index", # keep vs data
            use_session_dirs: bool = True,
            use_txt_chunking: bool = False,
            session_id: Optional[str] = None,
    ):
        try:
            self.model_loader = ModelLoader()

            self.use_session = use_session_dirs
            self.use_txt_chunking = use_txt_chunking
            self.session_id = session_id or generate_session_id()
            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)


            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

        except Exception as e:
            log.error(f"Failed to initialize ChatIngestor, error={str(e)}")
            raise ProjectException("Initialization error in ChatIngestor", sys)



    def _resolve_dir(self, base: Path):
        # if use_session the make session dirs
        if self.session_id:
            d = base / self.session_id # "faiss_index/abc123"
            d.mkdir(parents=True, exist_ok=True)
            return d
        return base # "faiss_index/"
    
    def _doc_splitter(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        doc_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )
        chunks = doc_splitter.split_documents(docs)
        log.info(f"Documents split, chunks={len(chunks)}, chunk_size={chunk_size}, overlap={chunk_overlap}")
        return chunks

    def _txt_splitter(self, text: str, chunk_size=1000, chunk_overlap=200) -> List[str]:
        txt_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )
        chunks = txt_splitter.split_text(text)
        log.info(f"Documents split, chunks={len(chunks)}, chunk_size={chunk_size}, overlap={chunk_overlap}")
        return chunks
    
    def build_retriever(
            self,
            uploaded_files: Iterable,
            *,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            k: int = 5,
            search_type: str = "mmr",
            fetch_k: int = 20,
            lambda_mult: float = 0.5
    ):
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            text = " "
            docs = load_documents(paths)

            fm = FaissManager(self.faiss_dir, self.model_loader)
            
            if self.use_txt_chunking:
                pass
            else:
                if not docs:
                    raise ProjectException("No valid documents loaded")
                
                chunks = self._doc_splitter(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                texts = [c.page_content for c in chunks]
                metas = [c.metadata for c in chunks]

                try:
                    vs = fm.load_or_create(texts=texts, metadatas=metas)
                except Exception:
                    vs = fm.load_or_create(texts=texts, metadatas=metas)

                added = fm.add_documents(chunks)
                log.info(f"FAISS index updated, added={added}, index={str(self.faiss_dir)}")

                # Configure search parameters
                search_kwargs = {"k": k}

                result = vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
                log.info("build_retriever completed...")
                print(f"type of vs: {type(result)}")
                return result

        except Exception as e:
            log.error(f"Failed to build retriever, error={str(e)}")
            raise ProjectException("Failed to build retriever", sys)

        






