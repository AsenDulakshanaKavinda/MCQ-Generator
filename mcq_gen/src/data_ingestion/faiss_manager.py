
import hashlib
import json
import sys
from typing import Optional, Any, Dict, List
from pathlib import Path
from mcq_gen.utils.model_loader import ModelLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from mcq_gen.exception import ProjectException
from mcq_gen.logger import logging as log

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = index_dir # create faiss_index dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: dict[str: Any] = {"rows": {}} # this is dict of rows

        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {} # load it if already there
            except Exception:
                self._meta = {"raws": {}} # init the empty one if dones not exists

        self.model_loader = model_loader or model_loader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None

    # make sure both index.faiss and index.pkl exists
    def _exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def add_documents(self, docs: List[Document]):
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents_idempotent().")

        new_docs: List[Document] = []

        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)

        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)

    def load_or_create(self,texts:Optional[List[str]]=None, metadatas: Optional[List[dict]] = None):
        ## if we running first time then it will not go in this block
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs


        if not texts:
            raise ProjectException("No existing FAISS index and no data to create one", sys)
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs


