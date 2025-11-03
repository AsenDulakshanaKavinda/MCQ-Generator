import os
import sys
from operator import itemgetter
from typing import List, Optional, Dict, Any
import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import RetrievalQA, LLMChain # for a single use task

from mcq_gen.src.data_ingestion.chat_ingestor import ChatIngestor

from mcq_gen.exception import ProjectException
from mcq_gen.prompts.prompt_library import prompt, custom_prompt
from mcq_gen.logger import logging as log
from mcq_gen.utils.model_loader import ModelLoader



class MCQGenRAG:
    def __init__(
            self, 
            session_id: Optional[str], 
            retriever=None,
            result_base = "results"
            
        ):
        """
        Handles loading the LLM, retriever, and building the MCQ generation chain.
        """
        try:
            self.session_id = session_id

            # save generated 
            self.result_base = Path(result_base); self.result_base.mkdir(parents=True, exist_ok=True)
            self.results_dir = self._resolve_dir(self.result_base)
            
            # load the llm model
            self.llm = self._load_llm()

            # Load initial and refine prompts from registry
            self.prompt = prompt

            self.retriever = retriever
            self.chain = None

            log.info(f"MCQGenRAG initialized, session_id={self.session_id}")

        except Exception as e:
            log.error(f"Failed to initialize MCQGenRAG, error={str(e)}")
            raise ProjectException("Initialization error in MCQGenRAG", sys)

    # -----------------------------------------------------------
    # Load FAISS retriever
    # -----------------------------------------------------------
    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "mmr",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load FAISS index and build retriever + chain.
        """
        try:
            if not os.path.isdir(index_path):
                raise ProjectException(f"FAISS index directory not found: {index_path}")

            embedding = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embedding,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )
            
            search_kwargs = {"k": k}
            self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

            self._build_chain()

            log.info("FAISS retriever loaded successfully",)

            return self.retriever

        except Exception as e:
            log.error(f"Failed to load retriever from FAISS, error={str(e)}")
            raise ProjectException("Loading error in MCQGenRAG", sys)
        
    # -----------------------------------------------------------
    # create results dir / session
    # -----------------------------------------------------------
    def _resolve_dir(self, base: Path):
        # if use_session the make session dirs
        if self.session_id:
            d = base / self.session_id # "results/abc123"
            d.mkdir(parents=True, exist_ok=True)
            return d
        return base # "results/"

    # -----------------------------------------------------------
    # Load LLM
    # -----------------------------------------------------------
    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ProjectException("LLM could not be loaded.", sys)
            log.info(f"LLM loaded successfully, session_id={self.session_id}")
            return llm
        except Exception as e:
            log.error(f"Failed to load LLM, error={str(e)}")
            raise ProjectException("LLM loading error in MCQGenRAG", sys)

    # -----------------------------------------------------------
    # chain
    # -----------------------------------------------------------
    def _build_chain(self):
        try:
            if self.retriever is None:
                raise ProjectException("No retriever, set before building again", sys)
            
            self.chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",         # combines all retrieved docs into one prompt
                retriever=self.retriever,
                return_source_documents=False  # optional, gives you which docs were used
            )
            log.info(f"LCEL chain built successfully, session_id={self.session_id}")

        except Exception as e:
            log.error(f"Failed to build chain, error={str(e)}")
            raise ProjectException("Error building chain", sys)


    # -----------------------------------------------------------
    # Main invoke function
    # -----------------------------------------------------------
    def genetate(self):
        prompt = self.prompt
        result = self.chain.invoke(prompt)
        self._save_as_json(result)
        return result

    # -----------------------------------------------------------
    # extract and save as json format
    # -----------------------------------------------------------
    def _save_as_json(self, raw_data):
        try:
            if raw_data:
                # Extract JSON string from result
                raw_result = raw_data['result'].strip()
                if raw_result.startswith("```json"):
                    raw_result = raw_result[len("```json"):].strip()
                if raw_result.endswith("```"):
                    raw_result = raw_result[:-3].strip()

                # Parse JSON
                mcq_list = json.loads(raw_result)

                # Save to correct path
                output_file = self.results_dir / f"{self.session_id or 'default'}.json"

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(mcq_list, f, ensure_ascii=False, indent=4)

                log.info(f"MCQs saved successfully to {output_file}")

        except Exception as e:
            log.error("Failed to save result to a json file.")
            raise ProjectException(f"{str(e)}", sys)




