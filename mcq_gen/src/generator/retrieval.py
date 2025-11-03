import os
import sys
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from mcq_gen.model.models import PromptType
from mcq_gen.exception import ProjectException
from mcq_gen.prompts.prompt_library import PROMPT_REGISTRY
from mcq_gen.logger import logging as log

from mcq_gen.utils.model_loader import ModelLoader



class MCQGenRAG:
    def __init__(self, session_id: Optional[str], retriever=None):
        """
        Handles loading the LLM, retriever, and building the MCQ generation chain.
        """
        try:
            self.session_id = session_id
            self.llm = self._load_llm()

            # Load initial and refine prompts from registry
            self.system_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.SYSTEM_PROMPT.value
            ]
            self.ai_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.AI_PROMPT.value
            ]
            self.human_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.HUMAN_PROMPT.value
            ]

            self.retriever = retriever
            self.chain = None

            if self.retriever is not None:
                self._build_lcel_chain()

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
        Load FAISS index and build retriever + LCEL chain.
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

            # Build retriever search configuration
            """ if search_kwargs is None:
                search_kwargs = {"k": k}
                if search_type == "mmr":
                    search_kwargs["fetch_k"] = fetch_k
                    search_kwargs["lambda_mult"] = lambda_mult """
            
            search_kwargs = {"k": k}
            self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

            # self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded successfully",
                # index_path=index_path,
                # index_name=index_name,
                # search_type=search_type,
                # k=k,
                # fetch_k=fetch_k if search_type == "mmr" else None,
                # lambda_mult=lambda_mult if search_type == "mmr" else None,
                # session_id=self.session_id,
            )

            return self.retriever

        except Exception as e:
            log.error(f"Failed to load retriever from FAISS, error={str(e)}")
            raise ProjectException("Loading error in MCQGenRAG", sys)

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
    # Document formatting helper
    # -----------------------------------------------------------
    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    # -----------------------------------------------------------
    # Main invoke function
    # -----------------------------------------------------------
    def invoke(self, user_input: str) -> str:
        """
        Invoke the chain to generate or refine MCQs.
        Args:
            user_input: Topic or question text
            existing_answer: (optional) Previously generated MCQs for refinement
        """
        if self.chain is None:
            raise ProjectException(
                "Chain not initialized. Call _build_lcel_chain first.", sys
            )

        try:
            payload = {"topic": user_input}

            result = self.chain.invoke(payload)
            return result

        except Exception as e:
            log.error(f"Chain invocation failed, error={str(e)}, session_id={self.session_id}")
            raise ProjectException("Failed to generate MCQs", e)

    # -----------------------------------------------------------
    # Build LCEL chain
    # -----------------------------------------------------------
    def _build_lcel_chain(self):
        """
        Build the LCEL chain for MCQ generation.
        - If user provides a topic/question: generate MCQs about that topic.
        - If user provides no details: generate general MCQs from the document.
        """

        try:
            if self.retriever is None:
                raise ProjectException("No retriever set before building again", sys)
            
            prompt = ChatPromptTemplate([
                self.system_prompt,
                self.human_prompt,
                self.ai_prompt
            ])

            format_docs = RunnablePassthrough(self._format_docs)
            
            self.chain = (
                {
                    "context": self.retriever | format_docs
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            log.info(f"LCEL chain built successfully, session_id={self.session_id}")

        except Exception as e:
            log.error(f"Failed to build LCEL chain, error={str(e)}")
            raise ProjectException("Error building LCEL chain", sys)



        










