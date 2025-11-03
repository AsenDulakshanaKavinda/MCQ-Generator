import os
import sys
from operator import itemgetter
from typing import List, Optional, Dict, Any
import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda


from mcq_gen.exception import ProjectException
from mcq_gen.prompts.prompt_library import prompt, custom_prompt_v1
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

            self.retriever = retriever
            


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
                raise ProjectException(f"FAISS index directory not found: {index_path}", sys)


            embedding = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embedding,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )
            
            search_kwargs = {"k": k}
            self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


            log.info("FAISS retriever loaded successfully")

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
    # prompt
    # -----------------------------------------------------------
    def _setup_prompt(self):
        
        try:
            mcq_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", custom_prompt_v1),
                    ("human", "Generate the MCQs now based on the topic: '{topic}'"), 
                ]
            )
            log.info("Setting up prompt complited.")
            return mcq_prompt
            
        except Exception as e:
            raise ProjectException(f"something went wrong while set-up prompt error={e}", sys)


    # -----------------------------------------------------------
    # chain
    # -----------------------------------------------------------
    def _build_chain(self):
        try:
            if self.retriever is None:
                raise ProjectException("No retriever, set before building again", sys)
            

            mcq_chain = (
                RunnableParallel(
                    # 'context' key gets populated by feeding the 'topic' to the retriever
                    context=lambda x: self.retriever.invoke(x['topic']), 
                    # 'topic' key simply passes the original input topic through
                    topic=RunnablePassthrough() 
                )
                | self._setup_prompt()
                | self._load_llm()
                | RunnableLambda(lambda msg: {"result": msg.content})

            )
            log.info(f"LCEL chain built successfully, session_id={self.session_id}")
            return mcq_chain
            
        except Exception as e:
            log.error(f"Failed to build chain, error={str(e)}")
            raise ProjectException("Error building chain", sys)


    # -----------------------------------------------------------
    # Main invoke function
    # -----------------------------------------------------------

    def generate(self, topic: str):
        chain = self._build_chain()
        response = chain.invoke({"topic": topic})
        self._save_as_json(response)
        return response

    # -----------------------------------------------------------
    # extract and save as json format
    # -----------------------------------------------------------
    def _save_as_json(self, raw_data):
        try:
            if raw_data:
                # Handle both AIMessage and dict
                if hasattr(raw_data, "content"):  # AIMessage case
                    raw_result = raw_data.content.strip()
                elif isinstance(raw_data, dict) and "result" in raw_data:
                    raw_result = raw_data["result"].strip()
                else:
                    raise ProjectException("Unexpected response type", sys)

                # Remove ```json fences if present
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



