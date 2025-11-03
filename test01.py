import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from mcq_gen.logger import logging as log
from mcq_gen.exception import ProjectException
from mcq_gen.src.data_ingestion.chat_ingestor import ChatIngestor
from mcq_gen.src.generator.generator import MCQGenRAG
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()


def test_document_ingestion_and_rag():
    try:
        test_files = [
           "E:/Project/MCQ-Generator/data/test_data/nlp.pdf",
        ]

        uploaded_files = []

        for file_path in test_files:
            if Path(file_path).exists():
                uploaded_files.append(open(file_path, "rb"))
            else:
                print(f"File does not exist: {file_path}")

        if not uploaded_files:
            print("No valid files to upload.")
            sys.exit(1)

        # Build index using single-module ChatIngestor
        ci = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)
        
        # Using MMR (Maximal Marginal Relevance) for diverse results
        # MMR parameters:
        # - fetch_k: Number of documents to fetch before MMR re-ranking (20)
        # - lambda_mult: Diversity parameter (0=max diversity, 1=max relevance, 0.5=balanced)
        retriever = ci.build_retriever(
            uploaded_files, 
        )
        
        # Alternative: Use similarity search instead of MMR
        # retriever = ci.build_retriever(uploaded_files, chunk_size=200, chunk_overlap=20, k=5, search_type="similarity")

        # Close file handles
        for f in uploaded_files:
            try:
                f.close()
            except Exception:
                pass

        session_id = ci.session_id
        index_dir = os.path.join("faiss_index", session_id)

        # Load RAG with MMR search
        rag = MCQGenRAG(session_id=session_id)
        rag.load_retriever_from_faiss(
            index_path=index_dir, 
        )

        answer = rag.genetate()
        

        

        if not uploaded_files:
            print("No valid files to upload.")
            sys.exit(1)

    except Exception as e:
        log.error(f"test fail error={str(e)}")
        sys.exit(1)


test_document_ingestion_and_rag()