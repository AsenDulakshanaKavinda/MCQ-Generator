import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from mcq_gen.src.data_ingestion.chat_ingestor import ChatIngestor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import RetrievalQA
from langchain_mistralai import ChatMistralAI

load_dotenv()


def test_document_ingestion_and_rag():

    test_files = ["E:/Project/MCQ-Generator/test_data/nlp.pdf",]

    uploaded_files = []

    for file_path in test_files:
        if Path(file_path).exists():
            uploaded_files.append(open(file_path, "rb"))
        else:
            print(f"File dose not exist: {file_path}")

    if not uploaded_files:
        print("No valid files to upload.")
        sys.exit(1)

    # Build index using single-module ChatIngestor
    ci = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)


    # 1. Build your retriever (from your ChatIngestor)
    retriever = ci.build_retriever(uploaded_files, k=5)

    # 2. Initialize your LLM
    llm = ChatMistralAI(model ="mistral-large-latest", temperature=0)

    # 3. Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",         # combines all retrieved docs into one prompt
        retriever=retriever,
        return_source_documents=True  # optional, gives you which docs were used
    )

    # 4. Run the chain with a query
    query = "give me 5 mcq questions from the data"
    result = qa_chain.invoke(query)

    # 5. Access results
    answer = result['result']              # generated answer
    sources = result['source_documents']   # documents retrieved (if needed)

    print("Answer:", answer)
    for doc in sources:
        print("Source:", doc.page_content[:200], "...")


test_document_ingestion_and_rag()
