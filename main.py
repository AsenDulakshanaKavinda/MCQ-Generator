from mcq_gen.src.data_ingestion.chat_ingestor import ChatIngestor


def test_chat_ingestor():
    files = [
    open("E:/Project/MCQ-Generator/data/test_data/AGM4367-Unit I-Session 1.pdf", "rb"),
    open("E:/Project/MCQ-Generator/data/test_data/web@dev.txt", "rb")
    ]
    ci = ChatIngestor()
    ci.build_retriever(files)


test_chat_ingestor()