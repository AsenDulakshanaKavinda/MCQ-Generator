import sys
from pathlib import Path
from mcq_gen.exception import ProjectException
from mcq_gen.utils.model_loader import ApiKeyManager, ModelLoader
from mcq_gen.utils.document_ops import load_documents
from mcq_gen.utils.file_io import save_uploaded_files
# 
def test_log_exception():
    try:
        result = 10/0
    except Exception as e:
        ProjectException(error_message=str(e), error_details=sys)

def test_model_loader():
    model_loader = ModelLoader()
    model_loader.load_embeddings()
    model_loader.load_llm()
    
def test_load_documents():
    test_files = [Path("E:/Project/MCQ-Generator/test_data/test.txt")]
    load_documents(test_files)

def test_save_uploaded_files():
    test_file = Path("E:/Project/MCQ-Generator/test_data/test.txt")
    target_dir = Path("E:/Project/MCQ-Generator/data")
    with open(test_file, "rb") as f:
        uploaded_files = [f]

        saved = save_uploaded_files(uploaded_files, target_dir)




# test_log_exception()
# test_model_loader()
# test_load_documents()
test_save_uploaded_files()