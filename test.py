import sys
from mcq_gen.exception import ProjectException
from mcq_gen.utils.model_loader import ApiKeyManager, ModelLoader
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
    


# test_log_exception()
# test_model_loader()