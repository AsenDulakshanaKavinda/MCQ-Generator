import os
from mcq_gen.utils.config_loader import load_config, _project_root, test
from mcq_gen.logger import GLOBAL_LOGGER as log
from mcq_gen.exception.custom_exception import DocumentPortalException
docs = [1,2,3,4]

log.info("Documents loaded", count=len(docs))
raise DocumentPortalException(ZeroDivisionError)

