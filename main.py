import sys
from mcq_gen.exception.custom_exception import ProjectException

try:
    10/0
except ZeroDivisionError as e:
    ProjectException(e, sys)

