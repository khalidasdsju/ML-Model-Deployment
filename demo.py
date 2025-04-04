
from HF.logger import logging
from HF.exception import HFException
import sys
logging.info("This is my first log")
try:
    a = 1 / 0
except Exception as e:
    raise HFException(e, sys)
