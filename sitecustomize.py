# imported automatically if on PYTHONPATH
import os; os.environ.setdefault("PYTHONUTF8", "1")
import logging

# Log at debug level instead of printing to stdout
logging.getLogger("sitecustomize").debug("PYTHONUTF8=%s", os.environ["PYTHONUTF8"])