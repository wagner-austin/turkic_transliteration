# imported automatically if on PYTHONPATH
import os; os.environ.setdefault("PYTHONUTF8", "1")

print("sitecustomize.py loaded, PYTHONUTF8 =", os.environ.get("PYTHONUTF8"))