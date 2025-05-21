# NumPy 2.x still breaks the C++ fastText code – guard on major only
import numpy as np
import pytest
from packaging.version import parse as vparse

if vparse(np.__version__).major >= 2:
    pytest.skip("fastText requires NumPy<2", allow_module_level=True)

# Absent fasttext on non-Windows is fine – mark xfail
try:
    import os
    import pathlib
    import urllib.request

    import fasttext
except ModuleNotFoundError:
    pytest.xfail("fasttext-wheel not installed on this platform")
model = "C:/Users/%USERNAME%/lid.176.bin".replace("%USERNAME%", os.getlogin())
if not pathlib.Path(model).exists():
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", model
    )
lid = fasttext.load_model(model)
print(lid.predict("Пример", k=1))
