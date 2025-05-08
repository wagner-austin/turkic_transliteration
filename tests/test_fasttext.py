import fasttext, os, urllib.request, pathlib, sys
model = "C:/Users/%USERNAME%/lid.176.bin".replace("%USERNAME%", os.getlogin())
if not pathlib.Path(model).exists():
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", model
    )
lid = fasttext.load_model(model)
print(lid.predict("Пример", k=1))
