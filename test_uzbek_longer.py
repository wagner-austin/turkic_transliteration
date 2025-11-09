import os
import sys

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"

import fasttext

model = fasttext.load_model("src/turkic_translit/lid218e.bin")

# Longer Uzbek Cyrillic text
text = "Ўзбекистон Республикаси Марказий Осиёда жойлашган мустақил давлатдир. Ўзбекистон халқи ўз тилида сўзлашади ва ўз маданиятига эга. Тошкент Ўзбекистоннинг пойтахти ва энг катта шаҳридир."

print("Testing longer Uzbek Cyrillic text:")
print(f"Text length: {len(text)} characters\n")

result = model.predict(text.replace("\n", " "), k=5)
print("Top 5 predictions:")
for i, (label, prob) in enumerate(zip(result[0], result[1])):
    clean_label = label.replace("__label__", "")
    print(f"  {i + 1}. {clean_label:15s} {prob:.4f}")
