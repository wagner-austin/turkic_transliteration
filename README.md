turkic\_transliterate
Deterministic Latin and IPA transliteration for Kazakh and Kyrgyz, plus helper utilities for tokenizer training and Russian-token filtering.

Quick install

1. Install Miniconda or Anaconda (recommended).
2. Clone the repo and create the environment:
   conda env create -f env.yml
3. Activate the environment:
   conda activate turkic
4. Run the verification tests:
   python -m pytest      (all tests should pass)

Python compatibility
• Works on CPython 3.10 and 3.11.
• CPython 3.12+ is supported everywhere except on Windows until official PyICU wheels are available; see “Windows & PyICU” below.

Package names
• Runtime import path:  turkic\_translit
• Distributable name on PyPI:  turkic\_transliterate
• Command-line entry point:  turkic-translit

Installing with pip
pip install -e .\[dev,ui]        # add ,winlid on Windows if you need fasttext-wheel

Optional extras
dev   → black, ruff, pytest
ui    → gradio web demo
winlid (Windows only) → fasttext-wheel for language ID

Windows & PyICU
For Python 3.12+ on Windows you must install a pre-built PyICU wheel:
python scripts/get\_pyicu\_wheel.py
The helper script downloads and installs the correct wheel from Christoph Gohlke’s repository.

Command-line usage
turkic-translit --lang kk --in text.txt --out\_latin kk\_lat.txt --ipa --out\_ipa kk\_ipa.txt --arabic --log-level debug
• --lang            kk or ky
• --ipa             emit IPA alongside Latin
• --arabic          also transliterate embedded Arabic script
• --benchmark       print throughput statistics
• --log-level       debug | info | warning | error | critical (default: info)

Logging
The central logging setup uses Rich for colour when available.
Set TURKIC\_LOG\_LEVEL or pass --log-level to the CLI.
Fallback to standard logging when Rich is absent.

Web demo
python web\_demo.py
Opens a local Gradio interface for real-time transliteration.

Tokenizer training example
python scripts/build\_spm.py --input corpora/kk\_lat.txt,corpora/ky\_lat.txt --model\_prefix spm/turkic12k --vocab\_size 12000

Filtering Russian tokens from Uzbek
cat uz\_raw\.txt | python scripts/filter\_russian.py --mode drop > uz\_clean.txt

Developer checklist
black .
ruff check .
pytest -q

All code is UTF-8-only; on Windows a BOM is written when piping to files to avoid encoding issues.

License
Apache-2.0
