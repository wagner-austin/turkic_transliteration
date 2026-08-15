---
title: Turkic Transliteration Demo
emoji: 🌖
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 6.17.3
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Deterministic IPA for 7 Turkic languages + Finnish
---

# Turkic Transliteration Demo

## Overview
This Hugging Face Space demonstrates deterministic broad-IPA transcription for the Turkic language family and a Uralic control language (Finnish). Each language's rule set is derived from a peer-reviewed phonological description of the language, so the output is auditable against the primary literature rather than a black-box model.

## Supported languages
- **Kazakh** (`kk`) — Cyrillic input, IPA output (McCollum & Chen 2021)
- **Kyrgyz** (`ky`) — Cyrillic input, IPA output (McCollum 2020)
- **Turkish** (`tr`) — Latin input, IPA output (Zimmer & Orgun 1992)
- **Azerbaijani** (`az`) — Latin input, IPA output (Ghaffarvand Mokari & Werner 2017)
- **Uzbek** (`uz`) — Latin or Cyrillic input, IPA output (Ido 2025)
- **Uyghur** (`ug`) — Arabic-script input, IPA output (McCollum 2021)
- **Finnish** (`fi`) — Latin input, IPA output (Suomi, Toivanen & Ylitalo 2008)

Kazakh and Kyrgyz additionally support Latin transliteration (Presidential Decree No. 569, 2017; Common Turkic Alphabet, 1991).

## Features
- **IPA Transcription** for the full language set above
- **Latin Transliteration** for Kazakh and Kyrgyz
- **Corpus Download tab**: stream text from OSCAR-2301, Wikipedia dumps, and Leipzig-corpus mirrors with optional FastText language-ID filtering
- **Interactive UI** powered by Gradio; no installation required

## Usage
1. Choose a tab: **Transliterate to IPA** for interactive text conversion, or **Download Corpus** for batch data ingestion
2. Select the language from the dropdown
3. Enter text (or upload a `.txt` file) and view the transliterated output
4. Download the result as a UTF-8 text file when it is more than a couple of lines long

## Technical Details
This demo is powered by the `turkic-translit` Python package, which wraps ICU's UTS #35 LDML transform engine (PyICU) with per-language rule files grounded in the phonological literature. The same package also exposes command-line tools (`turkic-translit`, `turkic-download-corpus`, `turkic-filter-russian`) and a Python API for programmatic use.

This Space is a thin wrapper — `app.py` is ~20 lines and holds no transliteration logic. Everything above comes from the package, pinned exactly in `requirements.txt`. The pin is exact rather than a floor because Hugging Face reuses a cached wheel when it is given a `>=` range.

Nothing here is edited by hand. This card, `app.py` and `requirements.txt` are all written from [`wagner-austin/turkic_transliteration`](https://github.com/wagner-austin/turkic_transliteration) — the card from `.github/hf-space/README.md`, the pin from the version in `pyproject.toml` — so a rule is fixed by pushing there and nowhere else.

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Kazakh to IPA Rule set from:
Mccollum, Adam & Chen, Si. (2018). Illustration of the IPA: Kazakh. 
https://www.researchgate.net/publication/328290116_Illustration_of_the_IPA_Kazakh#fullTextFileContent
