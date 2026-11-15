---
license: other
license_name: cc-by-sa-3.0-and-cc-by-nc-4.0
tags:
  - fasttext
  - language-identification
library_name: fasttext
---

# Language-identification weights for turkic-translit

This repository holds unmodified copies of two fastText
language-identification models published by Meta AI. It exists so that
[turkic-translit](https://github.com/wagner-austin/turkic_transliteration)
and its [demo Space](https://huggingface.co/spaces/AustinWagner/turkic-transliteration-demo)
download them from Hugging Face rather than from Meta's CDN. The files
here are copied by the repository's `mirror-lid-models` workflow, which
refuses to upload a file unless its SHA-256 is the one the package pins.
The package checks the same digest on every download, so these are
byte for byte the files Meta released.

| File | Origin | License |
|---|---|---|
| `lid.176.bin` | <https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin> | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| `lid218e.bin` | <https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin> | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

The pinned digests are in `src/turkic_translit/lid/registry.py` of the
package repository.

## Attribution

`lid.176.bin` is fastText's 176-language identification model, by
Facebook AI Research. Cite:

- Armand Joulin, Edouard Grave, Piotr Bojanowski and Tomas Mikolov,
  *Bag of Tricks for Efficient Text Classification*, 2016,
  arXiv:1607.01759.
- Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, Hervé
  Jégou and Tomas Mikolov, *FastText.zip: Compressing text
  classification models*, 2016, arXiv:1612.03651.

`lid218e.bin` is the 218-language identification model released with No
Language Left Behind, by Meta AI. Cite:

- NLLB Team, *No Language Left Behind: Scaling Human-Centered Machine
  Translation*, 2022, arXiv:2207.04672.

`lid218e.bin` is licensed for non-commercial use only.
