# This file was moved from the project root to examples/
# Usage: pip install turkic-transliterate[examples] to ensure dependencies

import streamlit as st
from turkic_translit.pipeline import TurkicTransliterationPipeline

st.title("Turkic Transliteration Demo")
pipeline = TurkicTransliterationPipeline()
text = st.text_area("Enter text:", "сәлем әлем!")
if st.button("Transliterate"):
    st.write(pipeline.process(text))
