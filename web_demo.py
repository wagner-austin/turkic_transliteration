"""
A simple web interface to demonstrate the Turkish transliteration.
"""
import gradio as gr
from turkic_translit.core import to_latin, to_ipa
import unicodedata as ud

def enable_submit(text):
    return bool(text)

def transliterate(text, lang, include_arabic, output_format):
    if not text:
        return "", ""
    try:
        if output_format == "Latin":
            result = to_latin(text, lang, include_arabic=include_arabic)
            format_label = "Latin"
        else:
            result = to_ipa(text, lang)
            format_label = "IPA"
        result = ud.normalize("NFC", result)
        stats_md = (f"**Bytes** — Cyrillic : {len(text.encode('utf8'))}, "
                    f"{format_label} : {len(result.encode('utf8'))}")
        return result, stats_md
    except Exception as e:
        raise gr.Error(str(e))

# Create the Gradio interface
with gr.Blocks(title="Turkic Transliteration Demo") as demo:
    gr.Markdown("# Turkic Transliteration Demo")
    gr.Markdown("Enter Cyrillic text for Kazakh (kk) or Kyrgyz (ky) and see the Latin transliteration")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text (Cyrillic)",
                placeholder="Enter Kazakh or Kyrgyz text in Cyrillic script...",
                lines=5
            )
            lang = gr.Radio(
                ["kk", "ky"], 
                label="Language", 
                info="kk = Kazakh, ky = Kyrgyz",
                value="kk"
            )
            output_format = gr.Radio(
                ["Latin", "IPA"],
                label="Output Format",
                info="Latin = Standard Latin alphabet, IPA = International Phonetic Alphabet",
                value="Latin"
            )
            include_arabic = gr.Checkbox(False, label="Also transliterate Arabic script (Latin mode only)")
            submit_btn = gr.Button("Transliterate", variant="primary", interactive=False)
            input_text.change(
                fn=enable_submit,
                inputs=input_text,
                outputs=submit_btn
            )
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Transliteration Output",
                lines=5,
                interactive=False
            )
            stats = gr.Markdown(value="")
    
    # Example inputs
    examples = [
        ["Қазақ тілі - Түркі тілдерінің бірі.", "kk", "Latin"],
        ["Қазақ тілі - Түркі тілдерінің бірі.", "kk", "IPA"],
        ["Кыргыз тили - Түрк тилдеринин бири.", "ky", "Latin"],
        ["Кыргыз тили - Түрк тилдеринин бири.", "ky", "IPA"]
    ]
    gr.Examples(examples, [input_text, lang, output_format])
    
    # Set up the event
    submit_btn.click(
        fn=transliterate,
        inputs=[input_text, lang, include_arabic, output_format],
        outputs=[output_text, stats]
    )
    
    # Also update on input change for real-time feedback
    input_text.change(
        fn=transliterate,
        inputs=[input_text, lang, include_arabic, output_format],
        outputs=[output_text, stats]
    )
    
    lang.change(
        fn=transliterate,
        inputs=[input_text, lang, include_arabic, output_format],
        outputs=[output_text, stats]
    )
    
    include_arabic.change(
        fn=transliterate,
        inputs=[input_text, lang, include_arabic, output_format],
        outputs=[output_text, stats]
    )
    
    output_format.change(
        fn=transliterate,
        inputs=[input_text, lang, include_arabic, output_format],
        outputs=[output_text, stats]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)
