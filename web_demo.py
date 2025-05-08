"""
A simple web interface to demonstrate the Turkish transliteration.
"""
import gradio as gr
from turkic_translit.core import to_latin
import unicodedata as ud

def enable_submit(text):
    return bool(text)

def transliterate(text, lang, include_arabic):
    if not text:
        return "", ""
    try:
        result = to_latin(text, lang, include_arabic=include_arabic)
        result = ud.normalize("NFC", result)
        stats_md = (f"**Bytes** — Cyrillic : {len(text.encode('utf8'))}, "
                    f"Latin : {len(result.encode('utf8'))}")
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
            include_arabic = gr.Checkbox(False, label="Also transliterate Arabic script")
            submit_btn = gr.Button("Transliterate", variant="primary", interactive=False)
            input_text.change(
                fn=enable_submit,
                inputs=input_text,
                outputs=submit_btn
            )
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Latin Transliteration",
                lines=5,
                interactive=False
            )
            stats = gr.Markdown(value="")
    
    # Example inputs
    examples = [
        ["Қазақ тілі - Түркі тілдерінің бірі.", "kk"],
        ["Кыргыз тили - Түрк тилдеринин бири.", "ky"]
    ]
    gr.Examples(examples, [input_text, lang])
    
    # Set up the event
    submit_btn.click(
        fn=transliterate,
        inputs=[input_text, lang, include_arabic],
        outputs=[output_text, stats]
    )
    
    # Also update on input change for real-time feedback
    input_text.change(
        fn=transliterate,
        inputs=[input_text, lang, include_arabic],
        outputs=[output_text, stats]
    )
    
    lang.change(
        fn=transliterate,
        inputs=[input_text, lang, include_arabic],
        outputs=[output_text, stats]
    )
    
    include_arabic.change(
        fn=transliterate,
        inputs=[input_text, lang, include_arabic],
        outputs=[output_text, stats]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)
