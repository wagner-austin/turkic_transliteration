# Turkic Transliteration Demo for Hugging Face Spaces
# This application demonstrates transliteration between Cyrillic, Latin, and IPA for Turkic languages

# Import the web demo UI builder from the turkic_translit package
# Using the new app module structure
from turkic_translit.web.web_demo import build_ui

# Create the Gradio interface with default settings
# The build_ui function configures a Gradio Interface with:
# - Input fields for text entry
# - Language selection (Kazakh/Kyrgyz)
# - Script selection (Cyrillic/Latin/IPA)
# - Real-time transliteration preview
demo = build_ui()

# Enable queuing for better performance with multiple users
# This prevents the server from being overwhelmed by concurrent requests
demo.queue()

# Launch the web application
# In Hugging Face Spaces, this will make the app available to users
demo.launch()