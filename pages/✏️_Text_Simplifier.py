import streamlit as st
from gtts import gTTS
import fitz  # PyMuPDF
import os

st.set_page_config(page_title="Simplify Text & Audio", layout="wide")
st.title("✏️ Simplify Text & Audio")

# Directory to save outputs
os.makedirs("output", exist_ok=True)

def read_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return text

def generate_audio(text, filename="output/audio.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    return filename

input_text = st.text_area("Paste your paragraph here")
pdf_file = st.file_uploader("Or upload a PDF", type=["pdf"])

if pdf_file:
    input_text = read_pdf(pdf_file)
    st.text_area("Extracted Text", input_text, height=200)

if st.button("Simplify Text") and input_text:
    simplified = "WIP"  # Placeholder for simplification logic
    st.text_area("Simplified Text", simplified, height=200)

    if st.button("🔊 Convert Simplified Text to Audio"):
        audio_path = generate_audio(simplified, filename="output/simplified.mp3")
        st.audio(audio_path)
