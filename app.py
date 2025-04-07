import streamlit as st
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import fitz # pdf
import io
import torch
import speech_recognition as sr     
from pydub import AudioSegment
from gtts import gTTS
import tempfile
import base64
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from lib import *

# Streamlit
st.set_page_config(page_title="Chroma AI", layout="centered")
st.title("📚 Chroma AI - Inclusive Learning Assistant")

st.markdown("## 📂 Upload")
input_type = st.radio("Select Input Type", ["Upload PDF", "Upload Image", "Upload Audio", "Paste Text"])
input_text = ""

if input_type == "Upload PDF":
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file:
        input_text = extract_text_from_pdf(pdf_file)

elif input_type == "Upload Image":
    image_file = st.file_uploader("Upload an Image file", type=["jpg", "jpeg", "png"])
    if image_file:
        image = Image.open(image_file)
        input_text = extract_text_from_image(image)

elif input_type == "Upload Audio":
    audio_file = st.file_uploader("Upload an Audio file", type=["wav", "mp3", "m4a"])
    if audio_file:
        input_text = transcribe_audio(audio_file)

elif input_type == "Paste Text":
    input_text = st.text_area("Paste your text here")

if input_text:
    st.markdown("## 🧠 Choose What You Need")
    read_out = st.button("🔊 Read Out Loud")
    simplify = st.button("📝 Simplify Notes")
    generate_visual = st.button("🖼 Generate Visual Aid (Placeholder)")

    st.markdown("---")
    st.markdown("## 📄 Output")
    st.markdown("### 📃 Extracted / Transcribed Text")
    st.write(input_text)

    if simplify:
        simple = summarize_text(input_text)
        st.markdown("### 📝 Simplified Version")
        st.write(simple)
        audio_path = convert_text_to_speech(simple)
        st.markdown("### 🔊 TTS of Simplified Version")
        st.audio(audio_path)

    if read_out:
        audio_path = convert_text_to_speech(input_text)
        st.markdown("### 🔊 Text-to-Speech Output")
        st.audio(audio_path)


    if generate_visual:
        st.markdown("🖼 Visual Aid Generator is under development!")

st.markdown("---")


# Preferences
st.markdown("## ⚙️ Preferences")

font_size = st.selectbox("Font Size", ["Small", "Medium", "Large"])

# force font
font_size_map = {
    "Small": "14px",
    "Medium": "18px",
    "Large": "24px"
}

st.markdown(
    f"""
    <style>
    body, p, div {{
        font-size: {font_size_map[font_size]} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
st.markdown("*Chroma AI: Learn in your way.*")
