import streamlit as st
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import tempfile
from gtts import gTTS
from pydub import AudioSegment
from transformers import pipeline
#import openai
import re
from lib import *

# Initialize models
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Chroma AI", layout="wide")
st.title("📚 Chroma AI - Inclusive Learning Assistant")

st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["Upload & Process", "Summarizer", "Accessibility Tools", "Visual Aids"])

if tab == "Upload & Process":
    st.header("Upload & Process Files")
    input_type = st.radio("Select Input Type", ["Upload PDF", "Upload Image", "Upload Audio", "Paste Text"])
    input_text = ""

    if input_type == "Upload PDF":
        pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if pdf_file:
            input_text = extract_text_from_pdf(pdf_file)

    elif input_type == "Upload Image":
        image_file = st.file_uploader("Upload an Image file", type=["jpg", "jpeg", "png"])
        if image_file:
            input_text = extract_text_from_image(image_file)

    elif input_type == "Upload Audio":
        audio_file = st.file_uploader("Upload an Audio file", type=["wav", "mp3", "m4a"])
        if audio_file:
            input_text = convert_audio_to_text(audio_file)

    elif input_type == "Paste Text":
        input_text = st.text_area("Paste your text here")

    if input_text:
        st.markdown("## 🧠 Choose What You Need")
        read_out = st.button("🔊 Read Out Loud")
        simplify = st.button("📝 Simplify Notes")

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

elif tab == "Summarizer":
    st.header("Summarize Content")
    user_text = st.text_area("Paste content to summarize:")
    if st.button("Summarize") and user_text:
        summary = summarize_text(user_text)
        st.success("Summary:")
        st.write(summary)
        if st.button("Play Summary"):
            audio_path = convert_text_to_speech(summary)
            st.audio(audio_path)

elif tab == "Accessibility Tools":
    st.header("Accessibility Tools")

    st.subheader("1. Dyslexia-Friendly Mode")
    d_text = st.text_area("Enter text for dyslexia mode:")
    if st.button("Enable Dyslexia Mode") and d_text:
        styled_html = dyslexia_mode(d_text)
        st.markdown(styled_html, unsafe_allow_html=True)

elif tab == "Visual Aids":
    st.header("Smart Visual Aid Generator")
    concept = st.text_area("Enter concept explanation:")
    api_key = st.text_input("OpenAI API Key", type="password")
    if st.button("Generate Diagram") and concept and api_key:
        diagram = generate_mermaid_diagram(concept, api_key)
        st.markdown(f"""```mermaid\n{diagram}\n```""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("*Chroma AI: Learn in your way.*")
