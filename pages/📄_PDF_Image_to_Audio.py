import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
from gtts import gTTS
import os

st.set_page_config(page_title="PDF/Image to Audio", layout="wide")
st.title("📄 PDF/Image to Audio")

# Directory to save outputs
os.makedirs("output", exist_ok=True)

def read_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return text

def read_image(uploaded_file):
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    return text

def generate_audio(text, filename="output/audio.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    return filename

uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = read_pdf(uploaded_file)
    else:
        text = read_image(uploaded_file)

    st.text_area("Extracted Text", text, height=200)

    if st.button("🔊 Convert to Speech"):
        audio_path = generate_audio(text)
        st.audio(audio_path)
