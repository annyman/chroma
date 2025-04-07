import streamlit as st
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import io
import torch
#import whisper
from transformers import pipeline
from gtts import gTTS
import tempfile
import base64
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# Initialize models
#model_whisper = whisper.load_model("base")
#summarizer = pipeline("summarization")

def convert_text_to_speech(text, filename="output/audio.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    return filename

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_file_path = tmp_file.name
    #result = model_whisper.transcribe(tmp_file_path)
    #return result["text"]


def summarize_text(text, sentence_count=3):
    """
    Summarizes text using Sumy's LexRank algorithm.
    
    :param text: The input text to summarize.
    :param sentence_count: The number of sentences to return in the summary.
    :return: Summarized text as a string.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)


# Streamlit Layout
st.set_page_config(page_title="Chroma AI", layout="centered")
st.title("📚 Chroma AI - Inclusive Learning Assistant")
st.markdown("Helping visually, hearing, and cognitively challenged students access learning easily.")

st.markdown("---")
st.markdown("## 🧑‍🏫 Choose Your Accessibility Need")
col1, col2, col3 = st.columns(3)
with col1:
    visual = st.button("👁 I have trouble seeing")
with col2:
    hearing = st.button("👂 I can't hear audio")
with col3:
    cognitive = st.button("🧠 I need simplified content")

st.markdown("## 📂 Upload or Record Your Input")
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

st.markdown("---")

if input_text:
    st.markdown("## 🧠 Choose What You Need")
    read_out = st.button("🔊 Read Out Loud")
    simplify = st.button("📝 Simplify Notes")
    generate_visual = st.button("🖼 Generate Visual Aid (Placeholder)")

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


# Preferences Section
st.markdown("## ⚙️ Preferences")

# Store user selections
font_size = st.selectbox("Font Size", ["Small", "Medium", "Large"])

# Apply Font Size
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
