import streamlit as st
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import fitz  # PyMuPDF
import io
import torch
import speech_recognition as sr  # Replacing whisper
from pydub import AudioSegment
from gtts import gTTS
import tempfile
import base64
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# Initialize models
recognizer = sr.Recognizer()


def convert_text_to_speech(text, filename="output/audio.wav"):
    # Step 1: Generate TTS as MP3
    temp_mp3 = "output/temp.mp3"
    tts = gTTS(text)
    tts.save(temp_mp3)

    # Step 2: Convert MP3 to WAV with specified format
    audio = AudioSegment.from_mp3(temp_mp3)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # PCM 16-bit (s16le), Mono, 16kHz
    audio.export(filename, format="wav")

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
    
    with sr.AudioFile(tmp_file_path) as source:
        audio_data = recognizer.record(source)
    
    try:
        return recognizer.recognize_google(audio_data)  # Google's free Web Speech API
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError:
        return "API request failed"


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
