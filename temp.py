import streamlit as st
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import tempfile
from gtts import gTTS
from pydub import AudioSegment
from transformers import pipeline
import cv2
from fer import FER
import openai
import re

# Initialize models
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ========== Helper Functions ==========

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

def convert_audio_to_text(audio_file):
    import whisper
    model = whisper.load_model("base")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name
    result = model.transcribe(temp_audio_path)
    return result["text"]

def summarize_text(text, sentence_count=3):
    summarizer = pipeline("summarization")
    return summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

def convert_text_to_speech(text, filename="output/audio.wav"):
    temp_mp3 = "output/temp.mp3"
    tts = gTTS(text)
    tts.save(temp_mp3)
    audio = AudioSegment.from_mp3(temp_mp3)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(filename, format="wav")
    return filename

def dyslexia_mode(text, font_size='18px'):
    style = f"""
    <style>
    .dyslexia-text {{
        font-family: 'OpenDyslexic', sans-serif;
        font-size: {font_size};
        line-height: 1.6;
        letter-spacing: 0.06em;
    }}
    @import url('https://fonts.googleapis.com/css2?family=OpenDyslexic');
    </style>
    """
    return f"{style}<div class='dyslexia-text'>{text}</div>"

def detect_emotion_from_webcam():
    cap = cv2.VideoCapture(0)
    detector = FER(mtcnn=True)
    emotion_result = ""
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            emotions = detector.detect_emotions(frame)
            if emotions:
                emotion_result = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
        cap.release()
    return emotion_result

def generate_mermaid_diagram(prompt, api_key):
    openai.api_key = api_key
    diagram_prompt = f"""
    You are a helpful assistant that converts simple explanations into a Mermaid diagram. 
    Only return the mermaid code block.

    Explanation: {prompt}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": diagram_prompt}]
    )
    mermaid_code = re.findall(r'```mermaid\n(.*?)```', response['choices'][0]['message']['content'], re.DOTALL)
    return mermaid_code[0] if mermaid_code else "Diagram generation failed."

# ========== Streamlit UI ==========

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

    st.subheader("2. Facial Emotion Detection")
    if st.button("Check My Expression"):
        emotion = detect_emotion_from_webcam()
        st.success(f"Detected Emotion: {emotion}" if emotion else "No face detected.")

elif tab == "Visual Aids":
    st.header("Smart Visual Aid Generator")
    concept = st.text_area("Enter concept explanation:")
    api_key = st.text_input("OpenAI API Key", type="password")
    if st.button("Generate Diagram") and concept and api_key:
        diagram = generate_mermaid_diagram(concept, api_key)
        st.markdown(f"""```mermaid\n{diagram}\n```""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("*Chroma AI: Learn in your way.*")
