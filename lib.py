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
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summary = summarizer(text, max_length=100, min_length=50, do_sample=False)
    return summary[0]['summary_text']

print(summarize_text("Your transcription text here"))

# Initialize models7
recognizer = sr.Recognizer()


def convert_text_to_speech(text, filename="output/audio.wav"):
    temp_mp3 = "output/temp.mp3"
    tts = gTTS(text)
    tts.save(temp_mp3)

    audio = AudioSegment.from_mp3(temp_mp3)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
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
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError:
        return "API request failed"

