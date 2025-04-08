import streamlit as st
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import fitz  # PyMuPDF
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
import openai
import os
import re
import numpy as np
from bark import SAMPLE_RATE, generate_audio
from bark.generation import preload_models
import soundfile as sf

# Preload Bark models
preload_models()

# ========== Helper Functions ==========

def summarize_text(text, sentence_count=3):
    """
    Summarizes the given text into a specified number of sentences.

    Args:
        text (str): The text to summarize.
        sentence_count (int): Number of sentences in the summary.

    Returns:
        str: The summarized text.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file.

    Args:
        file: The uploaded PDF file.

    Returns:
        str: The extracted text.
    """
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text

def extract_text_from_image(image_file):
    """
    Extracts text from an image file using OCR.

    Args:
        image_file: The uploaded image file.

    Returns:
        str: The extracted text.
    """
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

def convert_audio_to_text(audio_file):
    """
    Converts audio to text using Whisper.

    Args:
        audio_file: The uploaded audio file.

    Returns:
        str: The transcribed text.
    """
    import whisper
    model = whisper.load_model("base")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name
    result = model.transcribe(temp_audio_path)
    return result["text"]

def convert_text_to_speech(text):
    """
    Converts text to speech using Bark.

    Args:
        text (str): The text to convert.

    Returns:
        str: Path to the generated audio file.
    """
    preload_models()
    audio_array = generate_audio(text)
    temp_path = os.path.join(tempfile.gettempdir(), "bark_output.wav")
    AudioSegment(
        audio_array.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1
    ).export(temp_path, format="wav")
    return temp_path

def dyslexia_mode(text, font_size='18px'):
    """
    Applies a dyslexia-friendly style to the given text.

    Args:
        text (str): The text to style.
        font_size (str): The font size for the styled text.

    Returns:
        str: HTML string with the styled text.
    """
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

def generate_mermaid_diagram(prompt, api_key):
    """
    Generates a Mermaid diagram based on a given prompt.

    Args:
        prompt (str): The explanation to convert into a diagram.
        api_key (str): OpenAI API key.

    Returns:
        str: Mermaid diagram code.
    """
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

