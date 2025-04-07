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
import openai

def summarize_text(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

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

