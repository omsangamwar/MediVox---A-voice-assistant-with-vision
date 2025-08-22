# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Simplified function to record audio from the microphone and save it as an MP3 file.

    Args:
    file_path (str): Path to save the recorded audio file.
    timeout (int): Maximum time to wait for a phrase to start (in seconds).
    phrase_time_limit (int): Maximum time for the phrase to be recorded (in seconds).
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            
            # Record the audio
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            # Convert the recorded audio to an MP3 file
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            
            logging.info(f"Audio saved to {file_path}")
            return file_path

    except sr.WaitTimeoutError:
        logging.error("No speech detected within timeout period")
        return None
    except Exception as e:
        logging.error(f"An error occurred during recording: {e}")
        return None

def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
    """
    Transcribe audio file using Groq's Whisper model
    """
    if not audio_filepath or not os.path.exists(audio_filepath):
        raise ValueError("Audio file not found or invalid path")
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )

        if not transcription.text or transcription.text.strip() == "":
            return "No speech detected in the audio"
            
        return transcription.text
        
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        raise Exception(f"Failed to transcribe audio: {str(e)}")

# Test function
def test_recording():
    audio_filepath = "test_recording.mp3"
    print("Testing audio recording...")
    result = record_audio(file_path=audio_filepath, timeout=10)
    if result:
        print(f"Recording successful: {result}")
        return result
    else:
        print("Recording failed")
        return None

if __name__ == "__main__":
    # Uncomment to test
    # test_recording()
    pass