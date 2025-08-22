# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

import os
import subprocess
import platform
from gtts import gTTS
import elevenlabs
from elevenlabs.client import ElevenLabs

ELEVENLABS_API_KEY = os.environ.get("ELEVEN_API_KEY")

def text_to_speech_with_gtts_old(input_text, output_filepath):
    language = "en"
    audioobj = gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)

def text_to_speech_with_elevenlabs_old(input_text, output_filepath):
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text=input_text,
        voice="Aria",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)

def play_audio_file(filepath):
    """
    Improved audio playback function for different operating systems
    """
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', filepath], check=True)
        elif os_name == "Windows":  # Windows
            # Use Windows Media Player instead of SoundPlayer for MP3 support
            subprocess.run([
                'powershell', '-c', 
                f'Start-Process -FilePath "wmplayer.exe" -ArgumentList "{filepath}" -WindowStyle Hidden'
            ], check=True)
        elif os_name == "Linux":  # Linux
            # Try multiple players
            for player in ['mpg123', 'ffplay', 'aplay']:
                try:
                    subprocess.run([player, filepath], check=True)
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            else:
                raise OSError("No suitable audio player found")
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")
        print(f"Audio file saved to: {filepath}")

def text_to_speech_with_gtts(input_text, output_filepath):
    language = "en"
    audioobj = gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)
    play_audio_file(output_filepath)
    return output_filepath

def text_to_speech_with_elevenlabs(input_text, output_filepath):
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text=input_text,
        voice="Aria",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)
    play_audio_file(output_filepath)
    return output_filepath

# Alternative function without auto-play (for Gradio)
def text_to_speech_with_elevenlabs_no_play(input_text, output_filepath):
    """
    Generate speech without auto-playing (better for web interfaces)
    """
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text=input_text,
        voice="Aria",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)
    return output_filepath