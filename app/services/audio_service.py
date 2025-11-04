"""
Audio Service for speech-to-text and text-to-speech functionality.
"""
import os
import tempfile
import uuid
from typing import Optional, Dict, Any
from groq import Groq
from gtts import gTTS
from app.config import Config

class AudioService:
    """Service for handling audio recording, speech-to-text, and text-to-speech."""
    
    def __init__(self):
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        self.audio_storage_path = "static/audio"  # Directory to store audio files
        
        # Create audio storage directory if it doesn't exist
        os.makedirs(self.audio_storage_path, exist_ok=True)
    
    def record_audio(self, duration: int = 10, filename: Optional[str] = None) -> str:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            filename: Optional custom filename
            
        Returns:
            Path to the recorded audio file
        """
        try:
            import sounddevice as sd
            import soundfile as sf
            
            if not filename:
                filename = f"recording_{uuid.uuid4().hex[:8]}.wav"
            
            filepath = os.path.join(self.audio_storage_path, filename)
            
            print("🎙️ Recording... Speak now...")
            fs = 16000  # Sample rate
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()  # Wait until recording is finished
            sf.write(filepath, recording, fs)
            print("✅ Recording completed.")
            
            return filepath
            
        except ImportError:
            raise Exception("sounddevice and soundfile packages are required for audio recording")
        except Exception as e:
            raise Exception(f"Audio recording failed: {str(e)}")
    
    def speech_to_text(self, audio_path: str) -> str:
        """
        Convert speech to text using Groq's Whisper API.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            with open(audio_path, "rb") as file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(audio_path, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json"
                )
            return transcription.text
            
        except Exception as e:
            raise Exception(f"Speech-to-text conversion failed: {str(e)}")
    
    def text_to_speech(self, text: str, filename: Optional[str] = None, lang: str = 'en') -> str:
        """
        Convert text to speech and save as audio file.
        
        Args:
            text: Text to convert to speech
            filename: Optional custom filename
            lang: Language code (default: 'en')
            
        Returns:
            Path to the generated audio file
        """
        try:
            if not filename:
                filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
            
            filepath = os.path.join(self.audio_storage_path, filename)
            
            # Generate speech using gTTS
            tts = gTTS(text=text, lang=lang)
            tts.save(filepath)
            
            print("🔊 Text-to-speech generated.")
            return filepath
            
        except Exception as e:
            raise Exception(f"Text-to-speech conversion failed: {str(e)}")
    
    def play_audio(self, audio_path: str) -> bool:
        """
        Play an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.name == 'nt':  # Windows
                os.system(f"start {audio_path}")
            else:  # macOS and Linux
                os.system(f"mpg123 {audio_path}")
            
            print("🔊 Playing audio...")
            return True
            
        except Exception as e:
            print(f"Failed to play audio: {str(e)}")
            return False
    
    def speak_text(self, text: str, play_immediately: bool = True) -> str:
        """
        Convert text to speech and optionally play it immediately.
        
        Args:
            text: Text to speak
            play_immediately: Whether to play the audio immediately
            
        Returns:
            Path to the generated audio file
        """
        audio_path = self.text_to_speech(text)
        
        if play_immediately:
            self.play_audio(audio_path)
        
        return audio_path
    
    def cleanup_audio_file(self, audio_path: str) -> bool:
        """
        Delete an audio file to free up storage.
        
        Args:
            audio_path: Path to the audio file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"🗑️ Deleted audio file: {audio_path}")
                return True
            return False
            
        except Exception as e:
            print(f"Failed to delete audio file: {str(e)}")
            return False
    
    def get_audio_duration(self, audio_path: str) -> Optional[float]:
        """
        Get the duration of an audio file in seconds.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Duration in seconds, or None if unable to determine
        """
        try:
            import soundfile as sf
            data, sample_rate = sf.read(audio_path)
            duration = len(data) / sample_rate
            return duration
            
        except Exception as e:
            print(f"Failed to get audio duration: {str(e)}")
            return None
    
    def validate_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Validate an audio file and return metadata.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with validation results and metadata
        """
        result = {
            'valid': False,
            'duration': None,
            'size': None,
            'format': None,
            'error': None
        }
        
        try:
            if not os.path.exists(audio_path):
                result['error'] = 'File does not exist'
                return result
            
            # Get file size
            result['size'] = os.path.getsize(audio_path)
            
            # Get file format from extension
            result['format'] = os.path.splitext(audio_path)[1].lower()
            
            # Get duration
            result['duration'] = self.get_audio_duration(audio_path)
            
            # Validate format
            valid_formats = ['.wav', '.mp3', '.m4a', '.flac']
            if result['format'] not in valid_formats:
                result['error'] = f'Unsupported audio format: {result["format"]}'
                return result
            
            # Validate size (max 25MB for Groq API)
            max_size = 25 * 1024 * 1024  # 25MB
            if result['size'] > max_size:
                result['error'] = f'File too large: {result["size"]} bytes (max: {max_size})'
                return result
            
            result['valid'] = True
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result
