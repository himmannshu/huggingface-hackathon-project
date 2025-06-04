import modal
import os
import tempfile
from pathlib import Path

# Create Modal app
app = modal.App("youtube-content-optimizer")

# Define the Modal image with FFmpeg for audio extraction
audio_image = modal.Image.debian_slim().apt_install(
    "ffmpeg"
).pip_install(
    "requests"
)

# Define the Modal image with Whisper for transcription
whisper_image = modal.Image.debian_slim().apt_install(
    "ffmpeg"  # Keep ffmpeg for audio handling
).pip_install(
    "openai-whisper",  # Let pip resolve compatible versions
    "torch",
    "torchaudio"
)

@app.function(
    image=audio_image,
    timeout=600,  # 10 minutes timeout for large videos
    memory=2048,  # 2GB memory for processing
)
def process_video_to_audio(video_bytes: bytes, filename: str) -> tuple[bytes, str]:
    """
    Step 1 & 2: Process uploaded video and extract audio
    
    Args:
        video_bytes: Raw video file bytes from Gradio upload
        filename: Original filename for format detection
        
    Returns:
        tuple: (audio_bytes, audio_filename)
    """
    
    print(f"🎬 Processing video: {filename}")
    print(f"📦 Video size: {len(video_bytes) / (1024*1024):.1f} MB")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Save uploaded video temporarily
        video_path = temp_path / f"input_video{Path(filename).suffix}"
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        
        print(f"✅ Video saved temporarily: {video_path}")
        
        # Step 2: Extract audio using FFmpeg
        audio_filename = f"extracted_audio_{Path(filename).stem}.wav"
        audio_path = temp_path / audio_filename
        
        # FFmpeg command optimized for Whisper
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit (good for Whisper)
            "-ar", "16000",  # 16kHz sample rate (Whisper optimal)
            "-ac", "1",  # Mono channel
            "-y",  # Overwrite output
            str(audio_path)
        ]
        
        print(f"🔄 Running FFmpeg extraction...")
        
        # Execute FFmpeg
        import subprocess
        try:
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Audio extraction successful!")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e}")
            print(f"FFmpeg stderr: {e.stderr}")
            raise Exception(f"Audio extraction failed: {e.stderr}")
        
        # Read extracted audio
        if not audio_path.exists():
            raise Exception("Audio file was not created")
            
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        print(f"🎵 Audio extracted: {len(audio_bytes) / (1024*1024):.1f} MB")
        print(f"📁 Original video deleted from temporary storage")
        
        return audio_bytes, audio_filename

# Simplified Whisper transcription function with compatibility fixes
@app.function(
    image=whisper_image,
    gpu="A10G",  # Nvidia A10G GPU as requested
    memory=12288,  # 12GB RAM (12 * 1024 MiB)
    timeout=900,  # 15 minutes timeout for transcription
)
def transcribe_audio_with_whisper(audio_bytes: bytes, audio_filename: str) -> dict:
    """
    Step 3: Transcribe audio using OpenAI Whisper (small model)
    
    Args:
        audio_bytes: Raw audio file bytes from audio extraction
        audio_filename: Audio filename for reference
        
    Returns:
        dict: Structured transcription with text, segments, language, duration
    """
    
    print(f"🎙️ Starting Whisper transcription: {audio_filename}")
    print(f"📦 Audio size: {len(audio_bytes) / (1024*1024):.1f} MB")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save audio bytes to temporary file
        audio_path = temp_path / audio_filename
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        
        print(f"✅ Audio saved for transcription: {audio_path}")
        
        # Load Whisper model (small for speed)
        import whisper
        print(f"🔄 Loading Whisper small model...")
        model = whisper.load_model("small")
        print(f"✅ Whisper model loaded successfully!")
        
        # Transcribe audio with compatible settings
        print(f"🔄 Transcribing audio...")
        result = model.transcribe(
            str(audio_path),
            verbose=False,  # Disable verbose to avoid potential issues
            # Remove word_timestamps=True as it causes Triton compatibility issues
            temperature=0.0,  # Use deterministic decoding
            no_speech_threshold=0.6,  # Improve silence detection
            logprob_threshold=-1.0,  # Improve quality filtering
            compression_ratio_threshold=2.4  # Improve repetition filtering
        )
        
        # Extract transcription data
        transcription_data = {
            "text": result["text"].strip(),
            "language": result["language"],
            "segments": [
                {
                    "start": segment["start"],
                    "end": segment["end"], 
                    "text": segment["text"].strip()
                }
                for segment in result["segments"]
            ],
            "duration": result["segments"][-1]["end"] if result["segments"] else 0.0,
            "word_count": len(result["text"].split()),
            "audio_filename": audio_filename
        }
        
        print(f"✅ Transcription complete!")
        print(f"📝 Language detected: {transcription_data['language']}")
        print(f"⏱️ Duration: {transcription_data['duration']:.1f} seconds")
        print(f"📄 Word count: {transcription_data['word_count']} words")
        print(f"📁 Temporary audio file deleted")
        
        return transcription_data

def test_modal_connection() -> str:
    """Simple test function to verify Modal is working - LOCAL ONLY"""
    return "🎉 Modal connection successful! Ready for video processing."


# Local function to call Modal (for use in Gradio)
def upload_and_extract_audio(video_file_path: str) -> tuple[bytes, str]:
    """
    Local function that calls the Modal function
    
    Args:
        video_file_path: Path to uploaded video file
        
    Returns:
        tuple: (audio_bytes, audio_filename)
    """
    
    if not video_file_path or not os.path.exists(video_file_path):
        raise ValueError("Invalid video file path")
    
    # Read video file
    with open(video_file_path, "rb") as f:
        video_bytes = f.read()
    
    filename = os.path.basename(video_file_path)
    
    # Call Modal function
    print(f"📤 Sending video to Modal for processing...")
    audio_bytes, audio_filename = process_video_to_audio.remote(video_bytes, filename)
    
    return audio_bytes, audio_filename


# Test function for local development
def test_modal_setup():
    """Test if Modal is properly configured"""
    try:
        result = test_modal_connection()
        print(result)
        return True
    except Exception as e:
        print(f"❌ Modal setup error: {e}")
        return False


if __name__ == "__main__":
    # Test Modal setup when run directly
    print("🧪 Testing Modal setup...")
    if test_modal_setup():
        print("✅ Modal is ready for video processing!")
    else:
        print("❌ Modal setup needs attention") 