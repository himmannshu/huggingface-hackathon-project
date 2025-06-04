import modal
import os
import tempfile
from pathlib import Path

# Create Modal app
app = modal.App("youtube-content-optimizer")

# Define the Modal image with FFmpeg
image = modal.Image.debian_slim().apt_install(
    "ffmpeg"
).pip_install(
    "requests"
)

@app.function(
    image=image,
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

@app.function(
    image=image,
    timeout=60
)
def test_modal_connection() -> str:
    """Simple test function to verify Modal is working"""
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
        result = test_modal_connection.remote()
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