import modal
import os
import logging
import sys
from pathlib import Path
import time

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

# --- Configuration ---
# Attempt to load environment variables, e.g., for WHISPER_MODEL_SIZE if needed by a Modal function
# from dotenv import load_dotenv
# load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env') # Adjust path to .env if it's in project root

# Name of the Modal app as defined in modal_functions.py
MODAL_APP_NAME = "youtube-content-optimizer"

# Placeholder for sample file paths
# These files would ideally be small fixtures committed to the repository for testing.
# For now, we'll use placeholders and check if they exist.
SAMPLE_VIDEO_PATH = Path(__file__).parent / "fixtures" / "sample_video_10s.mp4" # Example path
SAMPLE_AUDIO_PATH = Path(__file__).parent / "fixtures" / "sample_audio.wav"   # Example path

# Ensure fixtures directory exists for clarity, though sample files might not
FIXTURES_DIR = Path(__file__).parent / "fixtures"
# os.makedirs(FIXTURES_DIR, exist_ok=True) # Not creating dir, just for path construction

def check_modal_availability():
    """Checks if Modal functions seem to be accessible."""
    try:
        # Try to "ping" or get a handle to one of the functions without calling it.
        # This doesn't guarantee the app is running correctly, but it's a basic check.
        modal.Function.lookup(MODAL_APP_NAME, "process_video_to_audio")
        logger.info("✅ Modal seems available. Attempting to look up functions.")
        return True
    except modal.exception.NotFoundError:
        logger.error(
            "❌ Modal app '%s' or its functions not found. "
            "Ensure the Modal app is deployed by running `modal deploy modal_deployments/modal_functions.py` "
            "from the project root.", MODAL_APP_NAME
        )
        return False
    except Exception as e:
        logger.error("An error occurred while checking Modal availability: %s", e, exc_info=True)
        return False


def test_process_video_to_audio_modal():
    logger.info("\n--- Testing Modal function: process_video_to_audio ---")
    if not SAMPLE_VIDEO_PATH.is_file():
        logger.warning(
            "⚠️ Sample video file not found at %s. Skipping test_process_video_to_audio_modal. "
            "Please add a small sample video (e.g., sample_video_10s.mp4) to the %s directory.",
            SAMPLE_VIDEO_PATH, FIXTURES_DIR
        )
        return None, None # Return None for audio_bytes and filename

    try:
        process_fn = modal.Function.from_name(MODAL_APP_NAME, "process_video_to_audio")
        logger.info("Attempting to call process_video_to_audio.remote()...")

        with open(SAMPLE_VIDEO_PATH, "rb") as f:
            video_bytes = f.read()

        start_time = time.time()
        audio_bytes, audio_filename = process_fn.remote(video_bytes, SAMPLE_VIDEO_PATH.name)
        end_time = time.time()

        logger.info(f"✅ process_video_to_audio completed in {end_time - start_time:.2f} seconds.")
        if audio_bytes and audio_filename:
            logger.info(f"Returned audio filename: {audio_filename}")
            logger.info(f"Returned audio bytes length: {len(audio_bytes)}")
            # Optionally save the audio to inspect it
            # with open(FIXTURES_DIR / audio_filename, "wb") as f_audio:
            #     f_audio.write(audio_bytes)
            # logger.info(f"Sample audio saved to {FIXTURES_DIR / audio_filename}")
            return audio_bytes, audio_filename
        else:
            logger.error("❌ process_video_to_audio returned None or empty data.")
            return None, None

    except Exception as e:
        logger.error(f"❌ Error testing process_video_to_audio: {e}", exc_info=True)
        return None, None

def test_transcribe_audio_modal(audio_bytes, audio_filename):
    logger.info("\n--- Testing Modal function: transcribe_audio_with_whisper ---")
    if not audio_bytes or not audio_filename:
        logger.warning("⚠️ No audio data provided from video processing test. Skipping transcribe_audio_modal.")
        # As a fallback, try to load a sample audio if video processing failed
        if SAMPLE_AUDIO_PATH.is_file():
            logger.info("Attempting to load fallback sample audio from %s", SAMPLE_AUDIO_PATH)
            with open(SAMPLE_AUDIO_PATH, "rb") as f:
                audio_bytes = f.read()
            audio_filename = SAMPLE_AUDIO_PATH.name
        else:
            logger.warning(
                "⚠️ Fallback sample audio file not found at %s. Skipping test_transcribe_audio_modal. "
                "Please add a small sample audio (e.g., sample_audio.wav) to the %s directory.",
                SAMPLE_AUDIO_PATH, FIXTURES_DIR
            )
            return

    try:
        transcribe_fn = modal.Function.from_name(MODAL_APP_NAME, "transcribe_audio_with_whisper")
        # Use a small model for faster testing; this can be configured via env var in actual app
        whisper_model_to_test = os.getenv("WHISPER_MODEL_SIZE", "tiny") # Default to tiny for test
        logger.info(f"Attempting to call transcribe_audio_with_whisper.remote() using model: {whisper_model_to_test}...")

        start_time = time.time()
        transcription_data = transcribe_fn.remote(audio_bytes, audio_filename, whisper_model_size=whisper_model_to_test)
        end_time = time.time()

        logger.info(f"✅ transcribe_audio_with_whisper completed in {end_time - start_time:.2f} seconds.")
        if transcription_data:
            logger.info(f"Transcription language: {transcription_data.get('language')}")
            logger.info(f"Transcription text snippet: '{transcription_data.get('text', '')[:100]}...'")
            # logger.info(f"Full transcription data: {transcription_data}") # Can be verbose
        else:
            logger.error("❌ transcribe_audio_with_whisper returned None or empty data.")

    except Exception as e:
        logger.error(f"❌ Error testing transcribe_audio_with_whisper: {e}", exc_info=True)

def test_generate_thumbnails_modal():
    logger.info("\n--- Testing Modal function: generate_thumbnails ---")
    try:
        generate_fn = modal.Function.from_name(MODAL_APP_NAME, "generate_thumbnails")

        sample_summary = "This is a fantastic video about coding amazing things with Python and AI. We explore new frontiers in technology."
        sample_title = "AI Python Project: Building the Future!"
        sample_description = "Join us as we build the future with AI and Python. This tutorial covers everything from basic setup to advanced deployment."
        sample_search_terms = ["AI programming", "Python projects", "future technology"]
        sample_competitive_analysis = { # Mocked competitive analysis data
            "competitive_analysis": {
                "top_performing_videos": [
                    {"title": "Old AI Project - Still good!", "channel": "Competitor1", "views": 100000, "engagement_rate": 5.0},
                    {"title": "Amazing Python Hacks", "channel": "Competitor2", "views": 200000, "engagement_rate": 4.5}
                ],
                "common_title_words": ["AI", "Python", "Tutorial"]
            }
        }

        logger.info("Attempting to call generate_thumbnails.remote()...")
        start_time = time.time()
        thumbnail_results = generate_fn.remote(
            video_summary=sample_summary,
            optimized_title=sample_title,
            optimized_description=sample_description,
            search_terms=sample_search_terms,
            competitive_analysis=sample_competitive_analysis
        )
        end_time = time.time()

        logger.info(f"✅ generate_thumbnails completed in {end_time - start_time:.2f} seconds.")
        if thumbnail_results:
            logger.info(f"Returned {len(thumbnail_results)} thumbnail results.")
            for i, result in enumerate(thumbnail_results):
                if result.get("image_data"):
                    logger.info(f"Thumbnail {i+1}: Style '{result.get('style')}', Image data length: {len(result.get('image_data'))}, Concepts: {result.get('concepts')}")
                else:
                    logger.error(f"Thumbnail {i+1}: Style '{result.get('style')}' failed with error: {result.get('error')}")
        else:
            logger.error("❌ generate_thumbnails returned None or empty data.")

    except Exception as e:
        logger.error(f"❌ Error testing generate_thumbnails: {e}", exc_info=True)


if __name__ == '__main__':
    logger.info("🚀 Starting Modal function integration tests...")

    # Create fixtures directory if it doesn't exist - for user convenience if they add samples
    # However, the script itself will not create the sample files.
    if not FIXTURES_DIR.exists():
        logger.info(f"Creating fixtures directory at: {FIXTURES_DIR}")
        os.makedirs(FIXTURES_DIR, exist_ok=True)
        logger.info(f"Please consider adding a short 'sample_video_10s.mp4' (e.g., 5-10 seconds) and a 'sample_audio.wav' to the '{FIXTURES_DIR}' directory to enable all tests.")


    if not check_modal_availability():
        logger.warning("Modal is not available or app not deployed. Tests will likely fail or not run completely.")
        sys.exit(1) # Exit if Modal is not available

    # Test process_video_to_audio
    # This function's output is used by the transcribe test
    audio_bytes_result, audio_filename_result = test_process_video_to_audio_modal()

    # Test transcribe_audio_with_whisper
    # It will use the output from the previous test, or try to load a sample if that failed.
    test_transcribe_audio_modal(audio_bytes_result, audio_filename_result)

    # Test generate_thumbnails
    test_generate_thumbnails_modal()

    logger.info("\n🏁 Modal function integration tests finished.")
    logger.info("ℹ️ Note: These tests call live Modal functions and may incur costs or use quotas.")
    logger.info("Ensure your Modal app '%s' is deployed and running.", MODAL_APP_NAME)
