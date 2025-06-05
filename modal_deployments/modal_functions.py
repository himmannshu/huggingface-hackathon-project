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

# Define the Modal image with Stable Diffusion for thumbnail generation
diffusion_image = modal.Image.debian_slim().pip_install(
    "diffusers>=0.24.0",
    "torch>=2.0.0",
    "torchvision",
    "transformers>=4.25.0",
    "accelerate>=0.20.0",
    "Pillow>=9.0.0",
    "numpy",
    "requests"
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

@app.function(
    image=diffusion_image,
    gpu="A10G",  # High-performance GPU for Stable Diffusion
    memory=16384,  # 16GB RAM for model loading
    timeout=600,  # 10 minutes timeout
    max_containers=2  # Limit concurrent thumbnail generations
)
def generate_thumbnails(
    video_summary: str,
    optimized_title: str,
    optimized_description: str,
    search_terms: list,
    competitive_analysis: dict = None
) -> list:
    """
    Step 7: Generate viral-optimized thumbnails using Stable Diffusion
    
    Args:
        video_summary: Summary from content analysis
        optimized_title: Optimized title from enhancement agent
        optimized_description: Optimized description from enhancement agent  
        search_terms: Search terms from content analysis
        competitive_analysis: Research data from YouTube analysis
        
    Returns:
        list: List of thumbnail image data (base64 encoded)
    """
    
    print(f"🎨 Starting thumbnail generation for: {optimized_title[:50]}...")
    
    import torch
    from diffusers import DiffusionPipeline
    import io
    import base64
    from PIL import Image
    import re
    
    # Load Stable Diffusion XL model
    print("🔄 Loading Stable Diffusion XL model...")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe = pipe.to("cuda")
    
    # Enable memory optimization
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()
    print("✅ Stable Diffusion XL model loaded successfully!")
    
    def extract_key_concepts(text: str) -> list:
        """Extract key visual concepts from text content"""
        
        # Technology/product terms that make good visual elements
        tech_keywords = [
            "AI", "machine learning", "technology", "software", "coding", "programming",
            "data", "analytics", "cloud", "mobile", "web", "app", "digital",
            "MacBook", "iPhone", "computer", "laptop", "smartphone", "gadget",
            "tutorial", "guide", "tips", "review", "comparison", "vs"
        ]
        
        # Action/engagement words for dynamic thumbnails
        action_keywords = [
            "learn", "build", "create", "master", "discover", "unlock", "reveal",
            "secret", "hack", "trick", "method", "strategy", "solution", "fix"
        ]
        
        # Extract concepts from the text
        concepts = []
        text_lower = text.lower()
        
        # Find tech/product concepts
        for keyword in tech_keywords:
            if keyword.lower() in text_lower:
                concepts.append(keyword)
        
        # Find action concepts  
        for keyword in action_keywords:
            if keyword.lower() in text_lower:
                concepts.append(keyword)
        
        # Extract quoted terms and capitalize words that might be products/concepts
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        concepts.extend(words[:3])  # Add up to 3 capitalized terms
        
        return list(set(concepts))[:8]  # Return unique concepts, max 8
    
    def generate_thumbnail_prompt(style: str, title: str, summary: str, concepts: list) -> str:
        """Generate optimized prompts for viral YouTube thumbnails"""
        
        # Base style definitions for different thumbnail approaches
        style_templates = {
            "dynamic": {
                "base": "high-energy YouTube thumbnail, dynamic composition, bright vibrant colors, ",
                "elements": "dramatic lighting, action-oriented, bold text overlay space, ",
                "quality": "professional photography style, studio lighting, 8K resolution, trending on YouTube"
            },
            "tech_modern": {
                "base": "modern tech YouTube thumbnail, sleek minimalist design, gradient backgrounds, ",
                "elements": "glowing tech elements, clean typography space, modern UI design, ",
                "quality": "professional digital art, high contrast, sharp details, premium look"
            },
            "educational": {
                "base": "educational YouTube thumbnail, friendly approachable style, warm colors, ",
                "elements": "clear focal point, teaching elements, infographic style, organized layout, ",
                "quality": "professional illustration, clean design, high readability, engaging"
            }
        }
        
        style_config = style_templates.get(style, style_templates["dynamic"])
        
        # Extract main subject from title and concepts
        main_subject = "content creator"
        if concepts:
            # Prioritize tech/product terms for main subject
            tech_terms = [c for c in concepts if c.lower() in ["ai", "macbook", "iphone", "computer", "laptop", "technology"]]
            if tech_terms:
                main_subject = tech_terms[0]
            else:
                main_subject = concepts[0] if concepts else "content creator"
        
        # Build the prompt
        prompt_parts = [
            style_config["base"],
            f"featuring {main_subject}, ",
            style_config["elements"],
            f"theme: {' '.join(concepts[:3]) if concepts else 'technology'}, ",
            "YouTube thumbnail format, 16:9 aspect ratio, ",
            "compelling visual hook, clickbait style, ",
            style_config["quality"]
        ]
        
        # Add specific elements based on content
        if "tutorial" in title.lower() or "how to" in title.lower():
            prompt_parts.insert(-1, "tutorial elements, step-by-step visual, ")
        
        if "review" in title.lower() or "vs" in title.lower():
            prompt_parts.insert(-1, "comparison elements, product showcase, ")
        
        if any(word in title.lower() for word in ["secret", "hidden", "trick", "hack"]):
            prompt_parts.insert(-1, "mysterious elements, revealed content hint, ")
        
        full_prompt = "".join(prompt_parts)
        
        # Negative prompt to avoid unwanted elements
        negative_prompt = (
            "text, watermarks, logos, signatures, blurry, low quality, "
            "oversaturated, distorted faces, multiple people, crowded composition, "
            "dark lighting, unclear subject, messy background"
        )
        
        return full_prompt, negative_prompt
    
    # Extract key concepts from all available content
    all_content = f"{optimized_title} {video_summary} {' '.join(search_terms)}"
    key_concepts = extract_key_concepts(all_content)
    
    print(f"🎯 Key concepts for thumbnails: {key_concepts}")
    
    # Generate 3 different thumbnail styles
    thumbnail_styles = ["dynamic", "tech_modern", "educational"]
    thumbnails = []
    
    for i, style in enumerate(thumbnail_styles):
        print(f"🎨 Generating thumbnail {i+1}/3 - {style} style...")
        
        try:
            # Generate prompt for this style
            prompt, negative_prompt = generate_thumbnail_prompt(
                style, optimized_title, video_summary, key_concepts
            )
            
            print(f"📝 Prompt: {prompt[:100]}...")
            
            # Generate image
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=720,  # YouTube thumbnail height
                width=1280,  # YouTube thumbnail width (16:9 ratio)
                num_inference_steps=30,  # Good quality vs speed balance
                guidance_scale=7.5,  # Creative but controlled generation
                generator=torch.Generator(device="cuda").manual_seed(42 + i)  # Consistent but different seeds
            ).images[0]
            
            # Convert to base64 for return
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", quality=95)
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            thumbnails.append({
                "style": style,
                "image_data": image_b64,
                "prompt_used": prompt[:200] + "...",  # Store truncated prompt for debugging
                "concepts": key_concepts
            })
            
            print(f"✅ Thumbnail {i+1}/3 generated successfully!")
            
        except Exception as e:
            print(f"❌ Error generating thumbnail {i+1}: {e}")
            # Add placeholder for failed generation
            thumbnails.append({
                "style": style,
                "image_data": None,
                "error": str(e),
                "concepts": key_concepts
            })
    
    print(f"🎉 Thumbnail generation complete! Generated {len([t for t in thumbnails if t.get('image_data')])} successful thumbnails")
    
    return thumbnails

