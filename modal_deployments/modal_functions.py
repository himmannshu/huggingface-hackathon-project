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
        competitive_analysis: Research data from YouTube analysis (currently unused in prompt gen)
        
    Returns:
        list: List of thumbnail image data (base64 encoded)
    """
    
    print("--- generate_thumbnails INPUTS ---")
    print(f"Optimized Title: {optimized_title}")
    print(f"Video Summary (first 100 chars): {video_summary[:100] if video_summary else 'N/A'}...")
    print(f"Optimized Description (first 100 chars): {optimized_description[:100] if optimized_description else 'N/A'}...")
    print(f"Search Terms: {search_terms}")
    print(f"Competitive Analysis (keys): {list(competitive_analysis.keys()) if competitive_analysis else 'N/A'}")
    print("------------------------------------")
    
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
    
    def extract_key_concepts(title: str, summary: str, search_terms: list) -> list:
        """Extract key visual concepts from video content for thumbnail generation."""
        
        # Expanded visual keywords (mix of tech, action, objects, concepts)
        visual_keywords = [
            "AI", "robot", "code", "data", "chart", "graph", "future", "brain", "learn", "build", "create", "discover",
            "secret", "hack", "tutorial", "guide", "tips", "review", "comparison", "vs", "new", "update",
            "technology", "software", "programming", "analytics", "cloud", "mobile", "web", "app", "digital",
            "MacBook", "iPhone", "computer", "laptop", "smartphone", "gadget", "device", "screen",
            "money", "success", "growth", "business", "marketing", "idea", "lightbulb",
            "gaming", "game", "virtual reality", "VR", "metaverse",
            "person", "man", "woman", "developer", "creator", "youtuber", "student", "teacher",
            "problem", "solution", "challenge", "journey", "story", "magic", "power"
        ]
        # Lowercase version for efficient checking
        lower_visual_keywords = [kw.lower() for kw in visual_keywords]
        
        concepts = []
        # Combine all text sources for keyword spotting
        full_content_text = f"{title.lower()} {summary.lower()} {' '.join(search_terms).lower()}"

        for keyword in visual_keywords: # Iterate original to preserve casing for concepts list
            if keyword.lower() in full_content_text:
                concepts.append(keyword)
        
        # Extract capitalized phrases from title and summary as potential specific subjects/products
        title_caps = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b', title)
        summary_caps = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b', summary)
        
        for cap_phrase in title_caps + summary_caps:
            # Check if it's not already added, not a generic keyword itself, and has some length
            if cap_phrase not in concepts and cap_phrase.lower() not in lower_visual_keywords and len(cap_phrase) > 2:
                concepts.append(cap_phrase)

        # Fallback: add some terms from title if concepts list is still small and title is available
        if not concepts and title:
            # A simple list of common English stop words
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", 
                          "have", "has", "had", "do", "does", "did", "will", "would", "should", 
                          "can", "could", "may", "might", "must", "and", "but", "or", "nor", 
                          "for", "so", "yet", "in", "on", "at", "by", "from", "to", "with", "about",
                          "of", "how", "what", "when", "where", "why", "which", "my", "your", "its"}
            title_words = [word for word in title.split() if len(word) > 3 and word.lower() not in stop_words]
            concepts.extend(title_words[:3]) # Add first few meaningful words from title

        # Remove duplicates while preserving order and limit
        unique_concepts_ordered = []
        seen_concepts = set()
        for concept in concepts:
            if concept.lower() not in seen_concepts:
                unique_concepts_ordered.append(concept)
                seen_concepts.add(concept.lower())
        return unique_concepts_ordered[:10]

    def generate_thumbnail_prompt(
        style: str, 
        video_title_for_overlay: str, 
        video_summary_for_cue: str, 
        concepts: list,
        original_video_title_for_context: str
    ) -> tuple[str, str]:
        """Generate optimized prompts for viral YouTube thumbnails with text overlays."""
        
        style_templates = {
            "dynamic": {
                "base": "High-energy YouTube thumbnail, dynamic composition, vibrant contrasting colors, modern graphic elements, ",
                "elements": "action lines, particle effects, glowing highlights, prominent space for bold text overlay, ",
                "quality": "professional digital art, studio quality, 8K resolution, trending on ArtStation, eye-catching"
            },
            "tech_modern": {
                "base": "Sleek modern tech YouTube thumbnail, minimalist and futuristic design, gradient background with subtle abstract patterns, ",
                "elements": "holographic projections, glowing circuits or icons, clean typography area for text overlay, UI elements, ",
                "quality": "high-tech digital illustration, sharp focus, cinematic lighting, premium and sophisticated feel"
            },
            "educational": {
                "base": "Engaging educational YouTube thumbnail, clear and inviting visual style, bright and warm color palette, ",
                "elements": "iconography representing key concepts, simplified diagrams, organized layout for clarity, visual metaphor for learning, space for informative text overlay, ",
                "quality": "professional illustration, high readability, friendly and approachable, conveys information effectively"
            },
            "cinematic_story": {
                "base": "Dramatic cinematic storytelling YouTube thumbnail, rich color grading, epic lighting, movie poster aesthetic, ",
                "elements": "compelling central figure or iconic object, evocative scene hinting at a narrative, sense of depth and scale, high emotional impact, area for stylized text overlay, ",
                "quality": "photorealistic rendering, epic composition, highly detailed, blockbuster movie visual quality, 8K"
            }
        }
        
        style_config = style_templates.get(style, style_templates["dynamic"])
        
        main_subject = "a key visual element related to the video" # Default
        if concepts:
            # Prefer non-generic, more specific concepts as main subject
            potential_subjects = [c for c in concepts if c.lower() not in ["tutorial", "guide", "review", "vs", "new", "update", "tips", "secret", "hack"]]
             # Prioritize capitalized concepts if available, or longer ones
            capitalized_subjects = [s for s in potential_subjects if any(char.isupper() for char in s)]
            if capitalized_subjects:
                 main_subject = capitalized_subjects[0]
            elif potential_subjects:
                main_subject = sorted(potential_subjects, key=len, reverse=True)[0] # Pick longest potential subject
            else:
                main_subject = concepts[0] # Fallback to first concept
        elif original_video_title_for_context:
            stop_words_for_title = {"a", "the", "is", "of", "for", "how", "to", "and", "with", "my", "your", "in", "on", "what", "why"}
            title_words = [w for w in original_video_title_for_context.split() if w.lower() not in stop_words_for_title]
            if title_words:
                main_subject_candidate = ' '.join(title_words[:3]) # take first few significant words
                if len(main_subject_candidate) > 2: 
                    main_subject = f'"{main_subject_candidate}"'


        # Prepare text for overlay from video_title_for_overlay
        overlay_text_words = video_title_for_overlay.split()
        overlay_text = video_title_for_overlay # Default to full title
        if len(overlay_text_words) > 6: # If title is long, try to pick a catchy part
            catchy_keywords = ["new", "secret", "best", "top", "easy", "fast", "ultimate", "guide", "tutorial", "hack"]
            catchy_part = ""
            for i, word in enumerate(overlay_text_words):
                if word.lower() in catchy_keywords or (word.isdigit() and i > 0):
                    # Try to get a phrase around the keyword/number
                    start_index = max(0, i - 1 if overlay_text_words[i-1].lower() != "a" else i - 2) # Avoid starting with "a"
                    end_index = min(len(overlay_text_words), i + 2)
                    phrase = " ".join(overlay_text_words[start_index:end_index])
                    if len(phrase.split()) <= 4 : # Keep it short
                        catchy_part = phrase
                        break
            if catchy_part:
                overlay_text = catchy_part
            else: # Fallback to first 3-4 words if no catchy part found or title very long
                overlay_text = " ".join(overlay_text_words[:4])
        
        overlay_text = overlay_text.strip().replace('"', '').replace("'", "")[:60] # Clean up, limit length

        # Visual cue from video_summary_for_cue
        summary_cue = ""
        if video_summary_for_cue:
            first_sentence = video_summary_for_cue.split('.')[0].strip()
            if 10 < len(first_sentence) < 150 : # Ensure it's a meaningful, concise sentence part
                 summary_cue = f"Visually hinting at content like: '{first_sentence[:80]}...'. "
        
        prompt_parts = [
            f"Design an ultra-detailed, highly engaging YouTube thumbnail for a video titled \"{original_video_title_for_context}\". ",
            style_config["base"],
            f"The main visual focus must be {main_subject}. ",
            summary_cue,
            f"Subtly incorporate visual themes or objects related to: {(', '.join(concepts[:3]) if concepts else 'the video topic')}. ",
            style_config["elements"],
            "The thumbnail MUST be 16:9 aspect ratio. It needs to be designed to maximize click-through rates using a powerful visual hook. ",
            f"Artistically integrate the text \"{overlay_text}\" into the thumbnail design. The text must be very prominent, extremely clear, easily readable, and stylishly composed with the overall visual aesthetics. Use bold, impactful fonts and strong contrasting colors for the text to ensure it stands out. ",
            style_config["quality"]
        ]
        
        title_lower = original_video_title_for_context.lower()
        if "tutorial" in title_lower or "how to" in title_lower or "guide" in title_lower:
            prompt_parts.append("Visually convey a sense of learning, step-by-step instruction, or a practical guide. ")
        if "review" in title_lower or "vs" in title_lower or "comparison" in title_lower:
            prompt_parts.append("Clearly showcase product(s) or elements being compared. ")
        if any(word in title_lower for word in ["secret", "hidden", "reveal", "exposed", "hack", "trick", "unlock"]):
            prompt_parts.append("Create an aura of mystery, discovery, or a surprising revelation in the visuals. ")

        full_prompt = "".join(prompt_parts)
        
        negative_prompt = (
            "text that is too small, unreadable text, poorly rendered text, illegible text, distorted text, text with bad kerning, text not integrated well, "
            "watermarks, author signatures, channel logos (unless explicitly part of the main subject described), "
            "blurry, low resolution, noisy, grainy, pixelated, artifacts, jpeg compression, "
            "oversaturated, undersaturated, flat lighting, dark or muddy visuals, unclear subject, "
            "messy background, tiling patterns, out of frame elements, cut-off subjects, "
            "boring, generic, ugly, amateurish, poorly composed, too much empty space, "
            "extra limbs, missing limbs, mutated hands, bad hands, malformed fingers, poorly drawn hands, poorly drawn faces, distorted anatomy, "
            "multiple panels, collage style, screenshot, low effort"
        )
        
        return full_prompt, negative_prompt

    # Extract key concepts from all available content
    print(f"🔄 Extracting key concepts for: {optimized_title[:50]}...")
    key_concepts = extract_key_concepts(
        title=optimized_title, 
        summary=video_summary, 
        search_terms=search_terms
    )
    
    print(f"🎯 Key concepts for thumbnails: {key_concepts}")
    
    # Generate 4 different thumbnail styles
    thumbnail_styles = ["dynamic", "tech_modern", "educational", "cinematic_story"]
    thumbnails = []
    
    for i, style in enumerate(thumbnail_styles):
        print(f"🎨 Generating thumbnail {i+1}/{len(thumbnail_styles)} - Style: {style}...")
        
        try:
            # Generate prompt for this style
            prompt, negative_prompt = generate_thumbnail_prompt(
                style=style,
                video_title_for_overlay=optimized_title, 
                video_summary_for_cue=video_summary,
                concepts=key_concepts,
                original_video_title_for_context=optimized_title
            )
            
            print(f"📝 Full Prompt ({style}): {prompt}") 
            print(f"🚫 Full Negative Prompt: {negative_prompt}")
            
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
            image.save(buffer, format="PNG", quality=95, compress_level=1) 
            
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            thumbnails.append({
                "style": style,
                "image_data": image_b64,
                "prompt_used": prompt, # Store full prompt
                "concepts": key_concepts,
                "negative_prompt_used": negative_prompt # Store negative prompt
            })
            
            print(f"✅ Thumbnail {i+1}/{len(thumbnail_styles)} generated successfully!")
            
        except Exception as e:
            print(f"❌ Error generating thumbnail {i+1}: {e}")
            # Add placeholder for failed generation
            thumbnails.append({
                "style": style,
                "image_data": None,
                "error": str(e),
                "concepts": key_concepts,
                "prompt_that_failed": prompt if 'prompt' in locals() else "Prompt not generated",
                "negative_prompt_that_failed": negative_prompt if 'negative_prompt' in locals() else "Negative prompt not generated"
            })
    
    print(f"🎉 Thumbnail generation complete! Generated {len([t for t in thumbnails if t.get('image_data')])} successful thumbnails")
    
    return thumbnails

