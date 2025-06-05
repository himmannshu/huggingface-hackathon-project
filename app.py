import gradio as gr
import os
import modal
from llama_index.core import Settings # Added for checking Settings.llm

from PIL import Image, ImageDraw, ImageFont
import base64
import io
from dotenv import load_dotenv
import logging
import sys

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

# --- Global Configuration & Environment Variables ---
load_dotenv() # Load environment variables from .env file

MODAL_AVAILABLE = False
SELECTED_WHISPER_MODEL = os.getenv("WHISPER_MODEL_SIZE", "small").strip().lower()
logger.info("🎙️ Using Whisper model size: %s (configurable via WHISPER_MODEL_SIZE env var)", SELECTED_WHISPER_MODEL)
# --- End Global Configuration ---

# Note: Step 4 is implemented via ContentAnalysisAgent in agents/ directory
# No need for separate src.step4_content_analysis module

try:
    # Import the deployed functions using Modal's Function.from_name method
    process_video_to_audio = modal.Function.from_name("youtube-content-optimizer", "process_video_to_audio")
    transcribe_audio_with_whisper = modal.Function.from_name("youtube-content-optimizer", "transcribe_audio_with_whisper")
    generate_thumbnails = modal.Function.from_name("youtube-content-optimizer", "generate_thumbnails")
    MODAL_AVAILABLE = True
    logger.info("✅ Modal connection established!")
    logger.info("✅ Thumbnail generation function loaded!")
except Exception as e:
    logger.warning("⚠️ Modal not available: %s", e)
    logger.info("🔄 Running in demo mode without Modal processing")
    MODAL_AVAILABLE = False

# Step 4 initialization is handled by ContentAnalysisAgent class

# Attempt to import Step 4 Agent
try:
    from agents.content_analysis_agent import ContentAnalysisAgent
    STEP4_AGENT_AVAILABLE = True
    content_analyzer = None 
    logger.info("✅ Step 4 (ContentAnalysisAgent) class loaded.")
except ImportError as e:
    logger.warning("⚠️ Step 4 Agent (agents.content_analysis_agent) not found or import error: %s. Content analysis will be skipped.", e)
    STEP4_AGENT_AVAILABLE = False
    # Define a dummy class and instance if not available so the app can run
    class ContentAnalysisAgent:
        def __init__(self):
            logger.info("Dummy ContentAnalysisAgent initialized.")
        def analyze_transcript(self, transcript: str):
            logger.info("Dummy ContentAnalysisAgent.analyze_transcript called.")
            return {
                "video_summary": "[Content analysis agent not available]",
                "search_terms": []
            }
    content_analyzer = ContentAnalysisAgent() # Instantiate dummy for safety

# Attempt to import Step 5 Agent (YouTube Research)
try:
    from agents.youtube_research_agent import YouTubeResearchAgent
    STEP5_AGENT_AVAILABLE = True
    youtube_researcher = None
    logger.info("✅ Step 5 (YouTubeResearchAgent) class loaded.")
except ImportError as e:
    logger.warning("⚠️ Step 5 Agent (agents.youtube_research_agent) not found or import error: %s. YouTube research will be skipped.", e)
    STEP5_AGENT_AVAILABLE = False
    youtube_researcher = None

# Attempt to import Step 6 Agent (Content Enhancement)
try:
    from agents.content_enhancement_agent import ContentEnhancementAgent
    STEP6_AGENT_AVAILABLE = True
    content_enhancer = None
    logger.info("✅ Step 6 (ContentEnhancementAgent) class loaded.")
except ImportError as e:
    logger.warning("⚠️ Step 6 Agent (agents.content_enhancement_agent) not found or import error: %s. Content enhancement will be skipped.", e)
    STEP6_AGENT_AVAILABLE = False
    content_enhancer = None

# Initialize agents for local use if Modal is not available
if not MODAL_AVAILABLE:
    # Initialize ContentAnalysisAgent
    if STEP4_AGENT_AVAILABLE:
        try:
            logger.info("🔄 Initializing ContentAnalysisAgent for local Step 4 processing...")
            content_analyzer = ContentAnalysisAgent()
            logger.info("✅ ContentAnalysisAgent initialized for local use.")
        except Exception as e:
            logger.error("⚠️ Error initializing ContentAnalysisAgent for local Step 4: %s. Step 4 might fail.", e, exc_info=True)
            # Fallback to dummy if initialization fails
            if not isinstance(content_analyzer, ContentAnalysisAgent) or content_analyzer is None:
                class ContentAnalysisAgent_Dummy: # This class is already defined above, maybe reuse or ensure scope
                    def __init__(self):
                        logger.info("Fallback Dummy ContentAnalysisAgent initialized after error during local init.")
                    def analyze_transcript(self, transcript: str):
                        return {"video_summary": "[Content analysis agent failed to initialize]", "search_terms": []}
                content_analyzer = ContentAnalysisAgent_Dummy()
    
    # Initialize YouTubeResearchAgent
    if STEP5_AGENT_AVAILABLE:
        try:
            logger.info("🔄 Initializing YouTubeResearchAgent for Step 5 processing...")
            youtube_researcher = YouTubeResearchAgent()
            logger.info("✅ YouTubeResearchAgent initialized.")
        except Exception as e:
            logger.error("⚠️ Error initializing YouTubeResearchAgent: %s. Step 5 will be skipped.", e, exc_info=True)
            youtube_researcher = None
    
    # Initialize ContentEnhancementAgent
    if STEP6_AGENT_AVAILABLE:
        try:
            logger.info("🔄 Initializing ContentEnhancementAgent for Step 6 processing...")
            content_enhancer = ContentEnhancementAgent()
            logger.info("✅ ContentEnhancementAgent initialized.")
        except Exception as e:
            logger.error("⚠️ Error initializing ContentEnhancementAgent: %s. Step 6 will be skipped.", e, exc_info=True)
            content_enhancer = None

def format_youtube_research_results(research_data: dict, search_terms: list) -> str:
    """
    Format YouTube research results for display in the UI.
    
    Args:
        research_data: Research data from YouTubeResearchAgent
        search_terms: Original search terms used
        
    Returns:
        Formatted string for UI display
    """
    if not research_data or not search_terms:
        return "No research data available."
    
    result = f"🔍 Search Terms Used: {', '.join(search_terms)}\n\n"
    
    # Research summary
    summary = research_data.get("research_summary", {})
    result += f"📊 RESEARCH SUMMARY:\n"
    result += f"• Total videos analyzed: {summary.get('total_videos_analyzed', 0)}\n"
    result += f"• Search terms researched: {summary.get('total_search_terms', 0)}\n\n"
    
    # Competitive analysis
    analysis = research_data.get("competitive_analysis", {})
    if analysis:
        result += f"📈 COMPETITIVE INSIGHTS:\n"
        result += f"• Average views: {analysis.get('average_views', 0):,}\n"
        result += f"• Average engagement rate: {analysis.get('average_engagement_rate', 0):.2f}%\n\n"
        
        # Top performing videos
        top_videos = analysis.get("top_performing_videos", [])
        if top_videos:
            result += f"🏆 TOP PERFORMING VIDEOS (used as optimization context):\n"
            for i, video in enumerate(top_videos[:3], 1):
                result += f"{i}. \"{video.get('title', 'N/A')}\" - {video.get('views', 0):,} views ({video.get('engagement_rate', 0):.2f}% engagement)\n"
            result += "\n"
        
        # Common keywords
        keywords = analysis.get("common_title_words", [])
        if keywords:
            result += f"🎯 TRENDING KEYWORDS (incorporated in optimization):\n"
            result += f"• {', '.join(keywords[:8])}\n\n"
    
    # Per-term results
    term_results = research_data.get("term_results", {})
    if term_results:
        result += f"📋 DETAILED RESULTS BY SEARCH TERM:\n"
        for term, data in term_results.items():
            videos = data.get("videos", [])
            result += f"\n🔸 '{term}' - {len(videos)} videos found:\n"
            for i, video in enumerate(videos[:3], 1):
                title = video.get('title', 'N/A')
                views = video.get('view_count', 0)
                channel = video.get('channel_name', 'Unknown')
                result += f"   {i}. {title[:60]}{'...' if len(title) > 60 else ''}\n"
                result += f"      📺 {channel} | 👁 {views:,} views\n"
    
    result += f"\n✨ This data was used to optimize your title and description using AI analysis of trending content!"
    
    return result

def process_video(video_file):
    """
    Main processing function for the video content optimization pipeline.
    Steps 1-3: Video upload, audio extraction, and Whisper transcription via Modal
    Step 4: Content Analysis (Summary & Search Terms) using ContentAnalysisAgent
    Step 5: YouTube Research & Competitive Analysis using YouTubeResearchAgent
    Step 6: Content Enhancement (Optimized Titles & Descriptions) using ContentEnhancementAgent
    """
    global content_analyzer, youtube_researcher, content_enhancer # Access all agents

    if video_file is None:
        return "Please upload a video file.", "", "", "", "", "", None, None, None

    # --- Font Handling ---
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Common on Linux
        "Arial.ttf",  # Common on Windows (may need to be in the same dir or adjust path)
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf", # Common on macOS
        "sans-serif.ttf" # A generic fallback if available
    ]
    font_size = 60 # Initial font size, can be adjusted
    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            logger.info("Successfully loaded font: %s", font_path)
            break
        except IOError:
            logger.debug("Font not found at %s, trying next.", font_path)

    if not font:
        logger.warning("Default fonts not found. Using PIL default font. Text quality may be suboptimal.")
        font = ImageFont.load_default() # Fallback to PIL's default
    # --- End Font Handling ---
    
    try:
        if MODAL_AVAILABLE:
            # Step 1: Video Upload Processing
            status = "🔄 Step 1/7: Processing video... Uploading to Modal for audio extraction"
            logger.info(status)
            
            # Read video file
            with open(video_file, "rb") as f:
                video_bytes = f.read()
            
            filename = os.path.basename(video_file)
            
            logger.info("📤 Sending video to Modal for processing: %s", filename)
            
            # Step 2: Audio Extraction
            audio_bytes, audio_filename = process_video_to_audio.remote(video_bytes, filename)
            
            status = f"✅ Step 2/7: Audio extraction successful! File: {audio_filename} ({len(audio_bytes) / (1024*1024):.1f} MB)\n🔄 Step 3/7: Starting Whisper transcription..."
            
            # Step 3: Whisper Transcription
            logger.info("🎙️ Sending audio to Whisper for transcription (using model: %s)...", SELECTED_WHISPER_MODEL)
            transcription_data = transcribe_audio_with_whisper.remote(
                audio_bytes,
                audio_filename,
                whisper_model_size=SELECTED_WHISPER_MODEL # Pass the selected model size
            )
            
            # Extract transcription details
            transcript_text = transcription_data["text"]
            language = transcription_data["language"]
            duration = transcription_data["duration"]
            word_count = transcription_data["word_count"]
            
            status = f"✅ Step 3/7: Transcription complete! Lang: {language.upper()}, Duration: {duration:.1f}s, Words: {word_count}\n🔄 Step 4/7: Analyzing content for summary and search terms..."
            
            # Step 4: AI Agent Content Analysis & Search Term Generation
            video_summary = "[Step 4 analysis not run in Modal flow yet]"
            search_terms = []

            if STEP4_AGENT_AVAILABLE:
                if content_analyzer is None: # Should ideally be initialized already if available
                    logger.warning("ContentAnalysisAgent was None in Modal flow, attempting re-initialization.")
                    try:
                        content_analyzer = ContentAnalysisAgent()
                    except Exception as e_agent_init:
                        logger.error("Error re-initializing ContentAnalysisAgent in modal context: %s", e_agent_init, exc_info=True)
                        # Fallback to a dummy to prevent crash
                        class ContentAnalysisAgent_Dummy_Modal: # Ensure this class is defined or handled properly
                            def analyze_transcript(self, transcript: str):
                                logger.info("Dummy ContentAnalysisAgent_Dummy_Modal used.")
                                return {"video_summary": "[Content analysis agent failed to initialize in Modal]", "search_terms": []}
                        content_analyzer = ContentAnalysisAgent_Dummy_Modal()
                
                # Ensure content_analyzer is not None before calling
                if hasattr(content_analyzer, 'analyze_transcript'):
                    analysis_results = content_analyzer.analyze_transcript(transcript_text)
                    video_summary = analysis_results["video_summary"]
                    search_terms = analysis_results["search_terms"]
                    status += f"\n✅ Step 4/7: Content analysis complete!\n🔍 Search Terms: {', '.join(search_terms) if search_terms else 'None generated'}"
                else:
                    status += "\n⚠️ Step 4/7: Content analysis agent not properly initialized."
            else:
                status += "\n⚠️ Step 4/7: Content analysis skipped (agent not available)."
            
            # Step 5: YouTube Research & Competitive Analysis
            research_data = {}
            if STEP5_AGENT_AVAILABLE and search_terms:
                status += f"\n🔄 Step 5/7: Researching competitive content on YouTube..."
                
                if youtube_researcher is None:
                    try:
                        youtube_researcher = YouTubeResearchAgent()
                    except Exception as e:
                        logger.error("Error initializing YouTubeResearchAgent in Modal flow: %s", e, exc_info=True)
                        youtube_researcher = None
                
                if youtube_researcher:
                    try:
                        research_data = youtube_researcher.research_competitive_content(search_terms)
                        videos_analyzed = research_data.get("research_summary", {}).get("total_videos_analyzed", 0)
                        avg_views = research_data.get("competitive_analysis", {}).get("average_views", 0)
                        status += f"\n✅ Step 5/7: YouTube research complete! Analyzed {videos_analyzed} videos (avg views: {avg_views:,})"
                        logger.info("YouTube research complete. Analyzed %d videos.", videos_analyzed)
                    except Exception as e:
                        logger.error("Error during YouTube research: %s", e, exc_info=True)
                        status += f"\n❌ Step 5/7: YouTube research failed: {str(e)}"
                        research_data = {}
                else:
                    status += "\n⚠️ Step 5/7: YouTube research skipped (agent initialization failed)."
            elif not search_terms:
                status += "\n⚠️ Step 5/7: YouTube research skipped (no search terms from Step 4)."
            else:
                status += "\n⚠️ Step 5/7: YouTube research skipped (agent not available)."
            
            # Step 6: Content Enhancement (Optimized Titles & Descriptions)
            optimized_title = f"[Step 6 needed] Video Title - Search Terms: {', '.join(search_terms) if search_terms else 'N/A'}"
            optimized_description = f"Video Summary (Step 4):\n{video_summary}\n\n[Step 6 needed for final description]"
            
            if STEP6_AGENT_AVAILABLE and video_summary:
                status += f"\n🔄 Step 6/6: Generating optimized content using competitive research..."
                
                if content_enhancer is None:
                    try:
                        content_enhancer = ContentEnhancementAgent()
                    except Exception as e:
                        logger.error("Error initializing ContentEnhancementAgent in Modal flow: %s", e, exc_info=True)
                        content_enhancer = None
                
                if content_enhancer:
                    try:
                        enhancement_results = content_enhancer.enhance_content(
                            video_summary, search_terms, research_data
                        )
                        optimized_title = enhancement_results["optimized_title"]
                        optimized_description = enhancement_results["optimized_description"]
                        metadata = enhancement_results.get("enhancement_metadata", {}) # Ensure metadata exists
                        
                        status += f"\n✅ Step 6/7: Content enhancement complete!"
                        logger.info("Content enhancement complete.")
                        # Example of logging some metadata, adapt as needed
                        logger.info("Enhancement metadata: Analyzed %d competitor videos, Avg views: %s",
                                    metadata.get('competitive_videos_analyzed', 0),
                                    metadata.get('avg_competitor_views', 0))
                        # status string updates can remain for UI, or be built from logger messages if desired
                    except Exception as e:
                        logger.error("Error during content enhancement: %s", e, exc_info=True)
                        status += f"\n❌ Step 6/7: Content enhancement failed: {str(e)}"
                else:
                    status += "\n⚠️ Step 6/6: Content enhancement skipped (agent initialization failed)."
            else:
                if not video_summary:
                    status += "\n⚠️ Step 6/6: Content enhancement skipped (no video summary from Step 4)."
                else:
                    status += "\n⚠️ Step 6/6: Content enhancement skipped (agent not available)."
            
            # Step 7: Thumbnail Generation
            status += "\n🔄 Step 7/7: Generating viral-optimized thumbnails..."
            thumbnail1_data = None
            thumbnail2_data = None  
            thumbnail3_data = None
            
            try:
                if optimized_title and video_summary:
                    logger.info("🎨 Generating thumbnails with context: title=%s...", optimized_title[:50])
                    
                    thumbnail_results = generate_thumbnails.remote(
                        video_summary=video_summary,
                        optimized_title=optimized_title,
                        optimized_description=optimized_description,
                        search_terms=search_terms,
                        competitive_analysis=research_data
                    )
                    
                    if thumbnail_results and len(thumbnail_results) > 0: # Check if any results
                        processed_images = 0
                        for i, thumbnail_data in enumerate(thumbnail_results[:3]): # Process up to 3
                            pil_image = None
                            if thumbnail_data and thumbnail_data.get("image_data"):
                                try:
                                    image_bytes = base64.b64decode(thumbnail_data["image_data"])
                                    pil_image = Image.open(io.BytesIO(image_bytes))

                                    if pil_image and optimized_title and font:
                                        pil_image = pil_image.convert("RGBA")
                                        draw = ImageDraw.Draw(pil_image)
                                        text_to_overlay = optimized_title
                                        current_font_size_local = font_size # Use a local copy for dynamic resizing per image

                                        if len(text_to_overlay) > 50:
                                            words = text_to_overlay.split()
                                            if len(words) > 6:
                                                mid_point = len(words) // 2
                                                line1 = " ".join(words[:mid_point])
                                                line2 = " ".join(words[mid_point:])
                                                if len(line1) < 30 and len(line2) < 30:
                                                    text_to_overlay = f"{line1}\n{line2}"
                                                else:
                                                    text_to_overlay = " ".join(words[:7]) + "..."
                                            else:
                                                text_to_overlay = text_to_overlay[:47] + "..."

                                        current_font_object_local = font
                                        if hasattr(font, "path") and font.path:
                                            try:
                                                current_font_object_local = ImageFont.truetype(font.path, current_font_size_local)
                                            except IOError:
                                                logger.warning("Could not reload font %s at size %d. Using default.", font.path, current_font_size_local)
                                                current_font_object_local = ImageFont.load_default()
                                        else:
                                            current_font_object_local = ImageFont.load_default()

                                        text_bbox = draw.textbbox((0, 0), text_to_overlay, font=current_font_object_local)
                                        text_width = text_bbox[2] - text_bbox[0]
                                        text_height = text_bbox[3] - text_bbox[1]
                                        image_width, image_height = pil_image.size

                                        while text_width > image_width * 0.9 and current_font_size_local > 20:
                                            current_font_size_local -= 5
                                            if hasattr(font, "path") and font.path:
                                                try:
                                                    current_font_object_local = ImageFont.truetype(font.path, current_font_size_local)
                                                except IOError:
                                                    logger.warning("Could not reload font %s at size %d during resize. Using default.", font.path, current_font_size_local)
                                                    current_font_object_local = ImageFont.load_default()
                                                    break
                                            else:
                                                current_font_object_local = ImageFont.load_default()
                                                logger.debug("Using default font, cannot dynamically resize further in this loop.")
                                                break
                                            text_bbox = draw.textbbox((0, 0), text_to_overlay, font=current_font_object_local)
                                            text_width = text_bbox[2] - text_bbox[0]
                                            text_height = text_bbox[3] - text_bbox[1]

                                        x = (image_width - text_width) / 2
                                        y = image_height * 0.8 - (text_height / 2)
                                        padding = 10
                                        rect_x0 = max(0, x - padding)
                                        rect_y0 = max(0, y - padding - (text_height * (text_to_overlay.count('\n')) * 0.1))
                                        rect_x1 = min(image_width, x + text_width + padding)
                                        rect_y1 = min(image_height, y + text_height + padding + (text_height * (text_to_overlay.count('\n')) * 0.1))

                                        rect_img = Image.new('RGBA', pil_image.size, (255, 255, 255, 0))
                                        rect_draw = ImageDraw.Draw(rect_img)
                                        rect_draw.rectangle((rect_x0, rect_y0, rect_x1, rect_y1), fill=(0, 0, 0, 180))
                                        pil_image = Image.alpha_composite(pil_image, rect_img)
                                        draw = ImageDraw.Draw(pil_image)

                                        try:
                                            draw.text((x + text_width / 2, y + text_height / 2), text_to_overlay, font=current_font_object_local, fill=(255, 255, 255, 255), anchor="mm", align="center")
                                        except TypeError:
                                            num_lines = text_to_overlay.count('\n') + 1
                                            adjusted_y = y - (text_height / num_lines) * (num_lines -1) / 2
                                            draw.text((x, adjusted_y), text_to_overlay, font=current_font_object_local, fill=(255, 255, 255, 255), align="center")
                                    
                                    if i == 0: thumbnail1_data = pil_image
                                    elif i == 1: thumbnail2_data = pil_image
                                    elif i == 2: thumbnail3_data = pil_image
                                    processed_images +=1
                                        
                                except Exception as img_error:
                                    logger.error("❌ Error processing or overlaying text on thumbnail %d: %s", i+1, img_error, exc_info=True)
                                    if pil_image: # if image was decoded but overlay failed
                                        if i == 0: thumbnail1_data = pil_image
                                        elif i == 1: thumbnail2_data = pil_image
                                        elif i == 2: thumbnail3_data = pil_image
                            elif thumbnail_data and thumbnail_data.get("error"): # If Modal function returned an error for this thumbnail
                                logger.error("Thumbnail generation for style '%s' failed: %s", thumbnail_data.get('style', 'unknown'), thumbnail_data.get('error'))


                        status += f"\n✅ Step 7/7: Thumbnail generation and text overlay (attempted) complete! Generated {processed_images}/{len(thumbnail_results)} successfully."
                        if thumbnail_results[0] and thumbnail_results[0].get("concepts"): # Check if first result exists
                            concepts_used = thumbnail_results[0].get("concepts", [])
                            if concepts_used:
                                status += f"\n🎯 Key concepts used for thumbnails: {', '.join(concepts_used[:5])}"
                    else:
                        status += "\n⚠️ Step 7/7: Thumbnail generation returned no results or unexpected format."
                        logger.warning("Thumbnail generation returned no results or unexpected format.")
                else:
                    status += "\n⚠️ Step 7/7: Thumbnail generation skipped (missing title or summary)."
                    logger.info("Thumbnail generation skipped (missing title or summary).")
                    
            except Exception as e:
                logger.error("❌ Error during thumbnail generation step: %s", e, exc_info=True)
                status += f"\n❌ Step 7/7: Thumbnail generation failed: {str(e)}"
            
            # Format search terms and research results for UI display (no logging needed for these display strings)
            search_terms_display = ", ".join(search_terms) if search_terms else "No search terms generated"
            research_results_display = format_youtube_research_results(research_data, search_terms)
            
            title = optimized_title
            description = optimized_description
            
            return (
                status,
                transcript_text,
                search_terms_display,
                research_results_display,
                title,
                description,
                thumbnail1_data, 
                thumbnail2_data, 
                thumbnail3_data  
            )
            
        else:
            # Demo mode without Modal
            filename = os.path.basename(video_file) if video_file else "demo_video.mp4"
            file_size = os.path.getsize(video_file) / (1024*1024) if video_file else 0
            
            status = f"🎭 DEMO MODE: Simulating video processing for {filename} ({file_size:.1f} MB)\n" \
                    f"✅ Step 1/6: Video upload simulated\n" \
                    f"✅ Step 2/6: Audio extraction simulated\n" \
                    f"✅ Step 3/6: Whisper transcription simulated"
            
            demo_transcript_full = """🎭 Demo Transcription (Step 3 output):

Hello and welcome to this tutorial on building AI applications with Gradio and Modal. In this video, we'll explore how to create scalable, cloud-based AI workflows that can process video content automatically.

First, we'll set up our Modal infrastructure with GPU support for running Whisper AI transcription. Then we'll integrate it with a beautiful Gradio interface that allows users to upload videos and get instant results.

Thank you for watching, and don't forget to subscribe for more AI development tutorials!"""
            demo_transcript_text_only = demo_transcript_full.split(":\n\n", 1)[1] if ":\n\n" in demo_transcript_full else demo_transcript_full

            # Simulate Step 4 using the ContentAnalysisAgent if available
            if STEP4_AGENT_AVAILABLE and content_analyzer and hasattr(content_analyzer, 'analyze_transcript'):
                status += "\n🔄 Step 4/6: Running local content analysis via agent..."
                try:
                    analysis_results = content_analyzer.analyze_transcript(demo_transcript_text_only) 
                    video_summary = analysis_results["video_summary"]
                    search_terms = analysis_results["search_terms"]
                    status += f"\n✅ Step 4/6: Local content analysis by agent complete!\n🔍 Search Terms: {', '.join(search_terms) if search_terms else 'None generated'}"
                except Exception as e:
                    print(f"❌ Error during local Step 4 agent simulation: {e}")
                    video_summary = "[Error in local Step 4 agent simulation]"
                    search_terms = []
                    status += f"\n❌ Step 4/6: Local content analysis by agent failed: {e}"
            else:
                video_summary = "[Content analysis agent (Step 4) would run here if available and initialized]"
                search_terms = []
                status += "\n⚠️ Step 4/6: Content analysis by agent skipped (agent not available/initialized)."

            # Simulate Step 5: YouTube Research
            research_data = {}
            if STEP5_AGENT_AVAILABLE and youtube_researcher and search_terms:
                status += "\n🔄 Step 5/6: Running YouTube research simulation..."
                try:
                    research_data = youtube_researcher.research_competitive_content(search_terms[:1])  # Use first term only for demo
                    videos_analyzed = research_data["research_summary"]["total_videos_analyzed"]
                    avg_views = research_data["competitive_analysis"]["average_views"]
                    status += f"\n✅ Step 5/7: YouTube research complete! Analyzed {videos_analyzed} videos (avg views: {avg_views:,})"
                except Exception as e:
                    print(f"❌ Error during YouTube research simulation: {e}")
                    status += f"\n❌ Step 5/6: YouTube research failed: {str(e)}"
                    research_data = {}
            else:
                status += "\n⚠️ Step 5/6: YouTube research skipped (agent not available or no search terms)."

            # Simulate Step 6: Content Enhancement
            if STEP6_AGENT_AVAILABLE and content_enhancer and video_summary:
                status += "\n🔄 Step 6/6: Running content enhancement simulation..."
                try:
                    enhancement_results = content_enhancer.enhance_content(
                        video_summary, search_terms, research_data
                    )
                    optimized_title = enhancement_results["optimized_title"]
                    optimized_description = enhancement_results["optimized_description"]
                    metadata = enhancement_results["enhancement_metadata"]
                    
                    status += f"\n✅ Step 6/6: Content enhancement complete!"
                    status += f"\n📊 Enhancement Stats: {metadata['competitive_videos_analyzed']} videos analyzed, "
                    status += f"avg engagement: {metadata['avg_engagement_rate']:.2f}%"
                    if metadata.get('successful_patterns'):
                        status += f"\n🎯 Applied patterns: {', '.join(metadata['successful_patterns'][:2])}"
                    if metadata.get('top_keywords'):
                        status += f"\n🔑 Key trending words: {', '.join(metadata['top_keywords'][:4])}"
                    
                    title = f"🎬 {optimized_title}"
                    description = f"🎭 DEMO MODE - Optimized Content:\n\n{optimized_description}"
                except Exception as e:
                    print(f"❌ Error during content enhancement simulation: {e}")
                    status += f"\n❌ Step 6/6: Content enhancement failed: {str(e)}"
                    title = f"🎬 Demo Generated Title: (Using Step 4 agent search terms: {', '.join(search_terms) if search_terms else 'N/A'})"
                    description = f"🎭 Demo Generated Description (incorporating Step 4 agent summary):\n\nSummary:\n{video_summary}\n\nThis is a demo of our YouTube Content Optimizer."
            else:
                status += "\n⚠️ Step 6/6: Content enhancement skipped (agent not available or no video summary)."
                title = f"🎬 Demo Generated Title: (Using Step 4 agent search terms: {', '.join(search_terms) if search_terms else 'N/A'})"
                description = f"🎭 Demo Generated Description (incorporating Step 4 agent summary):\n\nSummary:\n{video_summary}\n\nThis is a demo of our YouTube Content Optimizer."

            # Step 7: Demo Thumbnail Generation
            status += "\n🔄 Step 7/7: Simulating thumbnail generation..."
            
            # In demo mode, we'll create placeholder thumbnails showing the concepts
            thumbnail1_data = None
            thumbnail2_data = None
            thumbnail3_data = None
            
            try:
                from PIL import Image, ImageDraw, ImageFont
                import io
                
                # Create demo thumbnails with different styles
                thumbnail_styles = ["Dynamic Style", "Tech Modern", "Educational"]
                demo_thumbnails = []
                
                for i, style in enumerate(thumbnail_styles):
                    # Create a placeholder thumbnail
                    img = Image.new('RGB', (1280, 720), color=(50 + i*40, 100 + i*30, 200 - i*20))
                    draw = ImageDraw.Draw(img)
                    
                    # Try to use a default font, fallback to PIL default
                    try:
                        font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 60)
                        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 30)
                    except:
                        font_large = ImageFont.load_default()
                        font_small = ImageFont.load_default()
                    
                    # Add style text
                    draw.text((50, 50), f"DEMO THUMBNAIL", fill="white", font=font_large)
                    draw.text((50, 150), f"{style}", fill="yellow", font=font_small)
                    draw.text((50, 200), f"Video: {filename[:30]}...", fill="white", font=font_small)
                    
                    # Add some design elements
                    draw.rectangle([50, 300, 600, 320], fill="yellow")
                    draw.text((50, 350), "🎬 AI-Generated Content", fill="white", font=font_small)
                    
                    if i == 0:
                        thumbnail1_data = img
                    elif i == 1:
                        thumbnail2_data = img
                    elif i == 2:
                        thumbnail3_data = img
                
                status += "\n✅ Step 7/7: Demo thumbnails generated! (3 placeholder styles)"
                status += "\n🎯 Demo concepts: AI, tutorial, technology, optimization"
                
            except Exception as e:
                print(f"❌ Error creating demo thumbnails: {e}")
                status += f"\n❌ Step 7/7: Demo thumbnail creation failed: {str(e)}"
            
            # Format search terms and research results for demo display
            search_terms_display = ", ".join(search_terms) if search_terms else "No search terms generated"
            research_results_display = format_youtube_research_results(research_data, search_terms) if research_data else "Demo mode - YouTube research simulated"
            
            return (
                status,
                demo_transcript_full, # Show the full demo transcript string
                search_terms_display,
                research_results_display,
                title,
                description,
                thumbnail1_data, 
                thumbnail2_data, 
                thumbnail3_data  
            )
        
    except Exception as e:
        error_status = f"❌ Error processing video: {str(e)}" # This will be displayed in UI
        logger.exception("Critical error in process_video: %s", e) # Full stack trace for logs
        return error_status, "", "", "", "", "", None, None, None

def test_modal_connection():
    """Test if Modal connection works"""
    if not MODAL_AVAILABLE:
        logger.warning("⚠️ Modal not available - running in demo mode")
        return True
        
    try:
        # Test both functions are accessible
        if (hasattr(process_video_to_audio, 'remote') and 
            hasattr(transcribe_audio_with_whisper, 'remote')): # Add other functions if needed
            logger.info("✅ Modal app connection successful!")
            logger.info("✅ Audio extraction function ready")
            logger.info("✅ Whisper transcription function ready")
            # logger.info("✅ Thumbnail generation function ready") # Assuming it's also checked or loaded
            return True
        else:
            logger.error("❌ Modal functions not found")
            return False
    except Exception as e:
        logger.error("❌ Modal connection error: %s", e, exc_info=True)
        return False

def create_interface():
    """Create the Gradio interface for YouTube content optimization"""
    
    # Add demo notice if Modal is not available
    demo_notice = """
    <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 10px; margin: 10px 0;">
        <strong>🎭 Demo Mode:</strong> Running without Modal backend. Upload a video to see the simulated workflow!
    </div>
    """ if not MODAL_AVAILABLE else ""
    
    with gr.Blocks(
        title="YouTube Content Optimizer",
        theme=gr.themes.Soft(),
        css="""
        .container {
            max-width: 1000px;
            margin: auto;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .section {
            margin: 2rem 0;
            padding: 1.5rem;
            border-radius: 8px;
            background: rgba(0,0,0,0.02);
        }
        .section-indicator {
            color: #FFFFFF;
            font-size: 1.3em;
            margin: 1.5rem 0 1rem 0;
            text-align: center;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .thumbnail-placeholder {
            text-align: center;
            padding: 2rem;
            color: #666;
            font-style: italic;
        }
        """
    ) as app:
        
        gr.HTML(f"""
        <div class="header">
            <h1>🎬 YouTube Content Optimizer</h1>
            <p>AI-powered video processing for automated titles, descriptions, and thumbnails</p>
        </div>
        {demo_notice}
        """)
        
        # Upload Section - Section 1 of 6
        gr.HTML('<div class="section-indicator">Section 1 of 6</div>')
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(
                    label="Upload your video file",
                    height=300
                )
                
                process_btn = gr.Button(
                    "🚀 Process Video", 
                    variant="primary",
                    size="lg"
                )
                
                # Status right below the button
                status_output = gr.Textbox(
                    label="Processing Status",
                    lines=4,
                    interactive=False,
                    visible=True
                )
                
                # Transcription display - NEW!
                transcription_output = gr.Textbox(
                    label="📝 Generated Transcription",
                    lines=8,
                    interactive=False,
                    placeholder="Transcription will appear here after audio processing...",
                    visible=True
                )
        
        # YouTube Research Section - Section 2 of 4
        gr.HTML('<div class="section">')
        gr.HTML('<div class="section-indicator">Section 2 of 4 - YouTube Competitive Research</div>')
        gr.Markdown("## 🔍 Search Terms & Competitive Analysis")
        
        # Search terms display
        search_terms_output = gr.Textbox(
            label="🎯 Generated Search Terms",
            lines=2,
            interactive=False,
            placeholder="Search terms from content analysis will appear here..."
        )
        
        # YouTube research results
        youtube_research_output = gr.Textbox(
            label="📊 YouTube Competitive Research Results",
            lines=10,
            interactive=False,
            placeholder="Competitive analysis from YouTube will appear here..."
        )
        gr.HTML('</div>')

        # Content Section - Section 3 of 4  
        gr.HTML('<div class="section">')
        gr.HTML('<div class="section-indicator">Section 3 of 4 - Optimized Content</div>')
        gr.Markdown("## 📝 AI-Optimized Content (Based on Competitive Research)")
        
        title_output = gr.Textbox(
            label="🎬 Optimized Title",
            lines=2,
            interactive=False,
            placeholder="AI-optimized title using competitive research will appear here..."
        )
        
        description_output = gr.Textbox(
            label="📄 Optimized Description", 
            lines=8,
            interactive=False,
            placeholder="AI-optimized description using competitive research will appear here..."
        )
        gr.HTML('</div>')
        
        # Thumbnails Section - Section 4 of 4
        gr.HTML('<div class="section">')
        gr.HTML('<div class="section-indicator">Section 4 of 4 - AI-Generated Thumbnails</div>')
        gr.Markdown("## 🖼️ Viral-Optimized Thumbnails")
        
        with gr.Row():
            thumbnail1 = gr.Image(
                label="Option 1",
                height=250,
                show_label=True
            )
            thumbnail2 = gr.Image(
                label="Option 2", 
                height=250,
                show_label=True
            )
            thumbnail3 = gr.Image(
                label="Option 3",
                height=250,
                show_label=True
            )
        
        # Placeholder text for thumbnails when not generated
        thumbnail_placeholder = gr.HTML(
            '<div class="thumbnail-placeholder">🎨 Upload and process a video to generate viral-optimized thumbnails using AI!</div>',
            visible=True
        )
        gr.HTML('</div>')
        
        # Connect the processing function
        process_btn.click(
            fn=process_video,
            inputs=[video_input],
            outputs=[
                status_output,
                transcription_output,
                search_terms_output,
                youtube_research_output,
                title_output, 
                description_output,
                thumbnail1,
                thumbnail2,
                thumbnail3
            ]
        )
    
    return app

# Create and launch the interface
if __name__ == "__main__":
    # Test Modal setup before starting Gradio
    logger.info("🧪 Testing Modal connection...")
    
    if MODAL_AVAILABLE:
        try:
            if test_modal_connection():
                logger.info("✅ Modal is ready! Starting Gradio interface...")
                if STEP4_AGENT_AVAILABLE and content_analyzer is None: # Should be initialized by now if Modal is up
                    logger.info("INFO: Modal available, local ContentAnalysisAgent not pre-initialized. Will init if needed in process_video.")
            else:
                logger.warning("⚠️ Modal connection failed.")
                if STEP4_AGENT_AVAILABLE and (content_analyzer is None or not Settings.llm): # Check llama_index settings
                    logger.info("🔄 Modal failed, ensuring ContentAnalysisAgent for local Step 4 processing...")
                    try:
                        content_analyzer = ContentAnalysisAgent() # Attempt to init
                        logger.info("✅ ContentAnalysisAgent initialized for local fallback.")
                    except Exception as e:
                        logger.error("⚠️ Error initializing ContentAnalysisAgent for local fallback: %s.", e, exc_info=True)
        except Exception as e: # Catch any other exception during Modal test
            logger.error("⚠️ Modal connection error during startup: %s", e, exc_info=True)
            if STEP4_AGENT_AVAILABLE and (content_analyzer is None or not Settings.llm):
                logger.info("🔄 Modal error, ensuring ContentAnalysisAgent for local Step 4 processing...")
                try:
                    content_analyzer = ContentAnalysisAgent()
                    logger.info("✅ ContentAnalysisAgent initialized for local fallback after Modal error.")
                except Exception as e_agent:
                    logger.error("⚠️ Error initializing ContentAnalysisAgent on Modal error: %s.", e_agent, exc_info=True)
    else: # Modal not available from the start
        logger.info("🎭 Modal not available.")
        if STEP4_AGENT_AVAILABLE and (content_analyzer is None or not Settings.llm):
            logger.info("🔄 Modal not available, ensuring ContentAnalysisAgent for local Step 4 processing...")
            try:
                content_analyzer = ContentAnalysisAgent()
                logger.info("✅ ContentAnalysisAgent initialized for local use as Modal is unavailable.")
            except Exception as e:
                logger.error("⚠️ Error initializing ContentAnalysisAgent as Modal is unavailable: %s.", e, exc_info=True)

    # Log final status of agents before launching
    if not MODAL_AVAILABLE and not STEP4_AGENT_AVAILABLE:
        logger.info("🎭 Starting in full demo mode (Modal and Step 4 Agent not available)")
    elif not MODAL_AVAILABLE and STEP4_AGENT_AVAILABLE and content_analyzer and Settings.llm:
        logger.info("🎭 Starting in demo mode for Steps 1-3, with local Step 4 Content Analysis Agent.")
    elif not MODAL_AVAILABLE and STEP4_AGENT_AVAILABLE and (content_analyzer is None or not Settings.llm):
        logger.info("🎭 Starting in demo mode for Steps 1-3. Step 4 Agent available but may have failed to initialize LLM.")

    if MODAL_AVAILABLE : # Only init these if Modal is available, as they depend on Modal flow data
        logger.info("Modal available. Agents for Steps 4, 5, 6 will be initialized within Modal flow if needed.")
        # The logic to initialize these agents if they are None is already inside process_video for the Modal path.
        # No need to pre-initialize them here if Modal is the primary path.

    demo = create_interface()
    logger.info("🚀 Launching Gradio interface on 0.0.0.0:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False # Set to True if you want to share a public link (requires Gradio account/login sometimes)
    ) 