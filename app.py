import gradio as gr
import os
import modal
from llama_index.core import Settings # Added for checking Settings.llm

# Try to import and connect to Modal, but gracefully handle if not available
MODAL_AVAILABLE = False

# Note: Step 4 is implemented via ContentAnalysisAgent in agents/ directory
# No need for separate src.step4_content_analysis module

try:
    # Import the deployed functions using Modal's Function.from_name method
    process_video_to_audio = modal.Function.from_name("youtube-content-optimizer", "process_video_to_audio")
    transcribe_audio_with_whisper = modal.Function.from_name("youtube-content-optimizer", "transcribe_audio_with_whisper")
    MODAL_AVAILABLE = True
    print("✅ Modal connection established!")
except Exception as e:
    print(f"⚠️ Modal not available: {e}")
    print("🔄 Running in demo mode without Modal processing")
    MODAL_AVAILABLE = False

# Step 4 initialization is handled by ContentAnalysisAgent class

# Attempt to import Step 4 Agent
try:
    from agents.content_analysis_agent import ContentAnalysisAgent
    STEP4_AGENT_AVAILABLE = True
    # Instantiate the agent. This will also initialize LlamaIndex for Ollama on instantiation.
    # We will do this conditionally later based on whether Modal is running or not,
    # to avoid initializing Ollama if Modal will handle Step 4.
    content_analyzer = None 
    print("✅ Step 4 (ContentAnalysisAgent) class loaded.")
except ImportError as e:
    print(f"⚠️ Step 4 Agent (agents.content_analysis_agent) not found or import error: {e}. Content analysis will be skipped.")
    STEP4_AGENT_AVAILABLE = False
    # Define a dummy class and instance if not available so the app can run
    class ContentAnalysisAgent:
        def __init__(self):
            print("Dummy ContentAnalysisAgent initialized.")
        def analyze_transcript(self, transcript: str):
            print("Dummy ContentAnalysisAgent.analyze_transcript called.")
            return {
                "video_summary": "[Content analysis agent not available]",
                "search_terms": []
            }
    content_analyzer = ContentAnalysisAgent() # Instantiate dummy for safety

# Initialize ContentAnalysisAgent for local use if Step 4 is available and Modal is not
# This agent will handle its own LlamaIndex Ollama initialization.
if STEP4_AGENT_AVAILABLE and not MODAL_AVAILABLE:
    try:
        print("🔄 Initializing ContentAnalysisAgent for local Step 4 processing...")
        content_analyzer = ContentAnalysisAgent()
        print("✅ ContentAnalysisAgent initialized for local use.")
    except Exception as e:
        print(f"⚠️ Error initializing ContentAnalysisAgent for local Step 4: {e}. Step 4 might fail.")
        # Fallback to dummy if initialization fails
        if not isinstance(content_analyzer, ContentAnalysisAgent) or content_analyzer is None:
            class ContentAnalysisAgent_Dummy:
                def __init__(self):
                    print("Fallback Dummy ContentAnalysisAgent initialized after error.")
                def analyze_transcript(self, transcript: str):
                    return {"video_summary": "[Content analysis agent failed to initialize]", "search_terms": []}
            content_analyzer = ContentAnalysisAgent_Dummy()

def process_video(video_file):
    """
    Main processing function for the video content optimization pipeline.
    Steps 1-3: Video upload, audio extraction, and Whisper transcription via Modal
    Step 4: Content Analysis (Summary & Search Terms) using ContentAnalysisAgent
    Steps 5-6: To be implemented (YouTube research, enhanced content, thumbnails)
    """
    global content_analyzer # Make sure we're using the potentially initialized agent

    if video_file is None:
        return "Please upload a video file.", "", "", "", None, None, None
    
    try:
        if MODAL_AVAILABLE:
            # Step 1: Video Upload Processing
            status = "🔄 Step 1/4: Processing video... Uploading to Modal for audio extraction"
            
            # Read video file
            with open(video_file, "rb") as f:
                video_bytes = f.read()
            
            filename = os.path.basename(video_file)
            
            print(f"📤 Sending video to Modal for processing...")
            
            # Step 2: Audio Extraction
            audio_bytes, audio_filename = process_video_to_audio.remote(video_bytes, filename)
            
            status = f"✅ Step 2/4: Audio extraction successful! File: {audio_filename} ({len(audio_bytes) / (1024*1024):.1f} MB)\n🔄 Step 3/4: Starting Whisper transcription..."
            
            # Step 3: Whisper Transcription
            print(f"🎙️ Sending audio to Whisper for transcription...")
            transcription_data = transcribe_audio_with_whisper.remote(audio_bytes, audio_filename)
            
            # Extract transcription details
            transcript_text = transcription_data["text"]
            language = transcription_data["language"]
            duration = transcription_data["duration"]
            word_count = transcription_data["word_count"]
            
            status = f"✅ Step 3/4: Transcription complete! Lang: {language.upper()}, Duration: {duration:.1f}s, Words: {word_count}\n🔄 Step 4/4: Analyzing content for summary and search terms..."
            
            # Step 4: AI Agent Content Analysis & Search Term Generation
            # If Step 4 were a Modal function, it would be called here.
            # For now, if MODAL_AVAILABLE, we assume Step 4 might still run locally 
            # (e.g. app.py is hosted on Modal, but calls a local agent for Step 4)
            # OR that ContentAnalysisAgent is designed to be part of a Modal deployment itself.
            # This part needs clarification if Step 4 should *also* be a Modal remote call.
            # Assuming for now it uses the globally available `content_analyzer` instance
            # which would be configured based on where app.py runs.
            video_summary = "[Step 4 analysis not run in Modal flow yet]"
            search_terms = []

            if STEP4_AGENT_AVAILABLE:
                if content_analyzer is None:
                    print("Attempting to initialize ContentAnalysisAgent for Modal context (should be pre-initialized ideally)")
                    try:
                        content_analyzer = ContentAnalysisAgent()
                    except Exception as e_agent_init:
                        print(f"Error initializing agent in modal context: {e_agent_init}")
                        # Fallback to a dummy to prevent crash
                        class ContentAnalysisAgent_Dummy_Modal:
                            def analyze_transcript(self, transcript: str):
                                return {"video_summary": "[Content analysis agent failed to initialize in Modal]", "search_terms": []}
                        content_analyzer = ContentAnalysisAgent_Dummy_Modal()
                
                # Ensure content_analyzer is not None before calling (it should be an instance)
                if hasattr(content_analyzer, 'analyze_transcript'):
                    analysis_results = content_analyzer.analyze_transcript(transcript_text)
                    video_summary = analysis_results["video_summary"]
                    search_terms = analysis_results["search_terms"]
                    status += f"\n✅ Step 4/4: Content analysis complete!\n🔍 Search Terms: {', '.join(search_terms) if search_terms else 'None generated'}"
                else:
                    status += "\n⚠️ Step 4/4: Content analysis agent not properly initialized."
            else:
                status += "\n⚠️ Step 4/4: Content analysis skipped (agent not available)."
            
            title = f"Generated Title: [Step 5 & 6 needed - Search Terms: {', '.join(search_terms) if search_terms else 'N/A'}]"
            description = f"Video Summary (Step 4):\n{video_summary}\n\n[Step 5 & 6 needed for final description based on summary and research]"
            
            return (
                status,
                title,
                description,
                transcript_text, 
                None, 
                None, 
                None  
            )
            
        else:
            # Demo mode without Modal
            filename = os.path.basename(video_file) if video_file else "demo_video.mp4"
            file_size = os.path.getsize(video_file) / (1024*1024) if video_file else 0
            
            status = f"🎭 DEMO MODE: Simulating video processing for {filename} ({file_size:.1f} MB)\n" \
                    f"✅ Step 1/4: Video upload simulated\n" \
                    f"✅ Step 2/4: Audio extraction simulated\n" \
                    f"✅ Step 3/4: Whisper transcription simulated"
            
            demo_transcript_full = """🎭 Demo Transcription (Step 3 output):

Hello and welcome to this tutorial on building AI applications with Gradio and Modal. In this video, we'll explore how to create scalable, cloud-based AI workflows that can process video content automatically.

First, we'll set up our Modal infrastructure with GPU support for running Whisper AI transcription. Then we'll integrate it with a beautiful Gradio interface that allows users to upload videos and get instant results.

Thank you for watching, and don't forget to subscribe for more AI development tutorials!"""
            demo_transcript_text_only = demo_transcript_full.split(":\n\n", 1)[1] if ":\n\n" in demo_transcript_full else demo_transcript_full

            # Simulate Step 4 using the ContentAnalysisAgent if available (already initialized if not MODAL_AVAILABLE)
            if STEP4_AGENT_AVAILABLE and content_analyzer and hasattr(content_analyzer, 'analyze_transcript'):
                status += "\n🔄 Step 4/4: Running local content analysis via agent..."
                try:
                    analysis_results = content_analyzer.analyze_transcript(demo_transcript_text_only) 
                    video_summary = analysis_results["video_summary"]
                    search_terms = analysis_results["search_terms"]
                    status += f"\n✅ Step 4/4: Local content analysis by agent complete!\n🔍 Search Terms: {', '.join(search_terms) if search_terms else 'None generated'}"
                except Exception as e:
                    print(f"❌ Error during local Step 4 agent simulation: {e}")
                    video_summary = "[Error in local Step 4 agent simulation]"
                    search_terms = []
                    status += f"\n❌ Step 4/4: Local content analysis by agent failed: {e}"
            else:
                video_summary = "[Content analysis agent (Step 4) would run here if available and initialized]"
                search_terms = []
                status += "\n⚠️ Step 4/4: Content analysis by agent skipped (agent not available/initialized)."

            title = f"🎬 Demo Generated Title: (Using Step 4 agent search terms: {', '.join(search_terms) if search_terms else 'N/A'})"
            description = f"🎭 Demo Generated Description (incorporating Step 4 agent summary):\n\nSummary:\n{video_summary}\n\nThis is a demo of our YouTube Content Optimizer. Steps 5 & 6 will refine this further."

            return (
                status,
                title,
                description,
                demo_transcript_full, # Show the full demo transcript string
                None, 
                None, 
                None  
            )
        
    except Exception as e:
        error_status = f"❌ Error processing video: {str(e)}"
        print(f"Error details: {e}")
        return error_status, "", "", "", None, None, None

def test_modal_connection():
    """Test if Modal connection works"""
    if not MODAL_AVAILABLE:
        print("⚠️ Modal not available - running in demo mode")
        return True
        
    try:
        # Test both functions are accessible
        if (hasattr(process_video_to_audio, 'remote') and 
            hasattr(transcribe_audio_with_whisper, 'remote')):
            print("✅ Modal app connection successful!")
            print("✅ Audio extraction function ready")
            print("✅ Whisper transcription function ready")
            return True
        else:
            print("❌ Modal functions not found")
            return False
    except Exception as e:
        print(f"❌ Modal connection error: {e}")
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
        
        # Upload Section - Section 1 of 4
        gr.HTML('<div class="section-indicator">Section 1 of 4</div>')
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
        
        # Content Section - Section 2 of 4
        gr.HTML('<div class="section">')
        gr.HTML('<div class="section-indicator">Section 2 of 4</div>')
        gr.Markdown("## 📝 Generated Content")
        
        title_output = gr.Textbox(
            label="Title",
            lines=2,
            interactive=False,
            placeholder="Generated title will appear here..."
        )
        
        description_output = gr.Textbox(
            label="Description",
            lines=8,
            interactive=False,
            placeholder="Generated description will appear here..."
        )
        gr.HTML('</div>')
        
        # Thumbnails Section - Section 3 of 4
        gr.HTML('<div class="section">')
        gr.HTML('<div class="section-indicator">Section 3 of 4</div>')
        gr.Markdown("## 🖼️ Generated Thumbnails")
        
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
            '<div class="thumbnail-placeholder">Finalize the title and description to generate thumbnails</div>',
            visible=True
        )
        gr.HTML('</div>')
        
        # Connect the processing function
        process_btn.click(
            fn=process_video,
            inputs=[video_input],
            outputs=[
                status_output,
                title_output, 
                description_output,
                transcription_output,  # NEW transcription output
                thumbnail1,
                thumbnail2,
                thumbnail3
            ]
        )
    
    return app

# Create and launch the interface
if __name__ == "__main__":
    # Test Modal setup before starting Gradio
    print("🧪 Testing Modal connection...")
    
    if MODAL_AVAILABLE:
        try:
            if test_modal_connection():
                print("✅ Modal is ready! Starting Gradio interface...")
                # If Modal is primary, Step 4 agent might not be needed locally
                # unless app.py itself is run in Modal and ContentAnalysisAgent is part of that.
                if STEP4_AGENT_AVAILABLE and content_analyzer is None:
                    print("INFO: Modal available, local ContentAnalysisAgent not pre-initialized. Will init if needed in process_video.")
            else:
                print("⚠️ Modal connection failed.")
                # If Modal fails, try to ensure local agent is up if not already
                if STEP4_AGENT_AVAILABLE and (content_analyzer is None or not Settings.llm):
                    print("🔄 Modal failed, ensuring ContentAnalysisAgent for local Step 4 processing...")
                    try:
                        content_analyzer = ContentAnalysisAgent()
                        print("✅ ContentAnalysisAgent initialized for local fallback.")
                    except Exception as e:
                        print(f"⚠️ Error initializing ContentAnalysisAgent for local fallback: {e}.")
        except Exception as e:
            print(f"⚠️ Modal connection error: {e}")
            if STEP4_AGENT_AVAILABLE and (content_analyzer is None or not Settings.llm):
                print("🔄 Modal error, ensuring ContentAnalysisAgent for local Step 4 processing...")
                try:
                    content_analyzer = ContentAnalysisAgent()
                    print("✅ ContentAnalysisAgent initialized for local fallback after Modal error.")
                except Exception as e_agent:
                    print(f"⚠️ Error initializing ContentAnalysisAgent on Modal error: {e_agent}.")
    else:
        print("🎭 Modal not available.")
        # Ensure local agent is initialized if Modal is not available from the start
        if STEP4_AGENT_AVAILABLE and (content_analyzer is None or not Settings.llm):
            print("🔄 Modal not available, ensuring ContentAnalysisAgent for local Step 4 processing...")
            try:
                content_analyzer = ContentAnalysisAgent()
                print("✅ ContentAnalysisAgent initialized for local use as Modal is unavailable.")
            except Exception as e:
                print(f"⚠️ Error initializing ContentAnalysisAgent as Modal is unavailable: {e}.")

    if not MODAL_AVAILABLE and not STEP4_AGENT_AVAILABLE:
        print("🎭 Starting in full demo mode (Modal and Step 4 Agent not available)")
    elif not MODAL_AVAILABLE and STEP4_AGENT_AVAILABLE and content_analyzer and Settings.llm:
        print("🎭 Starting in demo mode for Steps 1-3, with local Step 4 Content Analysis Agent.")
    elif not MODAL_AVAILABLE and STEP4_AGENT_AVAILABLE and (content_analyzer is None or not Settings.llm):
        print("🎭 Starting in demo mode for Steps 1-3. Step 4 Agent available but failed to initialize Ollama.")
    elif MODAL_AVAILABLE and not STEP4_AGENT_AVAILABLE:
        # Logic for initializing or using content_analyzer when MODAL_AVAILABLE is handled within process_video and startup.
        print("✅ Modal available for Steps 1-3. ContentAnalysisAgent is available and will be used for Step 4.")
        # Ensure agent is initialized if Modal is up but local agent instance is still None
        if content_analyzer is None:
            try:
                print("Attempting to initialize ContentAnalysisAgent as Modal is up but instance was None.")
                content_analyzer = ContentAnalysisAgent()
            except Exception as e_init_modal_context:
                print(f"Failed to init ContentAnalysisAgent in Modal context startup: {e_init_modal_context}")

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False # Set to True if you want to share a public link (requires Gradio account/login sometimes)
    ) 