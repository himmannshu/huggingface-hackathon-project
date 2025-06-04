import gradio as gr
import os
import tempfile

# Try to import and connect to Modal, but gracefully handle if not available
MODAL_AVAILABLE = False
modal_app = None

try:
    import modal
    # Try to get Modal credentials from environment or local config
    modal_app = modal.App.lookup("youtube-content-optimizer", create_if_missing=False)
    MODAL_AVAILABLE = True
    print("✅ Modal connection established!")
except Exception as e:
    print(f"⚠️ Modal not available: {e}")
    print("🔄 Running in demo mode without Modal processing")
    MODAL_AVAILABLE = False

def process_video(video_file):
    """
    Main processing function for the video content optimization pipeline.
    Steps 1 & 2: Video upload and audio extraction via Modal (or demo mode)
    Steps 3-6: To be implemented (Whisper, content generation, thumbnails)
    """
    if video_file is None:
        return "Please upload a video file.", "", "", None, None, None
    
    try:
        if MODAL_AVAILABLE:
            # Real Modal processing
            status = "🔄 Processing video... Uploading to Modal for audio extraction"
            
            # Read video file
            with open(video_file, "rb") as f:
                video_bytes = f.read()
            
            filename = os.path.basename(video_file)
            
            print(f"📤 Sending video to Modal for processing...")
            
            # Call the deployed Modal function directly
            process_video_to_audio = modal_app.process_video_to_audio
            audio_bytes, audio_filename = process_video_to_audio.remote(video_bytes, filename)
            
            status = f"✅ Audio extraction successful! File: {audio_filename} ({len(audio_bytes) / (1024*1024):.1f} MB)"
            
            # Placeholder for Steps 3-6 (to be implemented)
            title = "Generated Title: [Step 3-6 to be implemented - Whisper transcription needed]"
            description = f"Generated Description: [Will be created after audio transcription]\n\nAudio file ready: {audio_filename}"
            
        else:
            # Demo mode without Modal
            filename = os.path.basename(video_file) if video_file else "demo_video.mp4"
            file_size = os.path.getsize(video_file) / (1024*1024) if video_file else 0
            
            status = f"🎭 DEMO MODE: Simulating video processing for {filename} ({file_size:.1f} MB)\n✅ In real deployment, this would extract audio via Modal + FFmpeg"
            
            title = "🎬 Demo Generated Title: 'How to Build Amazing AI Apps with Gradio and Modal'"
            description = """🎭 Demo Generated Description:

In this video, we explore the powerful combination of Gradio and Modal for building scalable AI applications. Learn how to:

• Process video content automatically
• Extract audio using FFmpeg in the cloud
• Generate engaging titles and descriptions
• Create stunning thumbnails with AI

This is a demo of our YouTube Content Optimizer - in the real version, this content would be generated from your actual video transcription using Whisper AI.

🚀 Built with: Gradio + Modal + OpenAI Whisper + AI Content Generation

#AI #ContentCreation #YouTube #Automation"""
        
        return (
            status,
            title,
            description, 
            None,  # Thumbnail 1
            None,  # Thumbnail 2
            None   # Thumbnail 3
        )
        
    except Exception as e:
        error_status = f"❌ Error processing video: {str(e)}"
        print(f"Error details: {e}")
        return error_status, "", "", None, None, None

def test_modal_connection():
    """Test if Modal connection works"""
    if not MODAL_AVAILABLE:
        print("⚠️ Modal not available - running in demo mode")
        return True
        
    try:
        # Simple connection test - just check if we can access the app
        process_video_to_audio = modal_app.process_video_to_audio
        print("✅ Modal app connection successful!")
        return True
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
        
        # Upload Section - Section 1 of 3
        gr.HTML('<div class="section-indicator">Section 1 of 3</div>')
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
                    label="Status",
                    lines=4,
                    interactive=False,
                    visible=True
                )
        
        # Content Section - Section 2 of 3
        gr.HTML('<div class="section">')
        gr.HTML('<div class="section-indicator">Section 2 of 3</div>')
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
        
        # Thumbnails Section - Section 3 of 3
        gr.HTML('<div class="section">')
        gr.HTML('<div class="section-indicator">Section 3 of 3</div>')
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
            else:
                print("⚠️ Modal connection failed, but continuing in demo mode...")
        except Exception as e:
            print(f"⚠️ Modal connection error: {e}")
            print("🔄 Continuing in demo mode...")
    else:
        print("🎭 Starting in demo mode (Modal not available)")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    ) 