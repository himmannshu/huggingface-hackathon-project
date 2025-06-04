---
title: YouTube Content Optimizer
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.32.1"
app_file: app.py
pinned: false
---

# SmartThumbnailer

## Goal
This project creates an AI-powered agent that automates YouTube content optimization by processing uploaded videos to generate transcriptions, enhanced titles, descriptions, and multiple thumbnail variations. The system uses advanced AI models including Whisper for transcription and content generation models for creating engaging YouTube-ready content.

For detailed project information, workflow process, and technical specifications, see [documentation.md](documentation/project-goals.md).

## Requirements
- Python 3.12.9
- Modal account (for cloud processing)

## Setup Instructions

1. **Create a virtual environment** (recommended):
   ```bash
   conda create -n hugging-face-hackathon python=3.12.9
   conda activate hugging-face-hackathon
   ```
   
   Or using venv:
   ```bash
   python3.12 -m venv hugging-face-hackathon
   source hugging-face-hackathon/bin/activate  # On macOS/Linux
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Modal for cloud processing**:
   
   a. **Create Modal account**: Sign up at [modal.com](https://modal.com)
   
   b. **Install Modal CLI** (if not already installed):
   ```bash
   pip install modal
   ```
   
   c. **Authenticate with Modal**:
   ```bash
   modal token new
   ```
   Follow the prompts to authenticate with your Modal account.
   
   d. **Deploy the Modal functions**:
   ```bash
   modal deploy modal_functions.py
   ```
   
   This will deploy both functions:
   - `process_video_to_audio` (FFmpeg audio extraction)
   - `transcribe_audio_with_whisper` (Whisper AI transcription with A10G GPU)

4. **Run the Gradio application**:
   ```bash
   python app.py
   ```
   
   The application will be available at: http://localhost:7860

## Modal Deployment Details

The project uses Modal for cloud-based processing with two main functions:

- **Audio Extraction** (`process_video_to_audio`):
  - Uses FFmpeg in Modal containers
  - 2GB memory, 10-minute timeout
  - Optimized audio output for Whisper (16kHz, mono, WAV)

- **Whisper Transcription** (`transcribe_audio_with_whisper`):
  - Uses OpenAI Whisper turbo model
  - A10G GPU with 12GB RAM
  - 15-minute timeout for long videos
  - Returns structured transcription with segments and metadata

### Redeploying Modal Functions

If you need to redeploy or update the Modal functions:

```bash
# Make sure you're in the project directory and environment is active
conda activate hugging-face-hackathon

# Deploy the updated functions
modal deploy modal_functions.py

# Verify deployment
modal app list
```

## Hugging Face Deployment

This project is configured to work seamlessly with Hugging Face Spaces:

1. **Push to Hugging Face**: Simply push this repository to a Hugging Face Space
2. **Entry Point**: The `app.py` file serves as the main entry point
3. **Auto-deployment**: Hugging Face will automatically detect and run the Gradio app

### Requirements for HF Deployment
- `app.py` file in the root directory ✅
- `requirements.txt` with dependencies ✅  
- Gradio configured to run on `0.0.0.0:7860` ✅
- Modal functions deployed separately ✅

## Project Structure

```
youtube-content-optimizer/
├── app.py                    # Main Gradio application (HF entry point)
├── modal_functions.py        # Modal cloud functions for processing
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
└── documentation/           # Detailed project docs
    └── project-goals.md    # Project workflow and technical details
```

## Packages Used
- gradio==5.32.1
- modal==1.0.2

## Features

### Current Implementation ✅
- **Video Upload**: Drag and drop video file input
- **Step 1-2: Audio Extraction**: FFmpeg processing in Modal cloud
- **Step 3: Whisper Transcription**: AI speech-to-text with GPU acceleration
- **Processing Pipeline**: Real-time status updates with progress tracking
- **Results Display**: 
  - Step-by-step processing status
  - Generated transcriptions with language detection
  - Placeholder for titles and descriptions
- **Responsive Design**: Modern UI with clean sections

### Planned Implementation 🚧
4. **Content Analysis** - AI-generated search terms from transcription
5. **Content Enhancement** - YouTube research integration  
6. **Thumbnail Generation** - Multiple style variations using transcription context

## Troubleshooting

### Modal Issues
- **Functions not found**: Ensure you've run `modal deploy modal_functions.py`
- **Authentication errors**: Run `modal token new` to re-authenticate
- **App in demo mode**: Check Modal deployment status with `modal app list`

### Local Development
- **Import errors**: Ensure virtual environment is activated
- **Missing dependencies**: Run `pip install -r requirements.txt`