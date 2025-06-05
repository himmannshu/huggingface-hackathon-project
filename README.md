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
   
   This will deploy the Modal functions required for video processing (audio extraction, transcription) and potentially thumbnail generation if you configure `modal_functions.py` for it.

4. **Configure Environment Variables**:
   Before running the application, you need to set up your local environment variables.
   Copy the `.env.example` file to a new file named `.env`:
   ```bash
   cp .env.example .env
   ```
   Now, edit the `.env` file with your specific configurations (Ollama endpoint, model name, Whisper model size, YouTube API key). Details on these variables are in the "AI Model Configuration" section below and within the `.env.example` file itself.

5. **Run the Gradio application**:
   ```bash
   python app.py
   ```
   
   The application will be available at: http://localhost:7860

## AI Model Configuration

The application utilizes AI models for several key tasks:
- **Text Generation (Summaries, Titles, Descriptions, etc.):** Handled by a model accessed via Ollama.
- **Audio Transcription:** Handled by OpenAI's Whisper model.

Configuring these appropriately is crucial for the application's performance and output quality.

### Ollama Configuration (for Text Generation)

The application leverages an Ollama model for various text generation tasks, including:
- Video summaries
- YouTube search term generation
- Optimized video titles
- Optimized video descriptions

The quality of these generated texts is highly dependent on the Ollama model you use.

### Environment Variables for Ollama:

You **must** configure these in your `.env` file:

-   `MODEL_ENDPOINT`: This is the base URL for your Ollama server.
    -   If you are running Ollama locally (recommended for ease of use), this will typically be `http://localhost:11434`.
    -   If you have deployed Ollama to a different server or are using a cloud-hosted Ollama instance (e.g., via Modal for Ollama specifically), use that URL.
-   `MODEL_NAME`: This is the specific model identifier that Ollama will use.
    -   You need to have this model downloaded in your Ollama instance (e.g., by running `ollama pull llama3`).

### Recommended Models:

For optimal performance and generation quality, we strongly recommend using modern, capable, instruction-tuned models. Older or smaller models might produce suboptimal or poorly formatted results.

Good choices include (but are not limited to):
-   `llama3` (or specific versions like `llama3:8b`, `llama3:70b`)
-   `llama3:instruct` (or specific versions like `llama3:8b-instruct`)
-   `mistral` (or `mistral:instruct`)
-   Other recent, high-performing models available through Ollama.

**Example `.env` configuration:**
```
MODEL_ENDPOINT=http://localhost:11434
MODEL_NAME=llama3:8b-instruct
```

Always ensure the chosen `MODEL_NAME` is available in your Ollama setup. You can list your downloaded models using `ollama list`.

### YouTube API Key:

Additionally, for the YouTube competitive research functionality (Step 5), you'll need to configure:

-   `YOUTUBE_API_KEY`: Your Google Cloud YouTube Data API v3 key. This should also be added to your `.env` file. If this key is not provided, the YouTube research step will be skipped. Refer to the `.env.example` file for the exact variable name.

### Whisper Model Configuration (for Transcription)

The application uses OpenAI's Whisper model for audio transcription. You can configure the size of the Whisper model used, which impacts a trade-off between transcription accuracy, speed, and resource consumption.

**Environment Variable:**

-   `WHISPER_MODEL_SIZE`: Specifies the Whisper model size. This variable should be set in your `.env` file.
    -   **Default:** `small` (if the variable is not set).

**Common Model Sizes & Trade-offs:**

-   **Smaller models** (e.g., `tiny`, `base`, `small`):
    -   Faster processing.
    -   Lower resource consumption (CPU, GPU memory, RAM).
    -   Generally lower transcription accuracy, especially with noisy audio or strong accents.
-   **Larger models** (e.g., `medium`, `large`, `large-v1`, `large-v2`, `large-v3`):
    -   Higher transcription accuracy.
    -   Slower processing.
    -   Higher resource consumption. `large-v3` is currently the most accurate.

We recommend starting with `small` for a balance. If higher accuracy is needed and you have sufficient resources, try `medium` or `large-v3`.

**Example `.env` configuration:**
```
WHISPER_MODEL_SIZE=medium
```

**Resource Note for Modal Users:**
Using larger Whisper models (especially `large`, `large-v2`, `large-v3`) on Modal may require more processing time and could potentially approach the default timeout or memory limits if not already configured for larger tasks. The current Modal function for transcription (`transcribe_audio_with_whisper`) is configured with a timeout of 900 seconds (15 minutes) and 12GB of RAM, which should generally accommodate up to `large-v3`, but performance can vary based on audio length and complexity. If you encounter issues, consider using a smaller model or adjusting the Modal function's resource allocation.


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