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

# huggingface-hackathon-project

## Goal
This project creates an AI-powered agent that automates YouTube content optimization by processing uploaded videos to generate transcriptions, enhanced titles, descriptions, and multiple thumbnail variations. The system uses advanced AI models including Whisper for transcription and content generation models for creating engaging YouTube-ready content.

For detailed project information, workflow process, and technical specifications, see [documentation.md](documentation/project-goals.md).

## Requirements
- Python 3.12.9

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

3. **Run the Gradio application**:
   ```bash
   python app.py
   ```
   
   The application will be available at: http://localhost:7860

4. **Alternative - Jupyter Notebook for development**:
   ```bash
   jupyter notebook code-testing.ipynb
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

## Project Structure

```
huggingface-hackathon-project/
├── app.py                    # Main Gradio application (HF entry point)
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── documentation/           # Detailed project docs
│   └── project-goals.md    # Project workflow and technical details
├── code-testing.ipynb      # Development notebook
└── .env                    # Environment variables
```

## Packages Used
- gradio==5.32.1
- modal==1.0.2

## Features

### Current Interface
- **Video Upload**: Drag and drop video file input
- **Processing Pipeline**: Visual workflow display
- **Results Display**: 
  - Generated titles and descriptions
  - Three thumbnail variations
  - Debug information panel
- **Responsive Design**: Modern UI with tabbed interface

### Planned Implementation
1. **Audio Extraction** - Separate audio from uploaded video
2. **Whisper Transcription** - Convert speech to text
3. **Content Analysis** - AI-generated search terms and descriptions  
4. **Content Enhancement** - YouTube research integration
5. **Thumbnail Generation** - Multiple style variations