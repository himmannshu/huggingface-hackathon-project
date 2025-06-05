import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

class ContentAnalysisAgent:
    """
    An agent responsible for analyzing video transcripts to generate summaries
    and YouTube search terms using a LlamaIndex-configured Ollama LLM.
    """

    def __init__(self):
        """
        Initializes the agent and sets up the LlamaIndex LLM configuration.
        """
        self._initialize_llamaindex_ollama()

    def _initialize_llamaindex_ollama(self):
        """
        Initializes LlamaIndex settings to use Ollama with endpoint and model
        from environment variables.
        """
        load_dotenv()
        model_endpoint = os.getenv("MODEL_ENDPOINT")
        model_name = os.getenv("MODEL_NAME")

        if not model_endpoint:
            raise ValueError("MODEL_ENDPOINT environment variable not set for ContentAnalysisAgent.")
        if not model_name:
            print("Warning: MODEL_NAME environment variable not set for ContentAnalysisAgent. Using a default or letting Ollama decide.")

        llm = Ollama(base_url=model_endpoint, model=model_name, request_timeout=120.0)
        Settings.llm = llm # Set as global for LlamaIndex
        print(f"✅ ContentAnalysisAgent: LlamaIndex configured to use Ollama: endpoint={model_endpoint}, model={model_name or 'default'}")

    def _parse_search_terms(self, raw_response: str) -> list:
        """
        Parse and clean the LLM response to extract clean search terms.
        
        Args:
            raw_response: Raw text response from the LLM
            
        Returns:
            List of cleaned search terms (max 3)
        """
        if not raw_response:
            return []
        
        # Remove common LLM response prefixes/suffixes
        cleaned_response = raw_response
        
        # Remove introductory phrases that LLMs sometimes add
        prefixes_to_remove = [
            "here are",
            "the search terms are:",
            "search terms:",
            "youtube search terms:",
            "comma-separated",
            "based on the transcript",
        ]
        
        # Remove prefixes and clean up colons, numbers, etc.
        for prefix in prefixes_to_remove:
            if cleaned_response.lower().startswith(prefix):
                cleaned_response = cleaned_response[len(prefix):].strip()
                break
        
        # Remove patterns like "3 targeted YouTube search terms:" from the beginning
        import re
        cleaned_response = re.sub(r'^\d+\s+[^:]+:\s*', '', cleaned_response)
        cleaned_response = re.sub(r'^[^:]+:\s*', '', cleaned_response)
        
        # Remove newlines and extra whitespace
        cleaned_response = ' '.join(cleaned_response.split())
        
        # Split by commas and clean each term
        terms = []
        for term in cleaned_response.split(','):
            term = term.strip()
            # Remove quotes if present
            term = term.strip('"\'')
            # Remove numbered prefixes like "1. " or "- "
            import re
            term = re.sub(r'^\d+\.\s*', '', term)
            term = re.sub(r'^[-•]\s*', '', term)
            
            if term and len(term) > 2:  # Only add meaningful terms
                terms.append(term)
        
        return terms[:3]  # Return max 3 terms

    def analyze_transcript(self, transcript_text: str) -> dict:
        """
        Analyzes video transcript to generate a summary and YouTube search terms.

        Args:
            transcript_text: The full text of the video transcription.

        Returns:
            A dictionary containing:
            - "video_summary": A string summary of the video.
            - "search_terms": A list of 1-3 YouTube search terms.
        """
        if not Settings.llm:
            # This should ideally not be hit if constructor ran successfully
            print("🚨 LLM not initialized in ContentAnalysisAgent. Re-initializing...")
            self._initialize_llamaindex_ollama()

        if not transcript_text or not transcript_text.strip():
            print("⚠️ Transcript text is empty. ContentAnalysisAgent cannot perform analysis.")
            return {
                "video_summary": "Transcript was empty or not provided.",
                "search_terms": []
            }

        print(f"🧠 ContentAnalysisAgent: Performing content analysis on transcript ({len(transcript_text)} chars)...")

        # --- 1. Video Summary Creation ---
        summary_prompt_template = (
            "You are an expert content analyst. Based on the following video transcript, "
            "generate a comprehensive summary. The summary should identify key topics, "
            "themes, and main discussion points. Make it concise yet informative, suitable for "
            "understanding the video's core message quickly.\n\n"
            "Transcript:\n"
            "---------------------\n"
            "{transcript}\n"
            "---------------------\n"
            "Comprehensive Summary:"
        )
        summary_prompt = summary_prompt_template.format(transcript=transcript_text)

        print("🔄 ContentAnalysisAgent: Generating video summary...")
        video_summary = "Error generating summary." # Default in case of failure
        try:
            summary_response = Settings.llm.complete(summary_prompt)
            video_summary = summary_response.text.strip()
            print("✅ ContentAnalysisAgent: Summary generated.")
        except Exception as e:
            print(f"❌ ContentAnalysisAgent: Error during summary generation: {e}")

        # --- 2. Search Term Generation ---
        search_term_prompt_template = (
            "You are a YouTube content strategy expert. Based on the following video transcript, "
            "generate exactly 3 targeted YouTube search terms that actual users would type to find this content. "
            "These terms should be highly relevant to the video's main topics and optimized for discoverability. "
            "Return ONLY the search terms separated by commas, nothing else. "
            "Example output: MacBook Pro 2024 review, latest MacBook Pro specs, MacBook Pro M3 features\n\n"
            "Transcript:\n"
            "---------------------\n"
            "{transcript}\n"
            "---------------------\n"
            "Search Terms:"
        )
        search_term_prompt = search_term_prompt_template.format(transcript=transcript_text)

        print("🔄 ContentAnalysisAgent: Generating search terms...")
        search_terms_list = []
        try:
            search_terms_response = Settings.llm.complete(search_term_prompt)
            search_terms_raw = search_terms_response.text.strip()
            
            # Clean and parse the response
            search_terms_list = self._parse_search_terms(search_terms_raw)
            print(f"✅ ContentAnalysisAgent: Search terms generated: {search_terms_list}")
        except Exception as e:
            print(f"❌ ContentAnalysisAgent: Error during search term generation: {e}")

        return {
            "video_summary": video_summary,
            "search_terms": search_terms_list[:3]  # Ensure only up to 3 terms
        }

# For basic testing of this module
if __name__ == '__main__':
    print("🧪 Running test for ContentAnalysisAgent...")

    # Create a dummy .env file for testing if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# MODEL_ENDPOINT=http://localhost:11434\n")
            f.write("# MODEL_NAME=mistral\n")
        print("ℹ️ Created a dummy .env file. Please edit it with your Ollama details.")
        print("Ensure MODEL_ENDPOINT is uncommented and set, e.g., http://your_ollama_server_ip:11434")
        print("Ensure MODEL_NAME is uncommented and set, e.g., llama3 or mistral")

    try:
        agent = ContentAnalysisAgent() # Initializes LLM
        
        sample_transcript = (
            "Hello everyone, and welcome back to AI Insights! Today, we're diving deep into "
            "the latest advancements in Retrieval Augmented Generation, or RAG. We'll explore "
            "how RAG is revolutionizing the way LLMs access and utilize external knowledge, "
            "making them more accurate and context-aware. We'll cover new indexing strategies, "
            "advanced retrieval techniques, and how to fine-tune RAG pipelines for specific tasks. "
            "We'll also look at some real-world applications and the future potential of this technology. "
            "So, grab your notebooks, and let's get started on understanding the future of AI with RAG!"
        )
        
        if Settings.llm: # Proceed only if LLM was successfully initialized by the agent
            analysis_results = agent.analyze_transcript(sample_transcript)
            print("\n--- Agent Analysis Results ---")
            print(f"Summary: {analysis_results['video_summary']}")
            print(f"Search Terms: {analysis_results['search_terms']}")
        else:
            print("❌ LLM not initialized by agent. Skipping analysis test.")

    except ValueError as ve:
        print(f"Configuration Error for Agent: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during agent testing: {e}") 