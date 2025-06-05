import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
import logging
import time
from typing import Optional
# It's good practice to import specific exceptions if known, e.g.
# from requests.exceptions import ConnectionError
# from ollama import ResponseError # Assuming ollama client might raise this

    """
    An agent responsible for analyzing video transcripts to generate summaries
    and YouTube search terms using a LlamaIndex-configured Ollama LLM.
    """

    def __init__(self):
        """
        Initializes the agent and sets up the LlamaIndex LLM configuration.
        """
        self.logger = logging.getLogger(__name__)
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
            self.logger.error("MODEL_ENDPOINT environment variable not set for ContentAnalysisAgent.")
            raise ValueError("MODEL_ENDPOINT environment variable not set for ContentAnalysisAgent.")
        if not model_name:
            self.logger.warning("MODEL_NAME environment variable not set for ContentAnalysisAgent. Using a default or letting Ollama decide.")

        llm = Ollama(base_url=model_endpoint, model=model_name, request_timeout=120.0)
        Settings.llm = llm # Set as global for LlamaIndex
        self.logger.info("✅ ContentAnalysisAgent: LlamaIndex configured to use Ollama: endpoint=%s, model=%s", model_endpoint, model_name or 'default')

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 2, delay_seconds: int = 5) -> Optional[str]:
        """
        Calls the LLM with a given prompt and retries on failure.

        Args:
            prompt: The prompt to send to the LLM.
            max_retries: Maximum number of retry attempts.
            delay_seconds: Delay between retries in seconds.

        Returns:
            The LLM response text if successful, otherwise None.
        """
        for attempt in range(max_retries):
            try:
                self.logger.info("LLM call attempt %d/%d", attempt + 1, max_retries)
                response = Settings.llm.complete(prompt)
                return response.text.strip()
            except Exception as e: # Catch a broad range of exceptions initially
                self.logger.error(
                    "LLM call failed on attempt %d/%d: %s",
                    attempt + 1,
                    max_retries,
                    e,
                    exc_info=True # Include stack trace for the error
                )
                if attempt < max_retries - 1:
                    self.logger.info("Waiting %d seconds before next retry...", delay_seconds)
                    time.sleep(delay_seconds)
                else:
                    self.logger.error("All LLM retry attempts failed for prompt.")
                    return None
        return None # Should be unreachable if loop completes, but as a safeguard

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
        import json
        import re

        if not raw_response:
            return []

        terms = []
        try:
            # Attempt to parse as JSON first
            # LLMs might sometimes add introductory text before the JSON, try to strip it.
            # Look for the start of a JSON list '['
            json_start_index = raw_response.find('[')
            if json_start_index != -1:
                raw_json = raw_response[json_start_index:]
                # Also look for the end of a JSON list ']' for more robust slicing
                json_end_index = raw_json.rfind(']')
                if json_end_index != -1:
                    raw_json = raw_json[:json_end_index+1]

                parsed_terms = json.loads(raw_json)
                if isinstance(parsed_terms, list):
                    for term in parsed_terms:
                        if isinstance(term, str):
                            cleaned_term = term.strip().strip('"\'')
                            if cleaned_term and len(cleaned_term) > 2:
                                terms.append(cleaned_term)
            else: # If no '[' found, assume it might not be JSON
                raise json.JSONDecodeError("No JSON list found", raw_response, 0)

        except json.JSONDecodeError:
            self.logger.warning("⚠️ JSON parsing failed for search terms, falling back to comma-separated parsing. Raw response: %s", raw_response)
            # Fallback to comma-separated parsing (simplified from original)
            # Remove common LLM response prefixes/suffixes
            cleaned_response = raw_response
            
            prefixes_to_remove = [
                "here are", "the search terms are:", "search terms:",
                "youtube search terms:", "based on the transcript",
                "here is a json list of strings for the youtube search terms:", # For models that explain before JSON
                "here are 5 targeted youtube search terms in a json list of strings format:",
                "```json"
            ]
            for prefix in prefixes_to_remove:
                if cleaned_response.lower().startswith(prefix.lower()):
                    cleaned_response = cleaned_response[len(prefix):].strip()

            # Remove patterns like "5 targeted YouTube search terms:"
            cleaned_response = re.sub(r'^\d+\s+[^:]+:\s*', '', cleaned_response)
            # Remove any remaining "json" or "```" artifacts if JSON parsing failed
            cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()

            # Remove newlines and extra whitespace
            cleaned_response = ' '.join(cleaned_response.split())

            raw_terms = cleaned_response.split(',')
            for term_str in raw_terms:
                term = term_str.strip().strip('"\'')
                term = re.sub(r'^\d+\.\s*', '', term) # Remove "1. ", "2. "
                term = re.sub(r'^[-•*]\s*', '', term)  # Remove "- ", "• ", "* "
                if term and len(term) > 2:
                    terms.append(term)
        
        # Clean all extracted terms one last time (e.g., ensure no surrounding list brackets/quotes if parsing was messy)
        final_cleaned_terms = []
        for term in terms:
            cleaned = term.strip().strip('"\'[]') # More aggressive stripping here
            if cleaned and len(cleaned) > 2:
                 final_cleaned_terms.append(cleaned)

        return final_cleaned_terms[:3] # Return max 3 terms

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
            self.logger.error("🚨 LLM not initialized in ContentAnalysisAgent. Re-initializing...")
            self._initialize_llamaindex_ollama() # Attempt re-init

        if not transcript_text or not transcript_text.strip():
            self.logger.warning("⚠️ Transcript text is empty. ContentAnalysisAgent cannot perform analysis.")
            return {
                "video_summary": "Transcript was empty or not provided.",
                "search_terms": []
            }

        self.logger.info("🧠 ContentAnalysisAgent: Performing content analysis on transcript (%d chars)...", len(transcript_text))

        # --- 1. Video Summary Creation ---
        summary_prompt_template = (
            "You are an expert content analyst. Based on the following video transcript, "
            "generate a comprehensive summary. The summary should identify key topics, "
            "themes, and main discussion points. Make it concise yet informative. "
            "The summary should be engaging, around 3-4 sentences, and suitable for "
            "inclusion in a YouTube video description to quickly grab viewer interest.\n\n"
            "Transcript:\n"
            "---------------------\n"
            "{transcript}\n"
            "---------------------\n"
            "Engaging YouTube Summary (3-4 sentences):"
        )
        summary_prompt = summary_prompt_template.format(transcript=transcript_text)

        self.logger.info("🔄 ContentAnalysisAgent: Generating video summary with retry...")
        summary_response_text = self._call_llm_with_retry(summary_prompt)

        video_summary = "Failed to generate summary after multiple attempts." # Default error message
        if summary_response_text is not None and summary_response_text.strip() and len(summary_response_text.strip()) > 10:
            video_summary = summary_response_text
            self.logger.info("✅ ContentAnalysisAgent: Summary generated successfully.")
        elif summary_response_text is not None: # LLM returned something, but it's too short/invalid
            self.logger.warning("⚠️ ContentAnalysisAgent: Summary generation resulted in invalid or too short summary: '%s'", summary_response_text)
            video_summary = "Summary could not be generated or was invalid."
        else: # All retries failed
             self.logger.error("❌ ContentAnalysisAgent: Failed to generate summary after all retries.")
             # video_summary is already set to the error message

        # --- 2. Search Term Generation ---
        search_term_prompt_template = (
            "You are a YouTube content strategy expert. Based on the following video transcript, "
            "generate exactly 5 targeted YouTube search terms that actual users would type to find this content. "
            "These terms should be highly relevant to the video's main topics and optimized for discoverability. "
            "Return ONLY a JSON list of strings, like this: "
            '`["search term 1", "search term 2", "search term 3", "search term 4", "search term 5"]`\n\n'
            "Do not include any other text, explanations, or markdown formatting around the JSON list.\n\n"
            "Transcript:\n"
            "---------------------\n"
            "{transcript}\n"
            "---------------------\n"
            "JSON list of 5 Search Terms:"
        )
        search_term_prompt = search_term_prompt_template.format(transcript=transcript_text)

        self.logger.info("🔄 ContentAnalysisAgent: Generating search terms with retry...")
        search_terms_raw = self._call_llm_with_retry(search_term_prompt)
        search_terms_list = []

        if search_terms_raw is not None and search_terms_raw.strip():
            search_terms_list = self._parse_search_terms(search_terms_raw) # This method now uses self.logger for its warnings
            if search_terms_list:
                self.logger.info("✅ ContentAnalysisAgent: Search terms generated and parsed: %s", search_terms_list)
            else:
                self.logger.warning("⚠️ ContentAnalysisAgent: Search terms generated but failed to parse or yielded no valid terms from raw: %s", search_terms_raw)
        else:
            self.logger.error("❌ ContentAnalysisAgent: Failed to generate search terms after all retries or got empty response.")
            # search_terms_list remains empty

        return {
            "video_summary": video_summary,
            "search_terms": search_terms_list[:3]  # Ensure only up to 3 terms
        }

# For basic testing of this module
if __name__ == '__main__':
    # Basic logging for the test script itself
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main_logger = logging.getLogger(__name__)

    main_logger.info("🧪 Running test for ContentAnalysisAgent...")

    load_dotenv() # Ensure .env is loaded for the test script
    model_endpoint = os.getenv("MODEL_ENDPOINT")
    model_name = os.getenv("MODEL_NAME")

    if not model_endpoint or not model_name:
        main_logger.error("🔴 MODEL_ENDPOINT and/or MODEL_NAME environment variables not set. Skipping ContentAnalysisAgent test.")
        main_logger.info("Ensure .env file exists and these variables are set (e.g., MODEL_ENDPOINT=http://localhost:11434, MODEL_NAME=llama3)")
    else:
        main_logger.info("✅ MODEL_ENDPOINT and MODEL_NAME are set. Proceeding with agent test.")
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
            main_logger.info("\n--- Agent Analysis Results ---")
            main_logger.info("Summary: %s", analysis_results['video_summary'])
            main_logger.info("Search Terms: %s", analysis_results['search_terms'])
        else:
            main_logger.error("❌ LLM not initialized by agent. Skipping analysis test.")

    except ValueError as ve:
            main_logger.error("Configuration Error for Agent: %s", ve, exc_info=True)
        except Exception as e:
            main_logger.exception("An unexpected error occurred during agent testing: %s", e)